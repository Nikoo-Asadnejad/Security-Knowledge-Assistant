"""Initial offline ingestion: persist parents, then encode and store both child vectors."""

import json
import logging
from collections.abc import Callable, Iterable
from functools import partial
from time import perf_counter
from uuid import UUID

from pydantic import BaseModel, Field

from security_rag.core.application.ports import (
    ChildIndex,
    ChildIndexEntry,
    Chunker,
    DenseEncoder,
    ParentStore,
    SparseEncoder,
)
from security_rag.core.domain.chunks import ParentChunk
from security_rag.core.domain.documents import Document
from security_rag.core.domain.lexical import identifiers, lexical_text
from security_rag.core.domain.revisions import ParentReference, Revision, document_revision

logger = logging.getLogger(__name__)


class IngestionBusy(RuntimeError):
    """An ingestion run already holds the writer lock."""


class IngestionError(RuntimeError):
    """The report preserves completed documents and identifies retryable failures."""

    def __init__(self, report: "IngestionReport") -> None:
        self.report = report
        super().__init__(f"{report.failed} document(s) failed; rerun ingest to retry")


class DocumentReport(BaseModel):
    document_id: str
    parents: int
    children: int
    min_child_tokens: int
    max_child_tokens: int
    embedding_seconds: float = 0
    parent_persistence_seconds: float = 0
    indexing_seconds: float = 0
    invalid_parent_references: int = 0
    sparse_config_hash: str
    sparse_seconds: float = 0
    sparse_nonzero_terms: int = 0


class IngestionReport(BaseModel):
    processed: int = 0
    failed: int = 0
    documents: list[DocumentReport] = Field(default_factory=list)
    failures: list[dict[str, str]] = Field(default_factory=list)


class IngestionPipeline:
    def __init__(
        self,
        chunker: Chunker,
        parents: ParentStore,
        children: ChildIndex,
        embedder: DenseEncoder,
        sparse_encoder: SparseEncoder,
        *,
        batch_size: int = 32,
    ) -> None:
        if batch_size < 1:
            raise ValueError("Ingestion batch size must be positive")
        self.chunker = chunker
        self.parents = parents
        self.children = children
        self.embedder = embedder
        self.sparse_encoder = sparse_encoder
        self.batch_size = batch_size

    def ingest_document(self, document: Document) -> DocumentReport:
        chunks = self.chunker.chunk(document)
        revision = document_revision(chunks)
        lengths = [self.chunker.child_token_count(child) for child in chunks.children]
        if max(lengths) > self.chunker.child_max_tokens:
            raise ValueError("Child embedding token budget exceeded")
        report = DocumentReport(
            document_id=revision.document_id,
            parents=len(chunks.parents),
            children=len(chunks.children),
            min_child_tokens=min(lengths),
            max_child_tokens=max(lengths),
            sparse_config_hash=self.sparse_encoder.config.fingerprint,
        )
        start = perf_counter()
        self.parents.prepare_revision(
            chunks, check_existing=partial(check_initial_revision, revision)
        )
        if not self.parents.verify_parents(chunks):
            raise ValueError("Persisted parents do not match the new revision")
        report.parent_persistence_seconds = perf_counter() - start
        entries: list[ChildIndexEntry] = []
        for offset in range(0, len(chunks.children), self.batch_size):
            batch = chunks.children[offset : offset + self.batch_size]
            references = [ParentReference.model_validate(child.model_dump()) for child in batch]
            persisted = resolve_parents(self.parents.get_parents, references)
            invalid = sum(ref.parent_id not in persisted for ref in references)
            if invalid:
                logger.error(
                    json.dumps(
                        {
                            "event": "invalid_parent_references",
                            "count": invalid,
                            "document_id": revision.document_id,
                        }
                    )
                )
                raise ValueError(f"{invalid} invalid persisted parent reference(s)")
            start = perf_counter()
            vectors = self.embedder.embed(batch)
            report.embedding_seconds += perf_counter() - start
            start = perf_counter()
            texts = [lexical_text(child) for child in batch]
            sparse_vectors = self.sparse_encoder.encode(texts)
            report.sparse_seconds += perf_counter() - start
            report.sparse_nonzero_terms += sum(len(vector.indices) for vector in sparse_vectors)
            start = perf_counter()
            batch_entries = tuple(
                ChildIndexEntry(child, tuple(identifiers(text)))
                for child, text in zip(batch, texts, strict=True)
            )
            self.children.upsert(batch_entries, vectors, sparse_vectors, report.sparse_config_hash)
            entries.extend(batch_entries)
            report.indexing_seconds += perf_counter() - start

        start = perf_counter()
        if not self.parents.verify_parents(chunks) or not self.children.verify_children(
            revision, tuple(entries), report.sparse_config_hash
        ):
            raise ValueError("Stored document verification failed")
        self.parents.finish_revision(chunks, report.sparse_config_hash)
        report.indexing_seconds += perf_counter() - start
        return report

    def ingest(self, names: Iterable[str], load: Callable[[str], Document]) -> IngestionReport:
        self.children.ensure_collection()
        report = IngestionReport()
        for name in names:
            try:
                result = self.ingest_document(load(name))
            except Exception as exc:
                report.failed += 1
                # Store the type, not driver error strings that may include credentials.
                failure = {"file_name": name, "error": type(exc).__name__}
                report.failures.append(failure)
                logger.error(json.dumps({"event": "document_failed", **failure}))
            else:
                report.documents.append(result)
                report.processed += 1
                logger.info(json.dumps({"event": "document_ingested", **result.model_dump()}))
        if report.failed:
            raise IngestionError(report)
        return report


def check_initial_revision(revision: Revision, existing: tuple[Revision, ...]) -> None:
    if any(previous != revision for previous in existing):
        raise ValueError("Document updates are deferred to Phase 8")


def resolve_parents(
    fetch: Callable[[list[UUID]], dict[UUID, ParentChunk]], references: list[ParentReference]
) -> dict[UUID, ParentChunk]:
    parents = fetch(list({reference.parent_id for reference in references}))
    return {
        ref.parent_id: parents[ref.parent_id]
        for ref in references
        if ref.parent_id in parents and Revision.of(parents[ref.parent_id]).key == ref.key
    }
