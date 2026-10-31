"""Retryable offline ingestion. PostgreSQL parents always precede child writes."""

import json
import logging
from contextlib import closing
from time import perf_counter

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from security_rag.config import Settings
from security_rag.embeddings.embedder import Embedder
from security_rag.ingestion.chunker import MarkdownChunker
from security_rag.ingestion.loader import Document, load_document
from security_rag.ingestion.manifest import DOCUMENT_FILES
from security_rag.ingestion.tokenizer import ChunkTokenizer
from security_rag.parent_store.models import ParentReference, document_revision
from security_rag.parent_store.postgres_store import PostgresStore
from security_rag.vector_store.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


class IngestionError(RuntimeError):
    """The report preserves completed documents and identifies retryable failures."""

    def __init__(self, report: "IngestionReport") -> None:
        self.report = report
        super().__init__(f"{report.failed} document(s) failed; rerun ingest to retry")


class DocumentReport(BaseModel):
    document_id: str
    skipped: bool = False
    parents: int
    children: int
    min_child_tokens: int
    max_child_tokens: int
    embedding_seconds: float = 0
    parent_persistence_seconds: float = 0
    indexing_seconds: float = 0
    invalid_parent_references: int = 0


class IngestionReport(BaseModel):
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    documents: list[DocumentReport] = Field(default_factory=list)
    failures: list[dict[str, str]] = Field(default_factory=list)


class IngestionPipeline:
    def __init__(
        self,
        chunker: MarkdownChunker,
        parents: PostgresStore,
        children: QdrantStore,
        embedder: Embedder,
        *,
        batch_size: int = 32,
    ) -> None:
        if batch_size < 1:
            raise ValueError("Ingestion batch size must be positive")
        self.chunker = chunker
        self.parents = parents
        self.children = children
        self.embedder = embedder
        self.batch_size = batch_size

    def ingest_document(self, document: Document) -> DocumentReport:
        chunks = self.chunker.chunk(document)
        revision = document_revision(chunks)
        lengths = [self.chunker.tokenizer.count(child.embedding_text) for child in chunks.children]
        if max(lengths) > self.chunker.config.child_max_tokens:
            raise ValueError("Child embedding token budget exceeded")
        report = DocumentReport(
            document_id=revision.document_id,
            parents=len(chunks.parents),
            children=len(chunks.children),
            min_child_tokens=min(lengths),
            max_child_tokens=max(lengths),
        )
        state = self.parents.ingestion_state(revision)
        if (
            state is not None
            and state["status"] == "completed"
            and state["expected_parents"] == len(chunks.parents)
            and state["expected_children"] == len(chunks.children)
            and self.parents.verify_parents(chunks, all_revisions=True)
            and self.children.verify_children(chunks, all_revisions=True)
        ):
            report.skipped = True
            return report

        start = perf_counter()
        self.parents.prepare_revision(chunks)
        if not self.parents.verify_parents(chunks, allow_extra=True):
            raise ValueError("Persisted parents do not match the new revision")
        report.parent_persistence_seconds = perf_counter() - start
        for offset in range(0, len(chunks.children), self.batch_size):
            batch = chunks.children[offset : offset + self.batch_size]
            references = [ParentReference.model_validate(child.model_dump()) for child in batch]
            persisted = self.parents.resolve_parents(references)
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
            self.children.upsert(batch, vectors)
            report.indexing_seconds += perf_counter() - start

        start = perf_counter()
        if not self.parents.verify_parents(
            chunks, allow_extra=True
        ) or not self.children.verify_children(chunks, allow_extra=True):
            raise ValueError("Stored revision verification failed before cleanup")
        self.children.remove_obsolete(chunks)
        if not self.children.verify_children(chunks, all_revisions=True):
            raise ValueError("Obsolete child cleanup verification failed")
        self.parents.finish_revision(chunks)
        report.indexing_seconds += perf_counter() - start
        return report

    def ingest(self, settings: Settings) -> IngestionReport:
        self.children.ensure_collection()
        report = IngestionReport()
        for name in DOCUMENT_FILES:
            try:
                # Do not use load_documents(): keep only one source and chunk result in memory.
                result = self.ingest_document(load_document(settings.data_dir / name))
            except Exception as exc:
                report.failed += 1
                # Store the type, not driver error strings that may include credentials.
                failure = {"file_name": name, "error": type(exc).__name__}
                report.failures.append(failure)
                logger.error(json.dumps({"event": "document_failed", **failure}))
            else:
                report.documents.append(result)
                if result.skipped:
                    report.skipped += 1
                else:
                    report.processed += 1
                logger.info(json.dumps({"event": "document_ingested", **result.model_dump()}))
        if report.failed:
            raise IngestionError(report)
        return report


def ingest_documents(settings: Settings) -> IngestionReport:
    # Fail for missing sources before initializing models or opening databases.
    for name in DOCUMENT_FILES:
        if not (settings.data_dir / name).is_file():
            raise ValueError(f"Missing document {name}; run download first")
    tokenizer = ChunkTokenizer.load(
        settings.embedding_model, path=settings.tokenizer_path, revision=settings.tokenizer_revision
    )
    parents = PostgresStore.from_settings(settings)
    try:
        with closing(QdrantClient(url=str(settings.qdrant_url), timeout=30)) as client:
            return IngestionPipeline(
                MarkdownChunker(tokenizer),
                parents,
                QdrantStore(client, settings.qdrant_collection, settings.ingestion_batch_size),
                Embedder(settings, tokenizer),
                batch_size=settings.ingestion_batch_size,
            ).ingest(settings)
    finally:
        parents.close()
