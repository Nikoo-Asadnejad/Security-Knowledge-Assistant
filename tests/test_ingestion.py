import copy
import json
from contextlib import closing
from unittest.mock import patch

import pytest
from qdrant_client import QdrantClient, models

from security_rag.ingestion.chunk_models import ChunkingConfig
from security_rag.ingestion.chunker import MarkdownChunker
from security_rag.ingestion.pipeline import IngestionError, IngestionPipeline
from security_rag.parent_store.models import Revision, document_revision
from security_rag.vector_store.qdrant_store import QdrantStore


class MemoryParents:
    """Transactional state stand-in; actual SQL is covered by PostgreSQL integration tests."""

    def __init__(self):
        self.parents = {}
        self.states = {}
        self.events = []
        self.config = None

    def ingestion_state(self, revision):
        return self.states.get(revision.key)

    def prepare_revision(self, chunks):
        revision = document_revision(chunks)
        existing_ids = {revision.document_id} | {
            p.document_id
            for p in self.parents.values()
            if p.file_name == chunks.parents[0].file_name
        }
        if any(key[0] in existing_ids and key != revision.key for key in self.states):
            raise ValueError("Document updates are deferred to Phase 8")
        self.events.append("prepare")
        self.states[revision.key] = {
            "status": "processing",
            "expected_parents": len(chunks.parents),
            "expected_children": len(chunks.children),
        }
        self.parents.update({p.id: p for p in chunks.parents})

    def verify_parents(self, chunks):
        revision = document_revision(chunks)
        actual = {
            key: parent for key, parent in self.parents.items() if Revision.of(parent) == revision
        }
        return actual == {p.id: p for p in chunks.parents}

    def resolve_parents(self, references):
        return {
            ref.parent_id: self.parents[ref.parent_id]
            for ref in references
            if ref.parent_id in self.parents
            and Revision.of(self.parents[ref.parent_id]).key == ref.key
        }

    def finish_revision(self, chunks, sparse_config_hash):
        self.events.append("finish")
        self.states[document_revision(chunks).key].update(
            status="completed", sparse_status="completed", sparse_config_hash=sparse_config_hash
        )

    def sparse_config(self):
        return self.config

    def save_sparse_config(self, config):
        if self.config is not None and self.config != config:
            raise ValueError("Sparse configuration updates are deferred to Phase 8")
        self.config = config

    def statistics(self):
        return [
            dict(
                zip(("document_id", "document_hash", "chunking_config_hash"), key, strict=True),
                **state,
            )
            for key, state in self.states.items()
        ]


class FakeEmbedder:
    def __init__(self, parents, batch_size):
        self.parents = parents
        self.batch_size = batch_size
        self.calls = 0

    def embed(self, children):
        self.calls += 1
        assert 0 < len(children) <= self.batch_size
        assert all(child.parent_id in self.parents.parents for child in children)
        assert all(
            self.parents.states[Revision.of(c).key]["status"] == "processing" for c in children
        )
        return [[1.0] + [0.0] * 383 for _ in children]


@pytest.fixture
def pipeline(tokenizer, sparse_encoder):
    parents = MemoryParents()
    with closing(QdrantClient(":memory:")) as client:
        children = QdrantStore(client, "isolated_test", batch_size=2)
        children.ensure_collection()
        yield IngestionPipeline(
            MarkdownChunker(tokenizer),
            parents,
            children,
            FakeEmbedder(parents, 2),
            sparse_encoder,
            batch_size=2,
        )


@pytest.fixture
def source(make_document):
    return make_document(
        "### Description\nalpha beta.\n\n#### Detail\ngamma delta.\n\n"
        "### Mitigation\nalpha gamma.\n"
    )


def test_ingestion_writes_both_vectors_on_every_run_without_duplicates(pipeline, source):
    first = pipeline.ingest_document(source)
    calls = pipeline.embedder.calls
    second = pipeline.ingest_document(source)
    assert first.parents == second.parents == 2
    assert first.children == second.children == 3
    assert pipeline.embedder.calls == 2 * calls
    assert pipeline.children.statistics()["children"] == 3
    chunks = pipeline.chunker.chunk(source)
    config_hash = pipeline.sparse_encoder.config.fingerprint
    assert pipeline.parents.verify_parents(chunks)
    assert pipeline.children.verify_children(chunks, config_hash)
    records = pipeline.children.client.retrieve(
        "isolated_test", [str(child.id) for child in chunks.children], with_vectors=True
    )
    assert all(set(record.vector) == {"", "bm25"} for record in records)
    assert first.sparse_nonzero_terms > 0
    assert all(record.payload["sparse_config_hash"] == config_hash for record in records)


@pytest.mark.parametrize("change", ["source", "config", "title"])
def test_document_updates_are_deferred_without_changing_stored_data(
    pipeline, source, make_document, change
):
    pipeline.ingest_document(source)
    before_parents = copy.deepcopy(pipeline.parents.parents)
    before_states = copy.deepcopy(pipeline.parents.states)
    before_children = pipeline.children.client.scroll("isolated_test", with_vectors=True)[0]
    changed = source
    if change == "source":
        changed = make_document("### Description\nNew alpha content.\n")
    elif change == "title":
        changed = make_document("### Description\nNew alpha content.\n", title="Updated Title")
    else:
        pipeline.chunker = MarkdownChunker(
            pipeline.chunker.tokenizer, ChunkingConfig(child_overlap_tokens=30)
        )
    with pytest.raises(ValueError, match="Phase 8"):
        pipeline.ingest_document(changed)
    assert pipeline.parents.parents == before_parents
    assert pipeline.parents.states == before_states
    assert pipeline.children.client.scroll("isolated_test", with_vectors=True)[0] == before_children


@pytest.mark.parametrize("boundary", ["embed", "sparse", "upsert", "after_upsert", "finish"])
def test_initial_ingestion_failure_never_marks_partial_records_complete(pipeline, source, boundary):
    target, method = {
        "embed": (pipeline.embedder, "embed"),
        "sparse": (pipeline.sparse_encoder, "encode"),
        "upsert": (pipeline.children, "upsert"),
        "after_upsert": (pipeline.children, "upsert"),
        "finish": (pipeline.parents, "finish_revision"),
    }[boundary]
    original = getattr(target, method)

    def fail(*args, **kwargs):
        if boundary == "after_upsert":
            original(*args, **kwargs)
        raise RuntimeError("interrupted")

    with patch.object(target, method, side_effect=fail), pytest.raises(RuntimeError):
        pipeline.ingest_document(source)
    assert {state["status"] for state in pipeline.parents.states.values()} == {"processing"}
    pipeline.ingest_document(source)
    chunks = pipeline.chunker.chunk(source)
    assert pipeline.children.verify_children(chunks, pipeline.sparse_encoder.config.fingerprint)
    assert pipeline.parents.verify_parents(chunks)


def test_invalid_persisted_reference_blocks_child_writes(pipeline, source, caplog):
    with patch.object(pipeline.parents, "resolve_parents", return_value={}):
        with pytest.raises(ValueError, match="invalid persisted parent"):
            pipeline.ingest_document(source)
    assert pipeline.embedder.calls == 0
    assert pipeline.children.statistics()["children"] == 0
    assert "invalid_parent_references" in caplog.text
    assert {s["status"] for s in pipeline.parents.states.values()} == {"processing"}


def test_failed_stored_verification_leaves_processing_state(pipeline, source):
    with patch.object(pipeline.children, "verify_children", return_value=False):
        with pytest.raises(ValueError, match="verification failed"):
            pipeline.ingest_document(source)
    assert {s["status"] for s in pipeline.parents.states.values()} == {"processing"}


def test_corpus_report_failures_do_not_prevent_other_documents_from_finishing(
    pipeline, settings, markdown, caplog
):
    from security_rag.ingestion.manifest import DOCUMENT_FILES

    settings.data_dir.mkdir()
    for name in DOCUMENT_FILES[1:]:
        (settings.data_dir / name).write_bytes(markdown(name))
    with pytest.raises(IngestionError) as exc:
        pipeline.ingest(settings)
    report = exc.value.report
    assert (report.processed, report.failed) == (9, 1)
    assert report.failures == [{"file_name": DOCUMENT_FILES[0], "error": "FileNotFoundError"}]
    assert all(json.loads(line.message)["event"] == "document_failed" for line in caplog.records)
    (settings.data_dir / DOCUMENT_FILES[0]).write_bytes(markdown(DOCUMENT_FILES[0]))
    retry = pipeline.ingest(settings)
    assert (retry.processed, retry.failed) == (10, 0)


def test_incompatible_collection_is_rejected():
    with closing(QdrantClient(":memory:")) as client:
        client.create_collection(
            "wrong", vectors_config=models.VectorParams(size=10, distance=models.Distance.DOT)
        )
        with pytest.raises(ValueError, match="384-dimensional cosine"):
            QdrantStore(client, "wrong").ensure_collection()
