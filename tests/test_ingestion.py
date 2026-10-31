import copy
import json
from contextlib import closing
from unittest.mock import patch
from uuid import uuid4

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

    def ingestion_state(self, revision):
        return self.states.get(revision.key)

    def prepare_revision(self, chunks):
        revision = document_revision(chunks)
        self.events.append("prepare")
        self.states[revision.key] = {
            "status": "processing",
            "expected_parents": len(chunks.parents),
            "expected_children": len(chunks.children),
        }
        self.parents.update({p.id: p for p in chunks.parents})

    def verify_parents(self, chunks, *, all_revisions=False, allow_extra=False):
        revision = document_revision(chunks)
        actual = {
            key: parent
            for key, parent in self.parents.items()
            if (
                (
                    parent.document_id == revision.document_id
                    or parent.file_name == chunks.parents[0].file_name
                )
                if all_revisions
                else Revision.of(parent) == revision
            )
        }
        expected = {p.id: p for p in chunks.parents}
        if allow_extra:
            actual = {key: value for key, value in actual.items() if key in expected}
        return actual == expected

    def resolve_parents(self, references):
        return {
            ref.parent_id: self.parents[ref.parent_id]
            for ref in references
            if ref.parent_id in self.parents
            and Revision.of(self.parents[ref.parent_id]).key == ref.key
        }

    def finish_revision(self, chunks):
        revision = document_revision(chunks)
        self.events.append("finish")
        document_ids = {revision.document_id} | {
            parent.document_id
            for parent in self.parents.values()
            if parent.file_name == chunks.parents[0].file_name
        }
        self.parents = {
            key: parent
            for key, parent in self.parents.items()
            if parent.document_id not in document_ids or key in {p.id for p in chunks.parents}
        }
        self.states = {
            key: state
            for key, state in self.states.items()
            if key[0] not in document_ids or key == revision.key
        }
        self.states[revision.key]["status"] = "completed"


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
def pipeline(tokenizer):
    parents = MemoryParents()
    with closing(QdrantClient(":memory:")) as client:
        children = QdrantStore(client, "isolated_test", batch_size=2)
        children.ensure_collection()
        yield IngestionPipeline(
            MarkdownChunker(tokenizer), parents, children, FakeEmbedder(parents, 2), batch_size=2
        )


@pytest.fixture
def source(make_document):
    return make_document(
        "### Description\nalpha beta.\n\n#### Detail\ngamma delta.\n\n"
        "### Mitigation\nalpha gamma.\n"
    )


def test_repeated_ingestion_checks_references_and_skips_without_embedding(pipeline, source):
    first = pipeline.ingest_document(source)
    calls = pipeline.embedder.calls
    second = pipeline.ingest_document(source)
    assert first.parents == 2 and first.children == 3
    assert not first.skipped and second.skipped
    assert pipeline.embedder.calls == calls
    assert pipeline.children.statistics()["children"] == 3
    chunks = pipeline.chunker.chunk(source)
    assert pipeline.parents.verify_parents(chunks, all_revisions=True)
    assert pipeline.children.verify_children(chunks, all_revisions=True)
    record = pipeline.children.client.retrieve(
        "isolated_test", [str(chunks.children[0].id)], with_payload=True
    )[0]
    assert record.payload == chunks.children[0].model_dump(mode="json")
    assert "parent_hash" not in record.payload


@pytest.mark.parametrize("damage", ["parent_id", "id", "hash", "missing", "parent", "counts"])
def test_counts_alone_never_allow_a_damaged_revision_to_skip(pipeline, source, damage):
    pipeline.ingest_document(source)
    chunks = pipeline.chunker.chunk(source)
    child = chunks.children[0]
    if damage == "parent":
        pipeline.parents.parents.pop(child.parent_id)
    elif damage == "counts":
        pipeline.parents.states[Revision.of(child).key]["expected_children"] += 1
    elif damage == "missing":
        pipeline.children.client.delete("isolated_test", [str(child.id)])
    else:
        payload = child.model_dump(mode="json")
        payload[{"parent_id": "parent_id", "id": "id", "hash": "document_hash"}[damage]] = str(
            uuid4()
        )
        pipeline.children.client.overwrite_payload("isolated_test", payload, [str(child.id)])
    assert not pipeline.ingest_document(source).skipped
    assert pipeline.children.verify_children(chunks, all_revisions=True)
    assert pipeline.ingest_document(source).skipped


def test_same_count_wrong_point_id_is_repaired(pipeline, source):
    pipeline.ingest_document(source)
    child = pipeline.chunker.chunk(source).children[0]
    client = pipeline.children.client
    client.delete("isolated_test", [str(child.id)])
    client.upsert(
        "isolated_test",
        [
            models.PointStruct(
                id=str(uuid4()), vector=[1.0] + [0.0] * 383, payload=child.model_dump(mode="json")
            )
        ],
    )
    assert client.count("isolated_test").count == 3
    assert not pipeline.ingest_document(source).skipped
    assert pipeline.ingest_document(source).skipped


@pytest.mark.parametrize("change", ["source", "config", "title"])
def test_updates_remove_children_before_parents_including_legacy_points(
    pipeline, source, make_document, change
):
    pipeline.ingest_document(source)
    old_ids = set(pipeline.parents.parents)
    # Simulate a legacy point lacking parent_id and revision hashes.
    pipeline.children.client.upsert(
        "isolated_test",
        [
            models.PointStruct(
                id=str(uuid4()),
                vector=[1.0] + [0.0] * 383,
                payload={
                    "document_id": source.metadata.document_id,
                    "file_name": source.metadata.file_name,
                    "text": "legacy",
                },
            )
        ],
    )
    updated = source
    if change == "source":
        updated = make_document("### Description\nNew alpha content.\n")
    elif change == "title":
        updated = make_document("### Description\nNew alpha content.\n", title="Updated Title")
    else:
        pipeline.chunker = MarkdownChunker(
            pipeline.chunker.tokenizer, ChunkingConfig(child_overlap_tokens=30)
        )
    finish = pipeline.parents.finish_revision

    def assert_cleanup(chunks):
        assert old_ids <= pipeline.parents.parents.keys()
        assert pipeline.children.verify_children(chunks, all_revisions=True)
        finish(chunks)

    with patch.object(pipeline.parents, "finish_revision", side_effect=assert_cleanup):
        pipeline.ingest_document(updated)
    assert not old_ids & pipeline.parents.parents.keys()
    assert len(pipeline.parents.states) == 1
    assert pipeline.ingest_document(updated).skipped


@pytest.mark.parametrize(
    "boundary", ["prepare", "embed", "upsert", "after_upsert", "cleanup", "after_cleanup", "finish"]
)
def test_partial_write_and_cleanup_failures_leave_processing_state_and_retry_safely(
    pipeline, source, make_document, boundary
):
    pipeline.ingest_document(source)
    old_ids = set(pipeline.parents.parents)
    old_state = copy.deepcopy(pipeline.parents.states)
    updated = make_document("### Description\nnew content.\n\n#### Detail\nmore content.\n")
    chunks = pipeline.chunker.chunk(updated)
    targets = {
        "prepare": (pipeline.parents, "prepare_revision"),
        "embed": (pipeline.embedder, "embed"),
        "upsert": (pipeline.children, "upsert"),
        "after_upsert": (pipeline.children, "upsert"),
        "cleanup": (pipeline.children, "remove_obsolete"),
        "after_cleanup": (pipeline.children, "remove_obsolete"),
        "finish": (pipeline.parents, "finish_revision"),
    }
    target, method = targets[boundary]
    original = getattr(target, method)

    def fail(*args, **kwargs):
        if boundary.startswith("after_"):
            original(*args, **kwargs)
        raise RuntimeError("interrupted")

    with patch.object(target, method, side_effect=fail), pytest.raises(RuntimeError):
        pipeline.ingest_document(updated)
    assert old_ids <= pipeline.parents.parents.keys()
    if boundary == "prepare":
        assert pipeline.parents.states == old_state
    else:
        assert pipeline.parents.states[document_revision(chunks).key]["status"] == "processing"
    assert not pipeline.ingest_document(updated).skipped
    assert pipeline.ingest_document(updated).skipped
    assert pipeline.children.verify_children(chunks, all_revisions=True)
    assert pipeline.parents.verify_parents(chunks, all_revisions=True)


def test_invalid_persisted_reference_blocks_child_writes(pipeline, source, caplog):
    with patch.object(pipeline.parents, "resolve_parents", return_value={}):
        with pytest.raises(ValueError, match="invalid persisted parent"):
            pipeline.ingest_document(source)
    assert pipeline.embedder.calls == 0
    assert pipeline.children.statistics()["children"] == 0
    assert "invalid_parent_references" in caplog.text
    assert {s["status"] for s in pipeline.parents.states.values()} == {"processing"}


def test_failed_stored_verification_never_removes_old_records(pipeline, source):
    with (
        patch.object(pipeline.children, "verify_children", return_value=False),
        patch.object(pipeline.children, "remove_obsolete") as cleanup,
    ):
        with pytest.raises(ValueError, match="verification failed"):
            pipeline.ingest_document(source)
    cleanup.assert_not_called()
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
    assert (report.processed, report.failed, report.skipped) == (9, 1, 0)
    assert report.failures == [{"file_name": DOCUMENT_FILES[0], "error": "FileNotFoundError"}]
    assert all(json.loads(line.message)["event"] == "document_failed" for line in caplog.records)
    (settings.data_dir / DOCUMENT_FILES[0]).write_bytes(markdown(DOCUMENT_FILES[0]))
    retry = pipeline.ingest(settings)
    assert (retry.processed, retry.failed, retry.skipped) == (1, 0, 9)


def test_incompatible_collection_is_rejected():
    with closing(QdrantClient(":memory:")) as client:
        client.create_collection(
            "wrong", vectors_config=models.VectorParams(size=10, distance=models.Distance.DOT)
        )
        with pytest.raises(ValueError, match="384-dimensional cosine"):
            QdrantStore(client, "wrong").ensure_collection()
