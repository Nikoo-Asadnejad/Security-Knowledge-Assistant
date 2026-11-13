from contextlib import closing, nullcontext

import pytest
from qdrant_client import QdrantClient
from test_ingestion import FakeEmbedder, MemoryParents

from security_rag.bootstrap import ingest_documents
from security_rag.core.application.sparse_config import configure_sparse
from security_rag.infrastructure.persistence.qdrant import QdrantStore, dense_vector, sparse_vector


def snapshot(store):
    return {str(record.id): record for batch in store.records({}, vectors=True) for record in batch}


def test_fresh_ingest_creates_both_vectors_and_repeat_regenerates_both(
    settings, markdown, tokenizer, sparse_encoder, monkeypatch, tmp_path
):
    from security_rag.core.domain.manifest import DOCUMENT_FILES
    from security_rag.core.domain.vectors import validate_sparse, validate_vectors

    settings.data_dir.mkdir()
    settings.ingestion_batch_size = 2
    for name in DOCUMENT_FILES:
        (settings.data_dir / name).write_bytes(
            markdown(name, "alpha beta.\n\n#### Detail\ngamma delta.\n")
        )
    parents = MemoryParents()
    monkeypatch.setattr(parents, "close", lambda: None, raising=False)
    monkeypatch.setattr(parents, "single_writer", nullcontext, raising=False)
    dense = FakeEmbedder(parents, settings.ingestion_batch_size)
    directory = tmp_path / "qdrant"
    monkeypatch.setattr(
        "security_rag.infrastructure.persistence.postgres.postgres_store.PostgresStore.from_settings",
        lambda settings: parents,
    )
    monkeypatch.setattr(
        "security_rag.infrastructure.embeddings.fastembed.tokenizer.ChunkTokenizer.load",
        lambda *args, **kwargs: tokenizer,
    )
    monkeypatch.setattr(
        "security_rag.infrastructure.embeddings.fastembed.embedder.Embedder", lambda *args: dense
    )
    monkeypatch.setattr(
        "qdrant_client.QdrantClient",
        lambda **kwargs: QdrantClient(path=str(directory)),
    )

    first = ingest_documents(settings)
    assert (first.processed, first.failed) == (10, 0)
    assert parents.sparse_config() is not None
    with closing(QdrantClient(path=str(directory))) as client:
        records = snapshot(QdrantStore(client, settings.qdrant_collection))
        assert len(records) == sum(document.children for document in first.documents) == 20
        assert len(parents.parents) == sum(document.parents for document in first.documents) == 10
        for record in records.values():
            validate_vectors([dense_vector(record)], 1)
            validate_sparse(sparse_vector(record))
            assert record.payload["sparse_config_hash"] == parents.sparse_config().fingerprint
            assert record.payload["parent_id"] in {str(key) for key in parents.parents}

    calls = dense.calls
    second = ingest_documents(settings)
    assert (second.processed, second.failed) == (10, 0)
    assert dense.calls == 2 * calls
    with closing(QdrantClient(path=str(directory))) as client:
        assert snapshot(QdrantStore(client, settings.qdrant_collection)) == records


def test_sparse_configuration_changes_are_deferred_without_changing_saved_config(
    settings, sparse_encoder
):
    parents = MemoryParents()
    parents.save_sparse_config(sparse_encoder.config)
    settings.bm25_k = 1.8
    with pytest.raises(ValueError, match="Phase 8"):
        configure_sparse(
            parents,
            (),
            count_terms=lambda text: pytest.fail("Should reuse the saved average"),
            build_config=lambda average: sparse_encoder.config.model_copy(
                update={"k": settings.bm25_k, "avg_len": average}
            ),
            build_encoder=lambda config: pytest.fail("Should reject the changed configuration"),
        )
    assert parents.sparse_config() == sparse_encoder.config
