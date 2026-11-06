"""Optional real-server vector round trip and retrieval in a disposable collection."""

import os
from contextlib import closing
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient

from security_rag.ingestion.chunker import MarkdownChunker
from security_rag.vector_store.qdrant_store import QdrantStore


def test_live_dense_and_sparse_vector_round_trip(tokenizer, make_document, sparse_encoder):
    url = os.environ.get("TEST_QDRANT_URL")
    if not url:
        pytest.skip("Set TEST_QDRANT_URL to test an isolated live collection")
    name = f"test_security_rag_{uuid4().hex}"
    with closing(QdrantClient(url=url, timeout=10)) as client:
        store = QdrantStore(client, name, batch_size=1)
        store.ensure_collection()
        try:
            chunker = MarkdownChunker(tokenizer)
            chunks = chunker.chunk(make_document("### Description\nalpha.\n\n### Detail\nbeta.\n"))
            for child in chunks.children:
                from security_rag.embeddings.bm25 import lexical_text

                store.upsert(
                    (child,),
                    [[1.0] + [0.0] * 383],
                    sparse_encoder.encode([lexical_text(child)]),
                    sparse_encoder.config.fingerprint,
                )
            hits = client.query_points(name, query=[1.0] + [0.0] * 383, limit=3).points
            assert len(hits) == 2
            assert all(hit.id == hit.payload["id"] for hit in hits)
            assert all(hit.payload["parent_id"] for hit in hits)
            assert store.verify_children(chunks, sparse_encoder.config.fingerprint)
            assert store.statistics()["children"] == 2
        finally:
            client.delete_collection(name)
