"""Optional real-server vector round trip and retrieval in a disposable collection."""

import os
from contextlib import closing
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient

from security_rag.ingestion.chunker import MarkdownChunker
from security_rag.vector_store.qdrant_store import QdrantStore


def test_live_vector_payload_query_and_revision_cleanup(tokenizer, make_document):
    url = os.environ.get("TEST_QDRANT_URL")
    if not url:
        pytest.skip("Set TEST_QDRANT_URL to test an isolated live collection")
    name = f"test_security_rag_{uuid4().hex}"
    with closing(QdrantClient(url=url, timeout=10)) as client:
        store = QdrantStore(client, name, batch_size=1)
        store.ensure_collection()
        try:
            chunker = MarkdownChunker(tokenizer)
            old = chunker.chunk(make_document("### Description\nalpha.\n\n### Detail\nbeta.\n"))
            new = chunker.chunk(make_document("### Description\ngamma.\n"))
            for child in (*old.children, *new.children):
                store.upsert((child,), [[1.0] + [0.0] * 383])
            hits = client.query_points(name, query=[1.0] + [0.0] * 383, limit=3).points
            assert len(hits) == 3
            assert all(hit.id == hit.payload["id"] for hit in hits)
            assert all(hit.payload["parent_id"] for hit in hits)
            assert store.verify_children(new)
            store.remove_obsolete(new)
            assert store.verify_children(new, all_revisions=True)
            assert store.statistics()["children"] == 1
        finally:
            client.delete_collection(name)
