from contextlib import closing
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from test_ingestion import index_entries

from security_rag.api import create_app
from security_rag.core.application.ports import Candidate, SearchFilters
from security_rag.core.application.search import Search, SearchUnavailable, completed_corpus
from security_rag.core.domain.lexical import lexical_text
from security_rag.core.domain.manifest import DOCUMENT_FILES
from security_rag.core.domain.revisions import document_revision
from security_rag.core.domain.vectors import SparseVector
from security_rag.infrastructure.markdown_chunker import MarkdownChunker
from security_rag.infrastructure.persistence.qdrant import QdrantStore, qdrant_sparse_vector

DENSE = [1.0] + [0.0] * 383
OTHER = [0.0, 1.0] + [0.0] * 382


def test_hybrid_qdrant_search_exact_keyword_dense_and_readiness(
    tokenizer, make_document, sparse_encoder
):
    chunker = MarkdownChunker(tokenizer)
    first = chunker.chunk(
        make_document("### Controls\nUse CWE-79 sandbox isolation.\n\n### Other\nCWE-790 beta.\n")
    )
    second = chunker.chunk(make_document("### Access\nLimit privileges.\n", title="Access"))
    config_hash = sparse_encoder.config.fingerprint
    revisions = [document_revision(first), document_revision(second)]
    with closing(QdrantClient(":memory:")) as client:
        store = QdrantStore(client, "test_search")
        store.ensure_collection()
        for chunks, dense in ((first, OTHER), (second, DENSE)):
            entries = index_entries(chunks)
            sparse = sparse_encoder.encode([lexical_text(entry.child) for entry in entries])
            store.upsert(entries, [dense] * len(entries), sparse, config_hash)
        index = store.search_index(revisions, config_hash, 3)
        sparse = sparse_encoder.encode(["CWE-79 sandbox"], query=True)[0]
        with pytest.raises(ValueError, match="coverage"):
            store.search_index(revisions, "wrong", 3)
        assert (
            index.candidates(DENSE, sparse, SearchFilters(identifier="CWE-79"), limit=20)[
                0
            ].child.id
            == first.children[0].id
        )
        assert not index.candidates(
            DENSE, sparse, SearchFilters(second.parents[0].document_id, "CWE-79"), limit=20
        )
        filtered = index.candidates(
            DENSE, sparse, SearchFilters(first.parents[0].document_id), limit=20
        )
        assert {hit.child.id for hit in filtered} == {child.id for child in first.children}
        keyword_hits = client.query_points(
            "test_search",
            query=qdrant_sparse_vector(sparse_encoder.encode(["sandbox"], query=True)[0]),
            using="bm25",
        ).points
        assert [hit.id for hit in keyword_hits] == [str(first.children[0].id)]
        dense_only = index.candidates(DENSE, None, SearchFilters(), limit=20)
        assert dense_only[0].child.id == second.children[0].id


def test_search_deduplicates_bounds_and_reranks_with_explicit_score_provenance(
    tokenizer, make_document
):
    chunker = MarkdownChunker(tokenizer)
    children = [
        chunker.chunk(make_document(f"### Controls\nControl {i}.\n")).children[0] for i in range(23)
    ]
    encoder, index, reranker = MagicMock(), MagicMock(), MagicMock()
    encoder.encode_query.return_value = (DENSE, SparseVector([1], [1.0]))
    index.candidates.return_value = [
        Candidate(children[0], 1.0),
        *(Candidate(child, 0.5) for child in children),
    ]
    reranker.score.return_value = [0.0, 1.0, 1.0] + [0.0] * 17
    search = Search(encoder, index, reranker)
    result = search.run("control", SearchFilters())
    assert [hit.id for hit in result.results] == [str(children[i].id) for i in (1, 2, 0)]
    assert result.retrieval_mode == "hybrid"
    assert not result.degraded
    assert result.results[0].fusion_score == 0.5
    assert result.results[0].reranker_score == 1
    assert result.results[0].dense_score is None
    assert result.results[0].bm25_rank is None
    assert len(reranker.score.call_args.args[1]) == 20

    encoder.encode_query.return_value = (DENSE, SparseVector([], []))
    reranker.score.side_effect = RuntimeError("model unavailable")
    result = search.run("...", SearchFilters())
    assert result.degraded and result.retrieval_mode == "dense_only"
    assert [hit.dense_rank for hit in result.results] == [1, 2, 3]
    assert all(hit.reranker_score is None and hit.fusion_score is None for hit in result.results)
    assert index.candidates.call_args.args[1] is None

    index.candidates.side_effect = RuntimeError("store unavailable")
    with pytest.raises(RuntimeError, match="store unavailable"):
        search.run("control", SearchFilters())


def test_completed_corpus_requires_every_current_document():
    states = [
        {
            "document_id": f"owasp-llm-2026-{name.split('_')[0].lower()}-title",
            "document_hash": "1" * 64,
            "chunking_config_hash": "2" * 64,
            "status": "completed",
            "sparse_status": "completed",
            "sparse_config_hash": "3" * 64,
            "expected_children": 2,
        }
        for name in DOCUMENT_FILES
    ]
    revisions, total = completed_corpus(states, "3" * 64, "2" * 64)
    assert len(revisions) == 10 and total == 20
    with pytest.raises(SearchUnavailable):
        completed_corpus(states[1:], "3" * 64, "2" * 64)
    states[0]["sparse_status"] = "processing"
    with pytest.raises(SearchUnavailable):
        completed_corpus(states, "3" * 64, "2" * 64)


def test_search_http_validation_and_dependency_failure():
    search = MagicMock(side_effect=SearchUnavailable("internal message"))
    with TestClient(create_app(search)) as client:
        for invalid in (
            {"query": " "},
            {"query": "control", "identifier": "contains CWE-79"},
            {"query": "control", "collection": "unexpected"},
        ):
            assert client.post("/search", json=invalid).status_code == 422
        search.assert_not_called()
        result = client.post("/search", json={"query": " control ", "identifier": "cwe-79"})
        assert result.status_code == 503
        assert "internal message" not in result.text
        search.assert_called_once_with("control", SearchFilters(identifier="CWE-79"))


def test_search_http_returns_child_hits(tokenizer, make_document):
    child = (
        MarkdownChunker(tokenizer).chunk(make_document("### Controls\nLimit access.\n")).children[0]
    )
    encoder, index, reranker = MagicMock(), MagicMock(), MagicMock()
    encoder.encode_query.return_value = (DENSE, SparseVector([1], [1.0]))
    index.candidates.return_value = [Candidate(child, 0.5)]
    reranker.score.return_value = [0.9]
    with TestClient(create_app(Search(encoder, index, reranker).run)) as client:
        response = client.post("/search", json={"query": "limit access"})
    assert response.status_code == 200
    body = response.json()
    assert body["retrieval_mode"] == "hybrid"
    assert not body["degraded"]
    (hit,) = body["results"]
    assert hit["id"] == str(child.id)
    assert hit["parent_id"] == str(child.parent_id)
    assert hit["text"] == child.text
    assert hit["source_url"] == child.source_url
    assert hit["fusion_score"] == 0.5
    assert hit["reranker_score"] == 0.9
    assert hit["bm25_score"] is None


def test_cross_encoder_reuses_model_and_rejects_oversized_pairs(settings, raw_tokenizer):
    from unittest.mock import patch

    from security_rag.infrastructure.embeddings.fastembed.reranker import CrossEncoder

    raw_tokenizer.enable_truncation(max_length=32)
    model = MagicMock()
    model.model.tokenizer = raw_tokenizer
    model.rerank.return_value = [0.25]
    reranker = CrossEncoder(settings)
    with patch("fastembed.rerank.cross_encoder.TextCrossEncoder", return_value=model) as factory:
        assert reranker.score("alpha", ["beta"]) == [0.25]
        with pytest.raises(ValueError, match="pair token budget"):
            reranker.score("alpha", ["beta " * 40])
        assert reranker.score("alpha", ["beta"]) == [0.25]
    factory.assert_called_once()
    assert raw_tokenizer.truncation is None
    assert model.rerank.call_count == 2
