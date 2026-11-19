from dataclasses import replace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from security_rag.api import create_app
from security_rag.core.application.answer import (
    INSUFFICIENT_EVIDENCE,
    Ask,
    GeneratedAnswer,
    GenerationUnavailable,
)
from security_rag.core.application.ports import Candidate, SearchFilters
from security_rag.core.application.search import InvalidSearchQuery, Search
from security_rag.core.domain.vectors import SparseVector
from security_rag.infrastructure.markdown_chunker import MarkdownChunker


def retrieval(children):
    encoder, index, reranker = MagicMock(), MagicMock(), MagicMock()
    encoder.encode_query.return_value = ([1.0], SparseVector([], []))
    index.candidates.return_value = [Candidate(child, 1.0) for child in children]
    reranker.score.return_value = [float(-i) for i in range(len(children))]
    return Search(encoder, index, reranker).retrieve("question", SearchFilters())


@pytest.fixture
def chunks(tokenizer, make_document):
    return MarkdownChunker(tokenizer).chunk(
        make_document(
            "### Controls\n#### Isolation\nUse sandbox isolation.\n"
            "#### Permissions\nLimit privileges.\n\n### Monitoring\nLog security events.\n"
        )
    )


def use_case(children, parents, *, budget=10000, answer="Use isolation [S1]."):
    retrieve = MagicMock(return_value=retrieval(children))
    fetch = MagicMock(return_value={parent.id: parent for parent in parents})
    generator = MagicMock()
    generator.estimate_input_tokens.side_effect = lambda question, context: 10 + len(context)
    generator.generate.return_value = GeneratedAnswer(answer=answer, insufficient_evidence=False)
    return (
        Ask(retrieve, fetch, generator, context_window=budget + 50, max_output_tokens=50),
        retrieve,
        fetch,
        generator,
    )


def test_ask_deduplicates_parents_preserves_rank_and_citation_provenance(chunks):
    children = [chunks.children[2], chunks.children[0], chunks.children[1]]
    ask, retrieve, fetch, generator = use_case(
        children, chunks.parents, answer="Log events [S1]. Limit privileges [S2]."
    )
    result = ask.run("  How do I secure this?  ", SearchFilters(identifier="LLM01"))
    retrieve.assert_called_once_with("How do I secure this?", SearchFilters(identifier="LLM01"))
    fetch.assert_called_once_with([children[0].parent_id, children[1].parent_id])
    assert [source.parent_id for source in result.sources] == [
        str(children[0].parent_id),
        str(children[1].parent_id),
    ]
    assert result.sources[1].matched_child_ids == [str(child.id) for child in children[1:]]
    assert all(source.context_kind == "parent" for source in result.sources)
    assert all(source.source_url == children[0].source_url for source in result.sources)
    context = generator.generate.call_args.args[1]
    assert context.index("Log security events") < context.index("Use sandbox isolation")
    assert context.count("Limit privileges") == 1
    assert result.estimated_input_tokens == 10 + len(context)
    assert result.status == "answered"


@pytest.mark.parametrize(
    "update",
    [
        None,
        {"document_hash": "1" * 64},
        {"chunking_config_hash": "2" * 64},
        {"document_id": "different-document"},
        {"id": uuid4()},
    ],
)
def test_ask_uses_matching_children_for_missing_or_wrong_parent(chunks, update):
    children = chunks.children[:2]
    ask, _, fetch, generator = use_case(children, [])
    if update is not None:
        fetch.return_value = {children[0].parent_id: chunks.parents[0].model_copy(update=update)}
    result = ask.run("How do I secure this?")
    assert result.sources[0].context_kind == "children"
    assert result.sources[0].matched_child_ids == [str(child.id) for child in children]
    assert all(child.context_text in generator.generate.call_args.args[1] for child in children)


def test_budget_falls_back_to_complete_child_without_unmatched_siblings(chunks):
    child = chunks.children[0]
    budget = 10 + len(f"[S1]\n{child.context_text}")
    ask, _, _, generator = use_case([child], chunks.parents, budget=budget)
    result = ask.run("How do I secure this?")
    assert result.sources[0].context_kind == "children"
    assert result.estimated_input_tokens == budget
    context = generator.generate.call_args.args[1]
    assert child.context_text in context
    assert "Limit privileges" not in context


def test_ask_skips_context_that_cannot_fit_and_does_not_call_model(chunks):
    ask, _, _, generator = use_case(chunks.children, chunks.parents, budget=11)
    result = ask.run("Question")
    assert result.status == "insufficient_evidence" and not result.sources
    assert result.estimated_input_tokens == 10
    generator.generate.assert_not_called()


def test_ask_empty_retrieval_and_model_abstention(chunks):
    ask, _, fetch, generator = use_case([], [])
    assert ask.run("Question").status == "insufficient_evidence"
    fetch.assert_not_called()
    generator.estimate_input_tokens.assert_not_called()
    generator.generate.assert_not_called()

    ask, _, _, generator = use_case(chunks.children, chunks.parents)
    generator.generate.return_value = GeneratedAnswer(
        answer="Not supported.", insufficient_evidence=True
    )
    result = ask.run("Unsupported question")
    assert result.answer == INSUFFICIENT_EVIDENCE
    assert not result.sources


def test_ask_propagates_degraded_retrieval(chunks):
    ask, retrieve, _, _ = use_case(chunks.children, chunks.parents)
    retrieve.return_value = replace(retrieve.return_value, degraded=True)
    assert ask.run("Question").degraded


@pytest.mark.parametrize(
    "answer",
    ["No citation.", "Invented [S9].", "Valid [S1] and invalid [S0].", "[S1](https://fake)"],
)
def test_ask_rejects_missing_fabricated_or_model_generated_links(chunks, answer):
    ask, _, _, _ = use_case(chunks.children[:1], chunks.parents, answer=answer)
    with pytest.raises(GenerationUnavailable):
        ask.run("Question")


def test_ask_returns_only_sources_cited_from_the_included_context(chunks):
    ask, _, _, _ = use_case(chunks.children, chunks.parents, answer="Log events [S2].")
    result = ask.run("Question")
    assert [source.citation_id for source in result.sources] == ["S2"]
    assert result.sources[0].parent_id == str(chunks.parents[1].id)


def test_ask_does_not_hide_store_or_model_failures(chunks):
    ask, _, fetch, generator = use_case(chunks.children, chunks.parents)
    fetch.side_effect = RuntimeError("database outage")
    with pytest.raises(RuntimeError, match="database outage"):
        ask.run("Question")
    generator.generate.assert_not_called()
    fetch.side_effect = None
    generator.generate.side_effect = RuntimeError("model outage")
    with pytest.raises(RuntimeError, match="model outage"):
        ask.run("Question")


def test_ask_rejects_empty_or_oversized_questions(chunks):
    ask, retrieve, _, generator = use_case(chunks.children, chunks.parents)
    with pytest.raises(InvalidSearchQuery):
        ask.run(" ")
    retrieve.assert_not_called()
    generator.estimate_input_tokens.return_value = 20000
    generator.estimate_input_tokens.side_effect = None
    with pytest.raises(InvalidSearchQuery, match="budget"):
        ask.run("Question")
    generator.generate.assert_not_called()


def test_ask_http_contract_and_sanitized_failures(chunks):
    ask, retrieve, fetch, _ = use_case(chunks.children, chunks.parents)
    with TestClient(create_app(ask=ask.run)) as client:
        for body in ({"question": " "}, {"question": "Question", "collection": "other"}):
            assert client.post("/ask", json=body).status_code == 422
        response = client.post("/ask", json={"question": "Question", "identifier": "llm01"})
        assert response.status_code == 200
        assert response.json()["sources"][0]["citation_id"] == "S1"
        assert retrieve.call_args.args[1].identifier == "LLM01"
        fetch.side_effect = RuntimeError("private database detail")
        response = client.post("/ask", json={"question": "Question"})
        assert response.status_code == 503
        assert "private database detail" not in response.text
