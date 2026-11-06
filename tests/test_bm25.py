import hashlib
import math

import pytest
from qdrant_client import models

from security_rag.embeddings.bm25 import (
    identifiers,
    lexical_text,
    preprocess,
    tokens,
    validate_sparse,
)
from security_rag.ingestion.chunker import MarkdownChunker


def test_actual_analyzer_preserves_codes_aliases_case_and_query_weights(sparse_encoder):
    text = "LLM01 CWE-79 CVE-2026-12345 ERR_CONNECTION_RESET 0x80070005"
    assert identifiers(text.lower()) == [
        "0x80070005",
        "CVE-2026-12345",
        "CWE-79",
        "ERR_CONNECTION_RESET",
        "LLM01",
    ]
    emitted = tokens(text, 512)
    assert emitted[:9] == [
        "llm01",
        "cwe",
        "79",
        "cve",
        "2026",
        "12345",
        "err_connection_reset",
        "0x80070005",
        "code" + hashlib.sha256(b"CVE-2026-12345").hexdigest()[:32],
    ]
    assert len([term for term in emitted if term.startswith("code")]) == 3
    assert "running" in tokens("running the", 512) and "the" in tokens("running the", 512)
    document = sparse_encoder.encode([text])[0]
    assert sparse_encoder.encode([text.lower()])[0] == document
    query = sparse_encoder.encode([text], query=True)[0]
    assert set(query.indices) == set(document.indices)
    assert query.values == [1.0] * len(query.indices)
    assert any(not math.isclose(value, 1) for value in document.values)
    assert sparse_encoder.encode(["..."], query=True)[0].indices == []
    with pytest.raises(ValueError, match="no lexical terms"):
        sparse_encoder.encode(["..."])


def test_identifiers_do_not_match_near_codes_or_substrings():
    assert identifiers("XLLM01 LLM010 CWE-790x XCWE-79 ERR_CONNECTION_RESETX") == [
        "ERR_CONNECTION_RESETX"
    ]
    assert identifiers("CWE-79 cwe-79") == ["CWE-79"]
    assert preprocess("CWE-79") != preprocess("CWE-790")


def test_long_identifier_not_silently_discarded(sparse_encoder):
    text = "ERR_" + "A" * 100
    assert text.lower() in tokens(text, 512)
    assert sparse_encoder.encode([text])[0].indices
    with pytest.raises(ValueError, match="BM25_TOKEN_MAX_LENGTH"):
        tokens("a" * 513, 512)


@pytest.mark.parametrize(
    "indices,values",
    [
        ([], []),
        ([1, 1], [1, 2]),
        ([1], []),
        ([1], [0]),
        ([1], [-1]),
        ([1], [float("nan")]),
        ([1], [float("inf")]),
        ([-1], [1]),
    ],
)
def test_invalid_sparse_vectors_are_rejected(indices, values):
    with pytest.raises(ValueError):
        validate_sparse(models.SparseVector(indices=indices, values=values))


def test_lexical_input_excludes_storage_metadata_and_duplicate_headings(tokenizer, make_document):
    chunks = MarkdownChunker(tokenizer).chunk(
        make_document("### Mitigation\nUse CWE-79 controls.\n")
    )
    child = chunks.children[0]
    text = lexical_text(child)
    assert text.count("Prompt Injection") == 1
    assert text.count("Mitigation") == 1
    assert "LLM01" in text and "CWE-79" in text
    assert str(child.id) not in text and child.document_hash not in text
    assert child.source_url not in text
