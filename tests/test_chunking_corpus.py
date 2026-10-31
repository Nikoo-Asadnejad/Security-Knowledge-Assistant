"""Opt-in acceptance checks using local OWASP files and the real BGE tokenizer."""

import os
from pathlib import Path

import pytest

from security_rag.ingestion.chunker import MarkdownChunker
from security_rag.ingestion.loader import load_document
from security_rag.ingestion.manifest import DOCUMENT_FILES
from security_rag.ingestion.markdown import MarkdownStructure
from security_rag.ingestion.tokenizer import ChunkTokenizer


@pytest.fixture(scope="module")
def bge_tokenizer():
    path = os.environ.get("TEST_TOKENIZER_PATH")
    if not path:
        pytest.skip("Set TEST_TOKENIZER_PATH to a local BGE tokenizer.json for corpus checks")
    return ChunkTokenizer.load("BAAI/bge-small-en-v1.5", path=Path(path))


@pytest.mark.parametrize("file_name", DOCUMENT_FILES)
def test_corpus_budgets_source_coverage_ids_and_parent_links(bge_tokenizer, file_name):
    path = Path(__file__).resolve().parents[1] / "data" / "owasp" / file_name
    if not path.exists():
        pytest.skip("Download the OWASP corpus before running acceptance checks")
    document = load_document(path)
    structure = MarkdownStructure(document)
    chunker = MarkdownChunker(bge_tokenizer)
    chunks = chunker.chunk(document)
    assert chunks == chunker.chunk(document)
    start = chunks.parents[0].source_start
    assert not document.content[structure.content_start : start].strip()
    assert "".join(p.text for p in chunks.parents) == document.content[start:]
    for left, right in zip(chunks.parents, chunks.parents[1:], strict=False):
        assert left.source_end == right.source_start

    parents = {parent.id: parent for parent in chunks.parents}
    for parent in chunks.parents:
        assert bge_tokenizer.count(parent.context_text) <= 1200
        assert parent.text == document.content[parent.source_start : parent.source_end]
    for child in chunks.children:
        assert bge_tokenizer.count(child.embedding_text) <= 450
        assert child.text == document.content[child.source_start : child.source_end]
        parent = parents[child.parent_id]
        assert parent.document_id == child.document_id
        assert parent.document_hash == child.document_hash
        assert parent.chunking_config_hash == child.chunking_config_hash
        assert parent.source_start <= child.source_start < child.source_end <= parent.source_end
    for left, right in zip(chunks.children, chunks.children[1:], strict=False):
        if left.parent_id != right.parent_id or left.heading_path != right.heading_path:
            assert left.source_end <= right.source_start
        elif right.source_start < left.source_end:
            shared = document.content[right.source_start : left.source_end]
            assert bge_tokenizer.count(shared, special_tokens=False) <= 60


@pytest.mark.parametrize(
    "body",
    [
        "```python\n" + "### alpha = beta + gamma\n" * 250 + "```\n",
        "| Name | Meaning |\n| --- | --- |\n" + "| alpha | beta gamma |\n" * 250,
        "| Name | Meaning |\n| --- | --- |\n| alpha | " + "beta gamma " * 1000 + "|\n",
        "中文🙂e\u0301" * 1000,
    ],
    ids=["code", "table", "oversized-row", "unicode"],
)
def test_real_tokenizer_oversized_blocks(bge_tokenizer, make_document, body):
    document = make_document("### Blocks\n\n" + body)
    chunks = MarkdownChunker(bge_tokenizer).chunk(document)
    assert len(chunks.parents) > 1
    assert (
        "".join(p.text for p in chunks.parents)
        == document.content[chunks.parents[0].source_start :]
    )
    assert all(bge_tokenizer.count(p.context_text) <= 1200 for p in chunks.parents)
    for child in chunks.children:
        assert bge_tokenizer.count(child.embedding_text) <= 450
        assert child.text == document.content[child.source_start : child.source_end]
        if body.startswith("```python"):
            assert child.embedding_text.count("```") == 2
            assert child.heading_path == ("Prompt Injection", "Blocks")
