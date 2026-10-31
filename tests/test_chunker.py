import hashlib
from uuid import UUID

import pytest
from pydantic import ValidationError
from tokenizers import models

from security_rag.ingestion.chunk_models import ChunkedDocument, ChunkingConfig, sha256
from security_rag.ingestion.chunker import MarkdownChunker
from security_rag.ingestion.tokenizer import ChunkTokenizer

MODEL = "BAAI/bge-small-en-v1.5"


def chunker(tokenizer, parent=120, child=50, overlap=8):
    return MarkdownChunker(
        tokenizer,
        ChunkingConfig(
            parent_max_tokens=parent, child_max_tokens=child, child_overlap_tokens=overlap
        ),
    )


def assert_integrity(document, result, splitter):
    parents = {parent.id: parent for parent in result.parents}
    for record in (*result.parents, *result.children):
        assert record.text == document.content[record.source_start : record.source_end]
        assert record.document_hash == hashlib.sha256(document.content.encode()).hexdigest()
        assert record.chunking_config_hash == splitter.config_hash
        assert isinstance(record.id, UUID)
        assert record.heading_path[0] == document.metadata.title
    for parent in result.parents:
        assert splitter.tokenizer.count(parent.context_text) <= splitter.config.parent_max_tokens
        assert parent.parent_hash == sha256(parent.text)
    for child in result.children:
        assert splitter.tokenizer.count(child.embedding_text) <= splitter.config.child_max_tokens
        parent = parents[child.parent_id]
        assert parent.source_start <= child.source_start < child.source_end <= parent.source_end
        assert child.chunk_hash == sha256(child.text)
        assert "parent_id" not in child.embedding_text
    assert [child.chunk_index for child in result.children] == list(range(len(result.children)))
    for parent in result.parents:
        children = [c for c in result.children if c.parent_id == parent.id]
        assert [c.child_index for c in children] == list(range(len(children)))


@pytest.mark.parametrize("title_level", [1, 2, 3])
def test_relative_headings_and_introduction(make_document, tokenizer, title_level):
    h = "#" * (title_level + 1)
    nested = "#" * (title_level + 2)
    document = make_document(
        f"Introductory alpha.\n\n{h} Types\n\nOverview beta.\n\n"
        f"{nested} Direct\n\nDirect gamma.\n\n{nested} Indirect\n\nIndirect delta.\n\n"
        f"{h} Mitigation\n\nPrevent attacks.\n",
        title_level=title_level,
    )
    splitter = MarkdownChunker(tokenizer)
    result = splitter.chunk(document)
    assert len(result.parents) == 3
    assert len(result.children) == 5
    assert result.children[0].heading_path == ("Prompt Injection",)
    assert result.children[2].heading_path == ("Prompt Injection", "Types", "Direct")
    assert result.children[1].parent_id == result.children[3].parent_id
    assert result.children[0].parent_id != result.children[-1].parent_id
    assert_integrity(document, result, splitter)


def test_short_parent_has_one_child_and_original_formatting(make_document, tokenizer):
    document = make_document("### Description\r\n\r\n  Original **alpha**.\r\n")
    splitter = MarkdownChunker(tokenizer)
    result = splitter.chunk(document)
    assert len(result.parents) == len(result.children) == 1
    assert result.parents[0].text == result.children[0].text
    assert "\r\n" in result.children[0].text
    assert "  Original" in result.children[0].text
    assert_integrity(document, result, splitter)


def test_oversized_parents_children_and_overlap_keep_full_source(make_document, tokenizer):
    document = make_document("### Long\n\n" + ("alpha beta gamma delta. " * 160) + "\n")
    splitter = chunker(tokenizer)
    result = splitter.chunk(document)
    assert len(result.parents) > 1
    assert len(result.children) > len(result.parents)
    assert (
        "".join(p.text for p in result.parents)
        == document.content[result.parents[0].source_start :]
    )
    for left, right in zip(result.parents, result.parents[1:], strict=False):
        assert left.source_end == right.source_start
    for parent in result.parents:
        children = [c for c in result.children if c.parent_id == parent.id]
        covered = parent.source_start
        for child in children:
            assert child.source_start <= covered
            if child.source_start < covered:
                shared = document.content[child.source_start : covered]
                assert tokenizer.count(shared, special_tokens=False) <= 8
            covered = child.source_end
        assert covered == parent.source_end
    assert any(
        a.source_end > b.source_start
        for a, b in zip(result.children, result.children[1:], strict=False)
    )
    assert_integrity(document, result, splitter)


def test_overlap_does_not_cross_headings_or_parents(make_document, tokenizer):
    document = make_document(
        "### Types\n\n#### Direct\n\n"
        + "alpha beta. " * 80
        + "\n\n#### Indirect\n\n"
        + "gamma delta. " * 80
        + "\n\n### Mitigation\n\n"
        + "alpha gamma. " * 80
    )
    splitter = chunker(tokenizer)
    result = splitter.chunk(document)
    for left, right in zip(result.children, result.children[1:], strict=False):
        if left.parent_id != right.parent_id or left.heading_path != right.heading_path:
            assert left.source_end <= right.source_start
    assert_integrity(document, result, splitter)


def test_repeated_headings_text_and_configuration_have_distinct_ids(make_document, tokenizer):
    document = make_document("### Same\n\nalpha beta.\n\n### Same\n\nalpha beta.\n")
    splitter = MarkdownChunker(tokenizer)
    first = splitter.chunk(document)
    assert first == splitter.chunk(document)
    assert len({p.id for p in first.parents}) == 2
    assert len({c.id for c in first.children}) == 2
    updated_document = make_document("### Same\n\nalpha beta.\n\n### Same\n\nalpha beta.\n\n")
    updated = splitter.chunk(updated_document)
    assert {p.id for p in first.parents}.isdisjoint(p.id for p in updated.parents)
    configured = chunker(tokenizer).chunk(document)
    assert {p.id for p in first.parents}.isdisjoint(p.id for p in configured.parents)
    assert first.parents[0].parent_hash == first.children[0].chunk_hash


@pytest.mark.parametrize("marker", ["```", "~~~~"])
def test_fenced_heading_lookalikes_and_fitting_block_stay_intact(make_document, tokenizer, marker):
    document = make_document(
        f"### Code\n\nBefore.\n\n{marker}python\n### Fake heading\nprint('alpha')\n{marker}\n"
    )
    result = MarkdownChunker(tokenizer).chunk(document)
    assert len(result.parents) == len(result.children) == 1
    assert result.children[0].heading_path == ("Prompt Injection", "Code")
    assert result.children[0].markdown_prefix == result.children[0].markdown_suffix == ""


def test_oversized_fence_retains_language_and_reconstructed_boundaries(make_document, tokenizer):
    document = make_document("### Code\n\n```python\n" + "alpha = beta + gamma\n" * 80 + "```\n")
    splitter = chunker(tokenizer, parent=160, child=60)
    result = splitter.chunk(document)
    continued = [c for c in result.children if c.markdown_prefix]
    assert continued
    assert all(c.markdown_prefix == "```python\n" for c in continued)
    assert any(c.markdown_suffix for c in result.children)
    for child in result.children:
        assert child.embedding_text.count("```") == 2
    assert_integrity(document, result, splitter)


def test_table_rows_repeat_header_only_when_rendered(make_document, tokenizer):
    header = "| Name | Meaning |\n| --- | --- |\n"
    document = make_document("### Table\n\n" + header + "| alpha | beta gamma |\n" * 60)
    splitter = chunker(tokenizer, parent=180, child=70)
    result = splitter.chunk(document)
    continued = [c for c in result.children if c.markdown_prefix]
    assert continued
    assert all(c.markdown_prefix == header for c in continued)
    assert all(header not in c.text for c in continued)
    assert_integrity(document, result, splitter)


def test_lists_and_subsections_prefer_natural_boundaries(make_document, tokenizer):
    body = "### Lists\n\n" + "".join(f"{i}. " + "alpha beta " * 8 + "\n" for i in range(1, 15))
    document = make_document(body)
    splitter = chunker(tokenizer, parent=220, child=65, overlap=0)
    result = splitter.chunk(document)
    assert all(
        c.text.lstrip().startswith(("###", *(str(i) for i in range(1, 15))))
        for c in result.children
    )
    assert_integrity(document, result, splitter)


def test_long_unbroken_unicode_content_does_not_get_decoded_or_truncated(make_document, tokenizer):
    document = make_document("### Unicode\n\n" + "中文🙂 e\u0301 " * 150)
    splitter = chunker(tokenizer, overlap=0)
    result = splitter.chunk(document)
    assert_integrity(document, result, splitter)
    for parent in result.parents:
        assert "".join(c.text for c in result.children if c.parent_id == parent.id) == parent.text


def test_heading_only_sections_create_no_empty_records(make_document, tokenizer):
    document = make_document("### Empty\n\n### Types\n\n#### Direct\n\nalpha beta.\n")
    result = MarkdownChunker(tokenizer).chunk(document)
    assert len(result.parents) == len(result.children) == 1
    assert result.children[0].heading_path == ("Prompt Injection", "Types", "Direct")


def test_impossible_heading_budget_is_rejected(make_document, tokenizer):
    document = make_document("### " + "alpha " * 80 + "\n\nbeta gamma.")
    with pytest.raises(ValueError, match="budget"):
        chunker(tokenizer).chunk(document)


@pytest.mark.parametrize(
    "values",
    [
        {"child_max_tokens": 513},
        {"parent_max_tokens": 200},
        {"child_overlap_tokens": 450},
        {"child_overlap_tokens": -1},
        {"child_overlap_tokens": 61},
        {"parent_max_tokens": 0},
    ],
)
def test_invalid_chunk_configuration(values):
    with pytest.raises(ValidationError):
        ChunkingConfig(**values)


def test_parent_links_and_revision_integrity_are_validated(make_document, tokenizer):
    result = MarkdownChunker(tokenizer).chunk(make_document("### Description\n\nalpha beta."))
    values = result.model_dump(mode="json")
    values["children"][0]["parent_id"] = "00000000-0000-0000-0000-000000000000"
    with pytest.raises(ValidationError, match="missing parent"):
        ChunkedDocument.model_validate(values)
    values = result.model_dump(mode="json")
    values["children"][0]["document_hash"] = "0" * 64
    with pytest.raises(ValidationError, match="revisions"):
        ChunkedDocument.model_validate(values)


def test_tokenizer_disables_truncation_and_padding_on_its_own_copy(raw_tokenizer):
    raw_tokenizer.enable_truncation(max_length=5)
    raw_tokenizer.enable_padding(length=30)
    adapter = ChunkTokenizer(raw_tokenizer, MODEL)
    assert adapter.count("alpha " * 40) == 42
    assert adapter.count("alpha", special_tokens=False) == 1
    assert raw_tokenizer.truncation is not None
    assert raw_tokenizer.padding is not None


def test_local_tokenizer_loading_and_artifact_fingerprint(raw_tokenizer, tmp_path):
    path = tmp_path / "tokenizer.json"
    raw_tokenizer.save(str(path))
    loaded = ChunkTokenizer.load(MODEL, path=path)
    assert loaded.fingerprint == ChunkTokenizer(raw_tokenizer, MODEL).fingerprint
    raw_tokenizer.add_tokens(["new-token"])
    assert loaded.fingerprint != ChunkTokenizer(raw_tokenizer, MODEL).fingerprint


def test_model_mismatch_is_rejected(tokenizer):
    with pytest.raises(ValueError, match="must match"):
        MarkdownChunker(tokenizer, ChunkingConfig(embedding_model="other"))


def test_fitting_blocks_stay_whole_when_surrounding_section_is_oversized(make_document, tokenizer):
    code = "```python\nalpha = beta + gamma\n```\n"
    table = "| Name | Meaning |\n| --- | --- |\n| alpha | beta |\n"
    document = make_document(
        "### Blocks\n\n" + "alpha beta gamma. " * 80 + "\n\n" + code + "\n" + table
    )
    splitter = chunker(tokenizer, parent=160, child=70)
    result = splitter.chunk(document)
    for block in (code, table):
        assert sum(block in p.text for p in result.parents) == 1
        assert sum(block in c.text for c in result.children) == 1
    assert_integrity(document, result, splitter)


def test_unbroken_text_falls_back_to_token_offsets(make_document, raw_tokenizer):
    vocabulary = raw_tokenizer.get_vocab()
    vocabulary.update({"abc": len(vocabulary), "##abc": len(vocabulary) + 1})
    raw_tokenizer.model = models.WordPiece(vocabulary, max_input_chars_per_word=10000)
    tokenizer = ChunkTokenizer(raw_tokenizer, MODEL)
    document = make_document("### Long\n\n" + "abc" * 300)
    body_start = document.content.index("abc")
    splitter = chunker(tokenizer, overlap=0)
    result = splitter.chunk(document)
    assert len(result.children) > 1
    for child in result.children:
        assert (child.source_end - body_start) % 3 == 0
        if child.source_start > body_start:
            assert (child.source_start - body_start) % 3 == 0
    assert_integrity(document, result, splitter)


def test_tokenizer_loading_failure_is_a_useful_validation_error(tmp_path):
    path = tmp_path / "invalid-tokenizer.json"
    path.write_text("not JSON")
    with pytest.raises(ValueError, match="Could not load tokenizer"):
        ChunkTokenizer.load(MODEL, path=path)


def test_fitting_logical_units_have_no_overlap(make_document, tokenizer):
    document = make_document("### Types\n\nalpha beta.\n\n#### Direct\n\ngamma delta.")
    result = chunker(tokenizer).chunk(document)
    assert len(result.children) == 2
    assert result.children[0].source_end == result.children[1].source_start
    assert result.children[0].embedding_text.count("Prompt Injection") == 1
    assert "Section: Types > Direct" in result.children[1].embedding_text


def test_overlap_is_reduced_to_fit_the_complete_input(make_document, tokenizer):
    document = make_document("### Long\n\n" + ("alpha " * 34 + ".\n\n") * 3)
    splitter = chunker(tokenizer, parent=200, child=50, overlap=8)
    result = splitter.chunk(document)
    assert len(result.parents) == 1
    assert len(result.children) == 3
    for left, right in zip(result.children, result.children[1:], strict=False):
        shared = document.content[right.source_start : left.source_end]
        assert 0 < tokenizer.count(shared, special_tokens=False) < 8
    assert_integrity(document, result, splitter)


def test_configuration_hash_covers_limits_model_and_tokenizer(tokenizer, raw_tokenizer):
    config = ChunkingConfig()
    baseline = config.fingerprint(tokenizer.fingerprint)
    assert len(baseline) == 64
    assert baseline == ChunkingConfig().fingerprint(tokenizer.fingerprint)
    for values in (
        {"parent_max_tokens": 1100},
        {"child_max_tokens": 440},
        {"child_overlap_tokens": 50},
        {"embedding_model": "other"},
    ):
        assert ChunkingConfig(**values).fingerprint(tokenizer.fingerprint) != baseline
    raw_tokenizer.add_tokens(["new-token"])
    assert config.fingerprint(ChunkTokenizer(raw_tokenizer, MODEL).fingerprint) != baseline
