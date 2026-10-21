import hashlib

import pytest

from security_rag.ingestion.loader import load_document, load_documents, parse_document
from security_rag.ingestion.manifest import DOCUMENT_FILES, SOURCE_BASE_URL


def test_metadata_and_hash_preserve_original_bytes(tmp_path):
    path = tmp_path / DOCUMENT_FILES[0]
    content = b"## LLM01:2026 Prompt Injection\r\n\r\nOriginal **Markdown**.\r\n"
    path.write_bytes(content)
    document = load_document(path)
    assert document.content.encode() == content
    assert document.metadata.model_dump(mode="json") == {
        "document_id": "owasp-llm-2026-llm01-prompt-injection",
        "document_code": "LLM01",
        "title": "Prompt Injection",
        "file_name": path.name,
        "version": "2026",
        "source_path": str(path),
        "source_url": f"{SOURCE_BASE_URL}{path.name}",
        "document_hash": hashlib.sha256(content).hexdigest(),
    }
    updated = parse_document(path, content + b"\n")
    assert updated.metadata.document_id == document.metadata.document_id
    assert updated.metadata.document_hash != document.metadata.document_hash


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b" \r\n\t",
        b"\xff",
        b"<html>GitHub error</html>",
        b"## LLM02:2026 Wrong document\n\nBody",
        b"## LLM01:2025 Wrong year\n\nBody",
        b"## LLM01:2026 Prompt Injection\n\n",
        b"## LLM01:2026  \n\nBody",
    ],
)
def test_invalid_documents_are_rejected(tmp_path, content):
    with pytest.raises(ValueError):
        parse_document(tmp_path / DOCUMENT_FILES[0], content)


def test_non_manifest_path_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Not a manifest document"):
        parse_document(tmp_path / "other.md", b"# Other\nBody")


def test_title_handles_bom_and_closing_heading_markers(tmp_path):
    content = "\ufeff## LLM01:2026 Prompt Injection ###\r\n\r\nBody.\r\n".encode()
    document = parse_document(tmp_path / DOCUMENT_FILES[0], content)
    assert document.metadata.title == "Prompt Injection"
    assert document.metadata.document_id == "owasp-llm-2026-llm01-prompt-injection"
    assert document.content.encode() == content
    assert document.metadata.document_hash == hashlib.sha256(content).hexdigest()


def test_loads_only_manifest_in_order(tmp_path, markdown):
    for name in DOCUMENT_FILES:
        (tmp_path / name).write_bytes(markdown(name))
    (tmp_path / "unrelated.md").write_text("unrelated")
    documents = load_documents(tmp_path)
    assert [document.metadata.file_name for document in documents] == list(DOCUMENT_FILES)
    assert len({document.metadata.document_id for document in documents}) == 10
    (tmp_path / DOCUMENT_FILES[-1]).unlink()
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path)
