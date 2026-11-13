"""Read the local corpus from disk."""

from pathlib import Path

from security_rag.core.domain.documents import Document, parse_document
from security_rag.core.domain.manifest import DOCUMENT_FILES


def load_document(path: Path) -> Document:
    return parse_document(path, path.read_bytes())


def load_documents(data_dir: Path) -> list[Document]:
    """Load exactly the ten manifest documents, failing on missing or invalid files."""
    return [load_document(data_dir / name) for name in DOCUMENT_FILES]
