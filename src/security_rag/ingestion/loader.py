"""Load original Markdown and derive metadata from its exact bytes."""

import hashlib
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from security_rag.ingestion.manifest import DOCUMENT_FILES, SOURCE_BASE_URL, VERSION


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    document_code: str
    title: str
    file_name: str
    version: Literal["2026"] = VERSION
    source_path: Path
    source_url: str
    document_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class Document(BaseModel):
    model_config = ConfigDict(frozen=True)

    metadata: DocumentMetadata
    content: str


def parse_document(path: Path, content: bytes) -> Document:
    """Validate a manifest document without altering its text or hash input."""
    if path.name not in DOCUMENT_FILES:
        raise ValueError(f"Not a manifest document: {path.name}")
    if not content.strip():
        raise ValueError(f"Empty document: {path.name}")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Document is not UTF-8: {path.name}") from exc

    code = path.name.split("_", 1)[0]
    # The canonical files start with an ATX heading containing the code and year.
    heading = re.match(rf"\A\ufeff?\s*#{{1,6}}[ \t]+{code}:{VERSION}[ \t]+([^\r\n]+)", text)
    if heading is None:
        raise ValueError(f"Missing {code}:{VERSION} document heading: {path.name}")
    title = re.sub(r"[ \t]+#+[ \t]*$", "", heading.group(1)).strip()
    if not title or not text[heading.end() :].strip():
        raise ValueError(f"Missing title or body: {path.name}")
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    metadata = DocumentMetadata(
        document_id=f"owasp-llm-{VERSION}-{code.lower()}-{slug}",
        document_code=code,
        title=title,
        file_name=path.name,
        source_path=path,
        source_url=f"{SOURCE_BASE_URL}{path.name}",
        document_hash=hashlib.sha256(content).hexdigest(),
    )
    return Document(metadata=metadata, content=text)


def load_document(path: Path) -> Document:
    return parse_document(path, path.read_bytes())


def load_documents(data_dir: Path) -> list[Document]:
    """Load exactly the ten manifest documents, failing on missing or invalid files."""
    return [load_document(data_dir / name) for name in DOCUMENT_FILES]
