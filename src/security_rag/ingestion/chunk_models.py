"""Validated in-memory parent and child chunk records."""

import hashlib
import json
from typing import Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from security_rag.ingestion.loader import DocumentMetadata

ALGORITHM_VERSION = "markdown-parent-child-v2"
HEADING_RULES = "atx-1-6-relative-to-title-fence-aware-source-occurrences"
DOCUMENT_PREFIX = "Document: {title}\n"
SECTION_PREFIX = "Section: {heading_path}\n"
BODY_FORMAT = "\n{prefix}{text}{suffix}"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ChunkingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    parent_max_tokens: int = Field(default=1200, gt=0)
    child_max_tokens: int = Field(default=450, gt=0, le=512)
    child_overlap_tokens: int = Field(default=60, ge=0, le=60)
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    @model_validator(mode="after")
    def validate_limits(self) -> Self:
        if self.parent_max_tokens < self.child_max_tokens:
            raise ValueError("Parent limit must be at least the child limit")
        if self.child_overlap_tokens >= self.child_max_tokens:
            raise ValueError("Overlap must be smaller than the child limit")
        return self

    def fingerprint(self, tokenizer_fingerprint: str) -> str:
        values = {
            **self.model_dump(),
            "algorithm_version": ALGORITHM_VERSION,
            "heading_rules": HEADING_RULES,
            "prefix_format": {
                "document": DOCUMENT_PREFIX,
                "optional_section": SECTION_PREFIX,
                "heading_separator": " > ",
                "body": BODY_FORMAT,
            },
            "tokenizer_fingerprint": tokenizer_fingerprint,
        }
        return sha256(json.dumps(values, sort_keys=True, separators=(",", ":")))


class ChunkMetadata(DocumentMetadata):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID
    section: str
    heading_path: tuple[str, ...] = Field(min_length=1)
    text: str = Field(min_length=1)
    source_start: int = Field(ge=0)
    source_end: int = Field(gt=0)
    chunking_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    markdown_prefix: str = ""
    markdown_suffix: str = ""

    @model_validator(mode="after")
    def validate_source_range(self) -> Self:
        if self.source_end - self.source_start != len(self.text):
            raise ValueError("Source range must match the original text length")
        if not self.text.strip():
            raise ValueError("Chunk text must contain content")
        return self

    @property
    def context_text(self) -> str:
        return render_context(
            self.title, self.heading_path, self.text, self.markdown_prefix, self.markdown_suffix
        )


def render_context(
    title: str, heading_path: tuple[str, ...], text: str, prefix: str = "", suffix: str = ""
) -> str:
    sections = heading_path[1:]
    header = DOCUMENT_PREFIX.format(title=title)
    if sections:
        header += SECTION_PREFIX.format(heading_path=" > ".join(sections))
    return header + BODY_FORMAT.format(prefix=prefix, text=text, suffix=suffix)


class ParentChunk(ChunkMetadata):
    parent_index: int = Field(ge=0)
    parent_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_hash(self) -> Self:
        if self.parent_hash != sha256(self.text):
            raise ValueError("Parent hash does not match text")
        return self


class ChildChunk(ChunkMetadata):
    parent_id: UUID
    chunk_index: int = Field(ge=0)
    child_index: int = Field(ge=0)
    chunk_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_hash(self) -> Self:
        if self.chunk_hash != sha256(self.text):
            raise ValueError("Child hash does not match text")
        return self

    @property
    def embedding_text(self) -> str:
        return self.context_text


class ChunkedDocument(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    parents: tuple[ParentChunk, ...]
    children: tuple[ChildChunk, ...]

    @model_validator(mode="after")
    def validate_links(self) -> Self:
        parents = {parent.id: parent for parent in self.parents}
        if len(parents) != len(self.parents) or len({c.id for c in self.children}) != len(
            self.children
        ):
            raise ValueError("Duplicate chunk IDs")
        referenced = set()
        for child in self.children:
            parent = parents.get(child.parent_id)
            if parent is None:
                raise ValueError("Child refers to a missing parent")
            if (
                child.document_id != parent.document_id
                or child.document_hash != parent.document_hash
                or child.chunking_config_hash != parent.chunking_config_hash
            ):
                raise ValueError("Child and parent revisions do not match")
            if (
                not parent.source_start
                <= child.source_start
                < child.source_end
                <= parent.source_end
            ):
                raise ValueError("Child source range is outside its parent")
            offset = child.source_start - parent.source_start
            if parent.text[offset : offset + len(child.text)] != child.text:
                raise ValueError("Child text does not match the parent source slice")
            referenced.add(parent.id)
        if referenced != set(parents):
            raise ValueError("Every parent must have at least one child")
        return self
