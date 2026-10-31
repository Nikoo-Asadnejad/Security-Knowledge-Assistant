"""Revision identity shared by persisted parents and child references."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from security_rag.ingestion.chunk_models import ChildChunk, ChunkedDocument, ParentChunk


class Revision(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    document_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunking_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def of(cls, chunk: ParentChunk | ChildChunk) -> "Revision":
        return cls.model_validate(chunk.model_dump())

    @property
    def key(self) -> tuple[str, str, str]:
        return self.document_id, self.document_hash, self.chunking_config_hash


class ParentReference(Revision):
    parent_id: UUID


def document_revision(chunks: ChunkedDocument) -> Revision:
    if not chunks.parents or not chunks.children:
        raise ValueError("Ingestion requires nonempty parents and children")
    revision = Revision.of(chunks.parents[0])
    if any(Revision.of(parent) != revision for parent in chunks.parents):
        raise ValueError("Ingestion requires exactly one document revision")
    return revision
