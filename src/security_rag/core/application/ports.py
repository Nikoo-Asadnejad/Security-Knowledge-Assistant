"""Contracts consumed by the ingestion use case."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from security_rag.core.domain.chunks import ChildChunk, ChunkedDocument, ParentChunk
from security_rag.core.domain.documents import Document
from security_rag.core.domain.revisions import Revision
from security_rag.core.domain.sparse_config import SparseConfig
from security_rag.core.domain.vectors import SparseVector


class Chunker(Protocol):
    def chunk(self, document: Document) -> ChunkedDocument: ...

    def parent_token_count(self, parent: ParentChunk) -> int: ...

    def child_token_count(self, child: ChildChunk) -> int: ...

    @property
    def child_max_tokens(self) -> int: ...


class ParentStore(Protocol):
    def prepare_revision(
        self, chunks: ChunkedDocument, *, check_existing: Callable[[tuple[Revision, ...]], None]
    ) -> None:
        """Run the revision policy within the transaction that persists state and parents."""
        ...

    def get_parents(self, ids: list[UUID]) -> dict[UUID, ParentChunk]: ...

    def verify_parents(self, chunks: ChunkedDocument) -> bool: ...

    def finish_revision(self, chunks: ChunkedDocument, sparse_config_hash: str) -> None: ...


class SparseConfigStore(Protocol):
    def sparse_config(self) -> SparseConfig | None: ...

    def save_sparse_config(self, config: SparseConfig) -> SparseConfig:
        """Insert the proposed configuration if absent and return the stored configuration."""
        ...


@dataclass(frozen=True)
class ChildIndexEntry:
    child: ChildChunk
    identifiers: tuple[str, ...]


class ChildIndex(Protocol):
    def ensure_collection(self) -> None: ...

    def upsert(
        self,
        entries: tuple[ChildIndexEntry, ...],
        vectors: list[list[float]],
        sparse_vectors: list[SparseVector],
        sparse_config_hash: str,
    ) -> None: ...

    def verify_children(
        self, revision: Revision, entries: tuple[ChildIndexEntry, ...], sparse_config_hash: str
    ) -> bool: ...


class DenseEncoder(Protocol):
    def embed(self, children: tuple[ChildChunk, ...]) -> list[list[float]]: ...


class SparseEncoder(Protocol):
    @property
    def config(self) -> SparseConfig: ...

    def encode(self, texts: list[str]) -> list[SparseVector]: ...
