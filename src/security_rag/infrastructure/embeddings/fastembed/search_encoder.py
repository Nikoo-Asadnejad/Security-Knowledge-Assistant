"""Dense and sparse query encoding with the corpus's frozen sparse configuration."""

from __future__ import annotations

from threading import Lock

from security_rag.config import Settings
from security_rag.core.application.search import InvalidSearchQuery
from security_rag.core.domain.sparse_config import SparseConfig
from security_rag.core.domain.vectors import SparseVector
from security_rag.infrastructure.embeddings.fastembed.bm25 import BM25Encoder, tokens
from security_rag.infrastructure.embeddings.fastembed.embedder import Embedder


class SearchEncoder:
    def __init__(self, dense: Embedder, settings: Settings) -> None:
        self.dense = dense
        self.cache_dir = settings.embedding_cache_dir
        self._sparse: BM25Encoder | None = None
        self._lock = Lock()

    def for_config(self, config: SparseConfig) -> ConfiguredQueryEncoder:
        with self._lock:
            if self._sparse is None or self._sparse.config != config:
                self._sparse = BM25Encoder(config, self.cache_dir)
            return ConfiguredQueryEncoder(self.dense, self._sparse)


class ConfiguredQueryEncoder:
    def __init__(self, dense: Embedder, sparse: BM25Encoder) -> None:
        self.dense = dense
        self.sparse = sparse

    def encode_query(self, query: str) -> tuple[list[float], SparseVector]:
        try:
            terms = tokens(query, self.sparse.config.token_max_length)
        except ValueError as exc:
            raise InvalidSearchQuery(str(exc)) from exc
        dense = self.dense.embed_query(query)
        sparse = self.sparse.encode([query], query=True)[0] if terms else SparseVector([], [])
        return dense, sparse
