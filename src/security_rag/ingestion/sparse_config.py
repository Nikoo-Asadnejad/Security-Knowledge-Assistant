"""Measure initial lexical lengths and persist the shared BM25 configuration."""

from collections.abc import Iterable

from security_rag.config import Settings
from security_rag.embeddings.bm25 import BM25Encoder, SparseConfig, lexical_text, tokens
from security_rag.ingestion.chunk_models import ChildChunk
from security_rag.parent_store.postgres_store import PostgresStore


def configure_sparse(
    settings: Settings, parents: PostgresStore, children: Iterable[ChildChunk]
) -> BM25Encoder:
    current = parents.sparse_config()
    if current is None:
        total = count = 0
        for child in children:
            length = len(tokens(lexical_text(child), settings.bm25_token_max_length))
            if not length:
                raise ValueError("Child contains no lexical terms")
            total += length
            count += 1
        if not count:
            raise ValueError("No children to encode")
        average = total / count
    else:
        average = current.avg_len
    config = SparseConfig.from_settings(settings, average)
    if current is not None and current != config:
        raise ValueError("Sparse configuration updates are deferred to Phase 8")
    encoder = BM25Encoder(config, settings.embedding_cache_dir)
    parents.save_sparse_config(config)
    return encoder
