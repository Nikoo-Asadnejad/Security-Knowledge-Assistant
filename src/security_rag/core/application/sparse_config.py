"""Measure and freeze the BM25 configuration for initial indexing."""

from collections.abc import Callable, Iterable

from security_rag.core.application.ports import SparseConfigStore, SparseEncoder
from security_rag.core.domain.chunks import ChildChunk
from security_rag.core.domain.lexical import lexical_text
from security_rag.core.domain.sparse_config import SparseConfig


def configure_sparse(
    store: SparseConfigStore,
    children: Iterable[ChildChunk],
    *,
    count_terms: Callable[[str], int],
    build_config: Callable[[float], SparseConfig],
    build_encoder: Callable[[SparseConfig], SparseEncoder],
) -> SparseEncoder:
    current = store.sparse_config()
    average = current.avg_len if current is not None else _average_length(children, count_terms)
    config = build_config(average)
    if current is not None and current != config:
        raise ValueError("Sparse configuration updates are deferred to Phase 8")

    encoder = build_encoder(config)
    if store.save_sparse_config(config) != config:
        raise ValueError("Sparse configuration updates are deferred to Phase 8")
    return encoder


def _average_length(children: Iterable[ChildChunk], count_terms: Callable[[str], int]) -> float:
    total = count = 0
    for child in children:
        length = count_terms(lexical_text(child))
        if not length:
            raise ValueError("Child contains no lexical terms")
        total += length
        count += 1
    if not count:
        raise ValueError("No children to encode")
    return total / count
