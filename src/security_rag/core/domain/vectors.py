"""Vector values and validation shared by encoders and storage."""

import math
from dataclasses import dataclass

VECTOR_SIZE = 384


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]


def validate_vectors(vectors: list[list[float]], expected: int) -> None:
    if len(vectors) != expected:
        raise ValueError("Embedding count does not match child count")
    if any(
        len(vector) != VECTOR_SIZE
        or not all(math.isfinite(value) for value in vector)
        or not any(vector)
        for vector in vectors
    ):
        raise ValueError("Expected finite, nonzero 384-dimensional embeddings")


def validate_sparse(vector: SparseVector, *, allow_empty: bool = False) -> None:
    if (
        len(vector.indices) != len(vector.values)
        or (not vector.indices and not allow_empty)
        or len(set(vector.indices)) != len(vector.indices)
        or any(index < 0 or index > 2**32 - 1 for index in vector.indices)
        or any(not math.isfinite(value) or value <= 0 for value in vector.values)
    ):
        raise ValueError("Expected unique nonnegative sparse indices and finite positive weights")
