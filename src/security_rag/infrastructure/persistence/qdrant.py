"""One child collection, with bounded writes and exact per-document verification."""

from __future__ import annotations

from collections.abc import Iterator
from math import isfinite

from qdrant_client import QdrantClient, models

from security_rag.core.application.ports import Candidate, ChildIndexEntry, SearchFilters
from security_rag.core.domain.chunks import ChildChunk
from security_rag.core.domain.revisions import Revision
from security_rag.core.domain.vectors import (
    VECTOR_SIZE,
    SparseVector,
    validate_sparse,
    validate_vectors,
)


class QdrantStore:
    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        batch_size: int = 32,
    ) -> None:
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        self.client = client
        self.collection = collection
        self.batch_size = batch_size

    def ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE, distance=models.Distance.COSINE
                ),
                sparse_vectors_config={
                    "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )
        self.check_schema()
        for field in ("document_id", "identifiers"):
            index = self.client.get_collection(self.collection).payload_schema.get(field)
            if index is None:
                self.client.create_payload_index(
                    self.collection, field, models.PayloadSchemaType.KEYWORD, wait=True
                )
            elif index.data_type != models.PayloadSchemaType.KEYWORD:
                raise ValueError(f"{field} requires a keyword payload index")

    def check_schema(self) -> None:
        params = self.client.get_collection(self.collection).config.params
        vectors = params.vectors
        if isinstance(vectors, dict):
            vectors = vectors.get("")
        if (
            not isinstance(vectors, models.VectorParams)
            or vectors.size != VECTOR_SIZE
            or vectors.distance != models.Distance.COSINE
        ):
            raise ValueError("Qdrant collection must use unnamed 384-dimensional cosine vectors")
        sparse = (params.sparse_vectors or {}).get("bm25")
        if sparse is None or sparse.modifier != models.Modifier.IDF:
            raise ValueError("Qdrant collection must define bm25 sparse vectors with IDF")

    def search_index(
        self, revisions: list[Revision], config_hash: str, expected_children: int
    ) -> QdrantSearchIndex:
        self.check_schema()
        ready = models.Filter(
            must=[
                _match("sparse_config_hash", config_hash),
                models.HasVectorCondition(has_vector=""),
                models.HasVectorCondition(has_vector="bm25"),
                models.Filter(
                    should=[
                        models.Filter(
                            must=[
                                _match(key, value) for key, value in revision.model_dump().items()
                            ]
                        )
                        for revision in revisions
                    ]
                ),
            ]
        )
        actual = self.client.count(self.collection, count_filter=ready, exact=True).count
        total = self.client.count(self.collection, exact=True).count
        if actual != expected_children or total != expected_children:
            raise ValueError("Qdrant child coverage does not match completed ingestion")
        return QdrantSearchIndex(self.client, self.collection, ready)

    @staticmethod
    def payload(entry: ChildIndexEntry, sparse_config_hash: str) -> dict:
        return {
            **entry.child.model_dump(mode="json"),
            "sparse_config_hash": sparse_config_hash,
            "identifiers": list(entry.identifiers),
        }

    def upsert(
        self,
        entries: tuple[ChildIndexEntry, ...],
        vectors: list[list[float]],
        sparse_vectors: list[SparseVector],
        sparse_config_hash: str,
    ) -> None:
        if not entries or len(entries) > self.batch_size:
            raise ValueError("Qdrant upsert requires a nonempty bounded batch")
        validate_vectors(vectors, len(entries))
        if len(sparse_vectors) != len(entries):
            raise ValueError("Sparse vector count does not match child count")
        for vector in sparse_vectors:
            validate_sparse(vector)
        self.client.upsert(
            self.collection,
            points=[
                models.PointStruct(
                    id=str(entry.child.id),
                    vector={"": dense, "bm25": qdrant_sparse_vector(sparse)},
                    payload=self.payload(entry, sparse_config_hash),
                )
                for entry, dense, sparse in zip(entries, vectors, sparse_vectors, strict=True)
            ],
            wait=True,
        )

    def records(
        self, conditions: dict[str, str], *, vectors: bool = False
    ) -> Iterator[list[models.Record]]:
        clauses = [
            models.FieldCondition(key=key, match=models.MatchValue(value=value))
            for key, value in conditions.items()
        ]
        query_filter = models.Filter(must=clauses)
        offset = None
        while True:
            records, offset = self.client.scroll(
                self.collection,
                scroll_filter=query_filter,
                limit=self.batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=vectors,
            )
            yield records
            if offset is None:
                return

    def verify_children(
        self, revision: Revision, entries: tuple[ChildIndexEntry, ...], sparse_config_hash: str
    ) -> bool:
        expected = {
            str(entry.child.id): self.payload(entry, sparse_config_hash) for entry in entries
        }
        seen = set()
        for records in self.records(revision.model_dump(), vectors=True):
            for record in records:
                point_id = str(record.id)
                if point_id in seen or record.payload != expected.get(point_id):
                    return False
                try:
                    dense, sparse = dense_vector(record), sparse_vector(record)
                    if dense is None or sparse is None:
                        return False
                    validate_vectors([dense], 1)
                    validate_sparse(sparse)
                except (ValueError, TypeError):
                    return False
                seen.add(point_id)
        return seen == expected.keys()

    def statistics(self) -> dict:
        info = self.client.get_collection(self.collection)
        return {"collection": self.collection, "children": info.points_count}


def dense_vector(record: models.Record) -> list[float] | None:
    vector = record.vector.get("") if isinstance(record.vector, dict) else record.vector
    return vector if isinstance(vector, list) else None


def sparse_vector(record: models.Record) -> SparseVector | None:
    vector = record.vector.get("bm25") if isinstance(record.vector, dict) else None
    if not isinstance(vector, models.SparseVector):
        return None
    return SparseVector(indices=vector.indices, values=vector.values)


def qdrant_sparse_vector(vector: SparseVector) -> models.SparseVector:
    return models.SparseVector(indices=vector.indices, values=vector.values)


class QdrantSearchIndex:
    def __init__(self, client: QdrantClient, collection: str, ready: models.Filter) -> None:
        self.client = client
        self.collection = collection
        self.ready = ready

    def candidates(
        self,
        dense: list[float],
        sparse: SparseVector | None,
        filters: SearchFilters,
        *,
        limit: int,
    ) -> list[Candidate]:
        clauses = [self.ready]
        if filters.document_id is not None:
            clauses.append(_match("document_id", filters.document_id))
        if filters.identifier is not None:
            clauses.append(_match("identifiers", filters.identifier))
        match = models.Filter(must=clauses)
        if sparse is None:
            response = self.client.query_points(
                self.collection, query=dense, query_filter=match, limit=limit, with_payload=True
            )
        else:
            response = self.client.query_points(
                self.collection,
                prefetch=[
                    models.Prefetch(query=dense, filter=match, limit=limit),
                    models.Prefetch(
                        query=qdrant_sparse_vector(sparse), using="bm25", filter=match, limit=limit
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
        candidates = []
        for point in response.points:
            payload = dict(point.payload or {})
            payload.pop("sparse_config_hash", None)
            payload.pop("identifiers", None)
            child = ChildChunk.model_validate(payload)
            if str(child.id) != str(point.id) or not isfinite(point.score):
                raise ValueError("Invalid indexed child result")
            candidates.append(Candidate(child, point.score))
        return candidates


def _match(key: str, value: str) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))
