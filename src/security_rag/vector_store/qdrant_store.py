"""One child collection, with bounded writes and exact per-document verification."""

from collections.abc import Iterator

from qdrant_client import QdrantClient, models

from security_rag.embeddings.bm25 import identifiers, lexical_text, validate_sparse
from security_rag.embeddings.embedder import VECTOR_SIZE, validate_vectors
from security_rag.ingestion.chunk_models import ChildChunk, ChunkedDocument
from security_rag.parent_store.models import document_revision


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
        for field in ("document_id", "identifiers"):
            index = self.client.get_collection(self.collection).payload_schema.get(field)
            if index is None:
                self.client.create_payload_index(
                    self.collection, field, models.PayloadSchemaType.KEYWORD, wait=True
                )
            elif index.data_type != models.PayloadSchemaType.KEYWORD:
                raise ValueError(f"{field} requires a keyword payload index")

    @staticmethod
    def payload(child: ChildChunk, sparse_config_hash: str) -> dict:
        return {
            **child.model_dump(mode="json"),
            "sparse_config_hash": sparse_config_hash,
            "identifiers": identifiers(lexical_text(child)),
        }

    def upsert(
        self,
        children: tuple[ChildChunk, ...],
        vectors: list[list[float]],
        sparse_vectors: list[models.SparseVector],
        sparse_config_hash: str,
    ) -> None:
        if not children or len(children) > self.batch_size:
            raise ValueError("Qdrant upsert requires a nonempty bounded batch")
        validate_vectors(vectors, len(children))
        if len(sparse_vectors) != len(children):
            raise ValueError("Sparse vector count does not match child count")
        for vector in sparse_vectors:
            validate_sparse(vector)
        self.client.upsert(
            self.collection,
            points=[
                models.PointStruct(
                    id=str(child.id),
                    vector={"": dense, "bm25": sparse},
                    payload=self.payload(child, sparse_config_hash),
                )
                for child, dense, sparse in zip(children, vectors, sparse_vectors, strict=True)
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

    def verify_children(self, chunks: ChunkedDocument, sparse_config_hash: str) -> bool:
        expected = {
            str(child.id): self.payload(child, sparse_config_hash) for child in chunks.children
        }
        seen = set()
        for records in self.records(document_revision(chunks).model_dump(), vectors=True):
            for record in records:
                point_id = str(record.id)
                if point_id in seen or record.payload != expected.get(point_id):
                    return False
                try:
                    validate_vectors([dense_vector(record)], 1)
                    validate_sparse(sparse_vector(record))
                except (ValueError, TypeError, AttributeError):
                    return False
                seen.add(point_id)
        return seen == expected.keys()

    def statistics(self) -> dict:
        info = self.client.get_collection(self.collection)
        return {"collection": self.collection, "children": info.points_count}


def dense_vector(record: models.Record) -> list[float] | None:
    return record.vector.get("") if isinstance(record.vector, dict) else record.vector


def sparse_vector(record: models.Record) -> models.SparseVector | None:
    return record.vector.get("bm25") if isinstance(record.vector, dict) else None
