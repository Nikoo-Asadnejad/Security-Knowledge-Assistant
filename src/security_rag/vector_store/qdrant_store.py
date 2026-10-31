"""One child collection, with bounded writes and exact per-document verification."""

from collections.abc import Iterator

from qdrant_client import QdrantClient, models

from security_rag.embeddings.embedder import VECTOR_SIZE, validate_vectors
from security_rag.ingestion.chunk_models import ChildChunk, ChunkedDocument
from security_rag.parent_store.models import document_revision


class QdrantStore:
    def __init__(self, client: QdrantClient, collection: str, batch_size: int = 32) -> None:
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
            )
        vectors = self.client.get_collection(self.collection).config.params.vectors
        if (
            not isinstance(vectors, models.VectorParams)
            or vectors.size != VECTOR_SIZE
            or vectors.distance != models.Distance.COSINE
        ):
            raise ValueError("Qdrant collection must use unnamed 384-dimensional cosine vectors")

    def upsert(self, children: tuple[ChildChunk, ...], vectors: list[list[float]]) -> None:
        if not children or len(children) > self.batch_size:
            raise ValueError("Qdrant upsert requires a nonempty bounded batch")
        validate_vectors(vectors, len(children))
        self.client.upsert(
            self.collection,
            points=[
                models.PointStruct(
                    id=str(child.id), vector=vector, payload=child.model_dump(mode="json")
                )
                for child, vector in zip(children, vectors, strict=True)
            ],
            wait=True,
        )

    def _records(
        self, conditions: dict[str, str], *, vectors: bool = False, match_any: bool = False
    ) -> Iterator[list[models.Record]]:
        clauses = [
            models.FieldCondition(key=key, match=models.MatchValue(value=value))
            for key, value in conditions.items()
        ]
        query_filter = models.Filter(should=clauses) if match_any else models.Filter(must=clauses)
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
        self, chunks: ChunkedDocument, *, all_revisions: bool = False, allow_extra: bool = False
    ) -> bool:
        revision = document_revision(chunks)
        conditions = (
            {"document_id": revision.document_id, "file_name": chunks.parents[0].file_name}
            if all_revisions
            else revision.model_dump()
        )
        expected = {str(child.id): child.model_dump(mode="json") for child in chunks.children}
        seen = set()
        for records in self._records(conditions, vectors=True, match_any=all_revisions):
            for record in records:
                point_id = str(record.id)
                if allow_extra and point_id not in expected:
                    continue
                if point_id in seen or record.payload != expected.get(point_id):
                    return False
                try:
                    validate_vectors([record.vector], 1)
                except (ValueError, TypeError):
                    return False
                seen.add(point_id)
        return seen == expected.keys()

    def remove_obsolete(self, chunks: ChunkedDocument) -> None:
        revision = document_revision(chunks)
        expected = {str(child.id) for child in chunks.children}
        # Filenames are manifest identities; a changed title can also change document_id.
        for records in self._records(
            {"document_id": revision.document_id, "file_name": chunks.parents[0].file_name},
            match_any=True,
        ):
            obsolete = [record.id for record in records if str(record.id) not in expected]
            if obsolete:
                self.client.delete(
                    self.collection, points_selector=models.PointIdsList(points=obsolete), wait=True
                )

    def statistics(self) -> dict:
        info = self.client.get_collection(self.collection)
        return {"collection": self.collection, "children": info.points_count}
