"""Retrieve and rank child chunks through the search ports."""

import logging
from dataclasses import dataclass
from math import isfinite
from typing import Literal

from pydantic import BaseModel

from security_rag.core.application.ports import (
    CandidateIndex,
    QueryEncoder,
    Reranker,
    SearchFilters,
)
from security_rag.core.domain.chunks import ChildChunk
from security_rag.core.domain.manifest import DOCUMENT_FILES, VERSION
from security_rag.core.domain.revisions import Revision

logger = logging.getLogger(__name__)
CANDIDATE_LIMIT = 20
RESULT_LIMIT = 3


class InvalidSearchQuery(ValueError):
    """A query or its filters cannot be encoded."""


class SearchUnavailable(RuntimeError):
    """The index or a required search dependency is unavailable."""


class SearchHit(BaseModel):
    id: str
    parent_id: str
    text: str
    document_id: str
    title: str
    file_name: str
    section: str
    source_url: str
    fusion_score: float | None
    reranker_score: float | None
    dense_score: float | None
    dense_rank: int | None
    bm25_score: float | None = None
    bm25_rank: int | None = None


class SearchResult(BaseModel):
    retrieval_mode: Literal["hybrid", "dense_only"]
    degraded: bool
    results: list[SearchHit]


@dataclass(frozen=True)
class RankedChild:
    child: ChildChunk
    hit: SearchHit


@dataclass(frozen=True)
class RetrievalResult:
    retrieval_mode: Literal["hybrid", "dense_only"]
    degraded: bool
    matches: tuple[RankedChild, ...]

    def search_result(self) -> SearchResult:
        return SearchResult(
            retrieval_mode=self.retrieval_mode,
            degraded=self.degraded,
            results=[match.hit for match in self.matches],
        )


class Search:
    def __init__(self, encoder: QueryEncoder, index: CandidateIndex, reranker: Reranker) -> None:
        self.encoder = encoder
        self.index = index
        self.reranker = reranker

    def run(self, query: str, filters: SearchFilters) -> SearchResult:
        return self.retrieve(query, filters).search_result()

    def retrieve(self, query: str, filters: SearchFilters) -> RetrievalResult:
        query = query.strip()
        if not query:
            raise InvalidSearchQuery("Query must contain text")
        dense, sparse = self.encoder.encode_query(query)
        hybrid = bool(sparse.indices)
        candidates = self.index.candidates(
            dense, sparse if hybrid else None, filters, limit=CANDIDATE_LIMIT
        )
        unique = {}
        for candidate in candidates:
            unique.setdefault(candidate.child.id, candidate)
        ranked = list(unique.values())[:CANDIDATE_LIMIT]
        scores: list[float | None] = [None] * len(ranked)
        degraded = False
        if ranked:
            try:
                scored = self.reranker.score(
                    query, [candidate.child.embedding_text for candidate in ranked]
                )
                if len(scored) != len(ranked) or not all(isfinite(score) for score in scored):
                    raise ValueError("Invalid reranker scores")
                scores = list(scored)
            except Exception as exc:
                logger.warning("reranking_failed error_type=%s", type(exc).__name__)
                degraded = True

        order = list(range(len(ranked)))
        if ranked and not degraded:
            order.sort(key=lambda index: (-scored[index], index, ranked[index].child.id))
        return RetrievalResult(
            retrieval_mode="hybrid" if hybrid else "dense_only",
            degraded=degraded,
            matches=tuple(
                RankedChild(
                    ranked[index].child,
                    _hit(ranked[index].child, ranked[index].score, scores[index], index, hybrid),
                )
                for index in order[:RESULT_LIMIT]
            ),
        )


def _hit(
    child: ChildChunk, score: float, reranker_score: float | None, index: int, hybrid: bool
) -> SearchHit:
    return SearchHit(
        id=str(child.id),
        parent_id=str(child.parent_id),
        text=child.text,
        document_id=child.document_id,
        title=child.title,
        file_name=child.file_name,
        section=child.section,
        source_url=child.source_url,
        fusion_score=score if hybrid else None,
        reranker_score=reranker_score,
        dense_score=None if hybrid else score,
        dense_rank=None if hybrid else index + 1,
    )


def completed_corpus(
    states: list[dict], config_hash: str, chunking_hash: str
) -> tuple[list[Revision], int]:
    expected = {name.split("_", 1)[0].lower() for name in DOCUMENT_FILES}
    prefix = f"owasp-llm-{VERSION}-"
    revisions = []
    child_count = 0
    for state in states:
        revision = Revision.model_validate(state)
        code, _, title = revision.document_id.removeprefix(prefix).partition("-")
        if (
            not revision.document_id.startswith(prefix)
            or not title
            or code not in expected
            or revision.chunking_config_hash != chunking_hash
            or state["status"] != "completed"
            or state["sparse_status"] != "completed"
            or state["sparse_config_hash"] != config_hash
            or state["expected_children"] <= 0
        ):
            raise SearchUnavailable("Ingestion state is incompatible or incomplete")
        expected.remove(code)
        revisions.append(revision)
        child_count += state["expected_children"]
    if expected:
        raise SearchUnavailable("Ingestion is incomplete")
    return revisions, child_count
