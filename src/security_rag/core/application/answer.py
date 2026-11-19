"""Build bounded, revision-checked context and generate a cited answer."""

import logging
import re
from collections.abc import Callable
from typing import Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from security_rag.core.application.ports import SearchFilters
from security_rag.core.application.search import InvalidSearchQuery, RetrievalResult
from security_rag.core.domain.chunks import ChildChunk, ParentChunk
from security_rag.core.domain.revisions import Revision

logger = logging.getLogger(__name__)
INSUFFICIENT_EVIDENCE = "The retrieved OWASP content does not provide enough evidence to answer."
CITATION = re.compile(r"\[(S[^\]\s]*)\]")


class GenerationUnavailable(RuntimeError):
    """A required generation dependency or its output is invalid."""


class GeneratedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    answer: str = Field(min_length=1)
    insufficient_evidence: bool


class AnswerGenerator(Protocol):
    def estimate_input_tokens(self, question: str, context: str) -> int: ...

    def generate(self, question: str, context: str) -> GeneratedAnswer: ...


class AnswerSource(BaseModel):
    citation_id: str
    document_id: str
    parent_id: str
    matched_child_ids: list[str]
    title: str
    file_name: str
    section: str
    source_url: str
    context_kind: Literal["parent", "children"]


class AnswerResult(BaseModel):
    answer: str
    status: Literal["answered", "insufficient_evidence"]
    sources: list[AnswerSource]
    retrieval_mode: Literal["hybrid", "dense_only"]
    degraded: bool
    estimated_input_tokens: int


class Ask:
    def __init__(
        self,
        retrieve: Callable[[str, SearchFilters], RetrievalResult],
        fetch_parents: Callable[[list[UUID]], dict[UUID, ParentChunk]],
        generator: AnswerGenerator,
        *,
        context_window: int,
        max_output_tokens: int,
    ) -> None:
        if not 0 < max_output_tokens < context_window:
            raise ValueError("Output budget must be positive and smaller than the context window")
        self.retrieve = retrieve
        self.fetch_parents = fetch_parents
        self.generator = generator
        self.input_budget = context_window - max_output_tokens

    def run(self, question: str, filters: SearchFilters | None = None) -> AnswerResult:
        question = question.strip()
        if not question:
            raise InvalidSearchQuery("Question must contain text")
        retrieval = self.retrieve(question, filters or SearchFilters())
        if not retrieval.matches:
            return self._insufficient(retrieval, 0)

        groups: dict[UUID, list[ChildChunk]] = {}
        for match in retrieval.matches:
            groups.setdefault(match.child.parent_id, []).append(match.child)
        # Preserve child rank independently of database row order. An outage must propagate.
        parents = self.fetch_parents(list(groups))
        blocks: list[str] = []
        sources: list[AnswerSource] = []
        estimated_input_tokens = self.generator.estimate_input_tokens(question, "")
        if estimated_input_tokens > self.input_budget:
            raise InvalidSearchQuery("Question exceeds the generation input budget")

        for parent_id, children in groups.items():
            parent = parents.get(parent_id)
            if parent is not None and (
                parent.id != parent_id
                or any(Revision.of(parent) != Revision.of(child) for child in children)
            ):
                parent = None
            if parent is None:
                logger.warning("answer_parent_missing_or_mismatched parent_id=%s", parent_id)

            citation_id = f"S{len(sources) + 1}"
            selected = children
            kind: Literal["parent", "children"] = "children"
            block = ""
            if parent is not None:
                candidate = _block(citation_id, parent.context_text)
                tokens = self.generator.estimate_input_tokens(
                    question, "\n\n".join([*blocks, candidate])
                )
                if tokens <= self.input_budget:
                    block, estimated_input_tokens, kind = candidate, tokens, "parent"
            if not block:
                # Fit complete matching children in rank order, never truncate their text.
                selected = []
                for child in children:
                    candidate = _block(
                        citation_id,
                        "\n\n".join(item.context_text for item in [*selected, child]),
                    )
                    tokens = self.generator.estimate_input_tokens(
                        question, "\n\n".join([*blocks, candidate])
                    )
                    if tokens <= self.input_budget:
                        selected.append(child)
                        block, estimated_input_tokens = candidate, tokens
            if not block:
                continue
            blocks.append(block)
            first = selected[0]
            sources.append(
                AnswerSource(
                    citation_id=citation_id,
                    document_id=first.document_id,
                    parent_id=str(parent_id),
                    matched_child_ids=[str(child.id) for child in selected],
                    title=first.title,
                    file_name=first.file_name,
                    section=parent.section if kind == "parent" else first.section,
                    source_url=first.source_url,
                    context_kind=kind,
                )
            )
        if not sources:
            return self._insufficient(retrieval, estimated_input_tokens)

        generated = self.generator.generate(question, "\n\n".join(blocks))
        if generated.insufficient_evidence:
            return self._insufficient(retrieval, estimated_input_tokens)
        cited = set(CITATION.findall(generated.answer))
        allowed = {source.citation_id for source in sources}
        if not generated.answer.strip() or not cited or not cited <= allowed:
            raise GenerationUnavailable("Answer contains missing or unknown source citations")
        # Source URLs come exclusively from persisted context, never from model output.
        if re.search(r"https?://|\]\(", generated.answer, re.IGNORECASE):
            raise GenerationUnavailable("Answer must use source labels instead of generated links")
        return AnswerResult(
            answer=generated.answer.strip(),
            status="answered",
            sources=[source for source in sources if source.citation_id in cited],
            retrieval_mode=retrieval.retrieval_mode,
            degraded=retrieval.degraded,
            estimated_input_tokens=estimated_input_tokens,
        )

    @staticmethod
    def _insufficient(retrieval: RetrievalResult, estimated_input_tokens: int) -> AnswerResult:
        return AnswerResult(
            answer=INSUFFICIENT_EVIDENCE,
            status="insufficient_evidence",
            sources=[],
            retrieval_mode=retrieval.retrieval_mode,
            degraded=retrieval.degraded,
            estimated_input_tokens=estimated_input_tokens,
        )


def _block(citation_id: str, text: str) -> str:
    return f"[{citation_id}]\n{text}"
