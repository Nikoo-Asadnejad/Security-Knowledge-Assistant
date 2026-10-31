"""Markdown parents and searchable children with exact source ranges and token budgets."""

import json
from uuid import UUID, uuid5

from langchain_core.documents import Document as SplitDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from security_rag.ingestion.chunk_models import (
    ChildChunk,
    ChunkedDocument,
    ChunkingConfig,
    ParentChunk,
    render_context,
    sha256,
)
from security_rag.ingestion.loader import Document
from security_rag.ingestion.markdown import MarkdownStructure
from security_rag.ingestion.tokenizer import ChunkTokenizer

CHUNK_NAMESPACE = UUID("639da67d-5fa9-552b-8170-40c950509bdc")


def _identifier(*parts: object) -> UUID:
    return uuid5(CHUNK_NAMESPACE, json.dumps(parts, separators=(",", ":")))


class MarkdownChunker:
    def __init__(self, tokenizer: ChunkTokenizer, config: ChunkingConfig | None = None) -> None:
        self.tokenizer = tokenizer
        self.config = config or ChunkingConfig(embedding_model=tokenizer.model_name)
        if self.config.embedding_model != tokenizer.model_name:
            raise ValueError("Chunking configuration and tokenizer model must match")
        self.config_hash = self.config.fingerprint(tokenizer.fingerprint)

    def chunk(self, document: Document) -> ChunkedDocument:
        structure = MarkdownStructure(document)
        splitter = _DocumentSplitter(structure, self.tokenizer, document.metadata.title)
        candidates: dict[int, list[tuple[int, int]]] = {}
        for unit in structure.units:
            candidates.setdefault(unit.section_index, []).append((unit.start, unit.end))

        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []
        for section_index, spans in candidates.items():
            if not structure.has_content(spans[0][0], spans[-1][1]):
                continue
            ranges = splitter.partition(spans, self.config.parent_max_tokens)
            for part, (start, end) in enumerate(ranges):
                parent_id = _identifier(
                    document.metadata.document_id,
                    document.metadata.document_hash,
                    self.config_hash,
                    section_index,
                    part,
                )
                parent = ParentChunk(
                    **self._metadata(document, structure, start, end),
                    id=parent_id,
                    parent_index=len(parents),
                    parent_hash=sha256(document.content[start:end]),
                )
                parent_children: list[ChildChunk] = []
                for unit in structure.units:
                    left, right = max(start, unit.start), min(end, unit.end)
                    if left >= right or not structure.has_content(left, right):
                        continue
                    child_ranges = splitter.partition(
                        [(left, right)],
                        self.config.child_max_tokens,
                        self.config.child_overlap_tokens,
                    )
                    for child_start, child_end in child_ranges:
                        text = document.content[child_start:child_end]
                        child_index = len(parent_children)
                        chunk_hash = sha256(text)
                        parent_children.append(
                            ChildChunk(
                                **self._metadata(document, structure, child_start, child_end),
                                id=_identifier(str(parent_id), child_index, chunk_hash),
                                parent_id=parent_id,
                                chunk_index=len(children) + child_index,
                                child_index=child_index,
                                chunk_hash=chunk_hash,
                            )
                        )
                if not parent_children:
                    raise ValueError("Parent contains no usable child content")
                parents.append(parent)
                children.extend(parent_children)
        if not parents:
            raise ValueError("Document contains no usable Markdown body content")
        return ChunkedDocument(parents=tuple(parents), children=tuple(children))

    def _metadata(
        self, document: Document, structure: MarkdownStructure, start: int, end: int
    ) -> dict:
        path = structure.unit_at(start).heading_path
        prefix, suffix = structure.context(start, end)
        return {
            **document.metadata.model_dump(),
            "section": path[-1],
            "heading_path": path,
            "text": document.content[start:end],
            "source_start": start,
            "source_end": end,
            "chunking_config_hash": self.config_hash,
            "markdown_prefix": prefix,
            "markdown_suffix": suffix,
        }


class _DocumentSplitter:
    def __init__(self, structure: MarkdownStructure, tokenizer: ChunkTokenizer, title: str) -> None:
        self.structure = structure
        self.tokenizer = tokenizer
        self.title = title

    def render(self, start: int, end: int) -> str:
        path = self.structure.unit_at(start).heading_path
        prefix, suffix = self.structure.context(start, end)
        return render_context(self.title, path, self.structure.text[start:end], prefix, suffix)

    def fits(self, start: int, end: int, limit: int) -> bool:
        return self.tokenizer.count(self.render(start, end)) <= limit

    def partition(
        self, spans: list[tuple[int, int]], limit: int, overlap: int = 0
    ) -> list[tuple[int, int]]:
        start, end = spans[0][0], spans[-1][1]
        if self.fits(start, end, limit):
            return [(start, end)]
        # A section heading without an introduction belongs with the following subsection.
        content_spans: list[tuple[int, int]] = []
        pending = start
        for left, right in spans:
            if self.structure.has_content(left, right):
                content_spans.append((pending, right))
                pending = right
        if content_spans and pending < end:
            content_spans[-1] = (content_spans[-1][0], end)
        atoms: list[tuple[int, int]] = []
        for left, right in content_spans:
            if self.fits(left, right, limit):
                atoms.append((left, right))
                continue
            for block_start, block_end in self.structure.atomic_spans(left, right):
                atoms.extend(self.split_block(block_start, block_end, limit))
        if not atoms:
            raise ValueError("No body fits the token budget")

        output: list[tuple[int, int]] = []
        start, end = atoms[0]
        for left, right in atoms[1:]:
            if self.fits(start, right, limit):
                end = right
                continue
            output.append((start, end))
            # Keep the following fitting block intact when choosing an overlap suffix.
            start = self.overlap_start(start, end, overlap, right, limit) if overlap else left
            end = right
        output.append((start, end))
        for left, right in output:
            if not self.fits(left, right, limit):
                raise ValueError("Chunk exceeds the complete-input token budget")
            if not self.structure.has_content(left, right):
                raise ValueError("Token budget would create a heading-only chunk")
        return output

    def split_block(self, start: int, end: int, limit: int) -> list[tuple[int, int]]:
        if self.fits(start, end, limit):
            return [(start, end)]
        path = self.structure.unit_at(start).heading_path
        prefix, suffix = self.structure.context(start, end)
        # Include restoration markers even when this original block is still complete.
        for block in self.structure.blocks:
            if block.start <= start < block.end:
                prefix, suffix = block.prefix, block.suffix
                break
        overhead = self.tokenizer.count(render_context(self.title, path, "", prefix, suffix))
        # Exact rendered inputs are checked below; body and prefix counts are not additive.
        budget = limit - overhead
        if budget <= 0:
            raise ValueError("Heading or Markdown context leaves no usable body token budget")
        recursive = RecursiveCharacterTextSplitter(
            chunk_size=budget,
            chunk_overlap=0,
            length_function=lambda value: self.tokenizer.count(value, special_tokens=False),
            separators=[r"\n", r"(?<=[.!?])\s+", r"[ \t]+"],
            is_separator_regex=True,
            keep_separator="end",
            strip_whitespace=False,
        )
        pieces = recursive.split_documents(
            [SplitDocument(page_content=self.structure.text[start:end], metadata={"start": start})]
        )
        output: list[tuple[int, int]] = []
        cursor = start
        pending = start
        for piece in pieces:
            right = cursor + len(piece.page_content)
            if self.structure.text[cursor:right] != piece.page_content:
                raise ValueError("Recursive splitter changed original source text")
            if self.structure.has_content(pending, right):
                output.extend(self.token_fallback(pending, right, limit))
                pending = right
            cursor = right
        if cursor != end:
            raise ValueError("Recursive splitter omitted original source text")
        if pending < end and output:
            left, _ = output.pop()
            output.extend(self.token_fallback(left, end, limit))
        return output

    def token_fallback(self, start: int, end: int, limit: int) -> list[tuple[int, int]]:
        output: list[tuple[int, int]] = []
        while start < end:
            if self.fits(start, end, limit):
                output.append((start, end))
                break
            offsets = self.tokenizer.offsets(self.structure.text[start:end])
            stops = sorted({start + right for _, right in offsets if right > 0})
            # Bound by real token offsets; never decode tokens back into source text.
            stops = stops[:limit]
            while stops and not self.fits(start, stops[-1], limit):
                stops.pop()
            if not stops:
                raise ValueError("Token budget cannot hold a source token with its context")
            right = stops[-1]
            output.append((start, right))
            start = right
        return output

    def overlap_start(self, start: int, end: int, overlap: int, next_end: int, limit: int) -> int:
        offsets = self.tokenizer.offsets(self.structure.text[start:end])
        if not offsets or not overlap:
            return end
        # Try progressively smaller suffixes, rather than dropping all overlap when
        # the full allowance does not fit. Token offsets retain the original Unicode.
        candidates = sorted({start + left for left, _ in offsets if 0 < left < end - start})
        for candidate in candidates[-overlap:]:
            if self.tokenizer.count(
                self.structure.text[candidate:end], special_tokens=False
            ) <= overlap and self.fits(candidate, next_end, limit):
                return candidate
        return end
