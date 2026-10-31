"""Small, source-preserving ATX Markdown scanner for the OWASP corpus."""

import re
from dataclasses import dataclass

from security_rag.ingestion.loader import Document

HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)\s*$")
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})([^\r\n]*)")
LIST_ITEM = re.compile(r"^ {0,3}(?:[-+*]|\d+[.)])[ \t]+")
TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")


@dataclass(frozen=True)
class MarkdownUnit:
    start: int
    end: int
    heading_path: tuple[str, ...]
    section_index: int


@dataclass(frozen=True)
class MarkdownBlock:
    start: int
    end: int
    body_start: int
    body_end: int
    prefix: str
    suffix: str = ""


class MarkdownStructure:
    def __init__(self, document: Document) -> None:
        self.text = document.content
        self.blocks: list[MarkdownBlock] = []
        headings: list[tuple[int, int, int, str]] = []
        lines = self.text.splitlines(keepends=True)
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line))

        index = 0
        while index < len(lines):
            line = lines[index].lstrip("\ufeff") if index == 0 else lines[index]
            fence = FENCE.match(line)
            if fence and not (fence[1].startswith("`") and "`" in fence[2]):
                opening = index
                marker = fence[1]
                closing = re.compile(rf"^ {{0,3}}{re.escape(marker[0])}{{{len(marker)},}}[ \t]*$")
                index += 1
                while index < len(lines) and not closing.match(lines[index].rstrip("\r\n")):
                    index += 1
                body_end = offsets[index]
                if index < len(lines):
                    index += 1
                self.blocks.append(
                    MarkdownBlock(
                        offsets[opening],
                        offsets[index],
                        offsets[opening + 1],
                        body_end,
                        lines[opening].rstrip("\r\n") + "\n",
                        "\n" + marker,
                    )
                )
                continue
            heading = HEADING.match(line.rstrip("\r\n"))
            if heading:
                title = re.sub(r"[ \t]+#+[ \t]*$", "", heading[2]).strip()
                headings.append((offsets[index], offsets[index + 1], len(heading[1]), title))
            elif (
                index + 1 < len(lines)
                and "|" in line
                and TABLE_RULE.match(lines[index + 1].rstrip("\r\n"))
            ):
                opening = index
                index += 2
                body_start = offsets[index]
                while index < len(lines) and "|" in lines[index] and lines[index].strip():
                    index += 1
                self.blocks.append(
                    MarkdownBlock(
                        offsets[opening],
                        offsets[index],
                        body_start,
                        offsets[index],
                        "".join(lines[opening : opening + 2]),
                    )
                )
                continue
            index += 1

        if not headings:
            raise ValueError("Document has no ATX title")
        _, self.content_start, title_level, _ = headings[0]
        self.units: list[MarkdownUnit] = []
        stack: list[tuple[int, str]] = []
        section_index = 0
        pending = self.content_start
        path = (document.metadata.title,)
        for start, _end, level, title in headings[1:]:
            if start > pending:
                self.units.append(MarkdownUnit(pending, start, path, section_index))
            while stack and stack[-1][0] >= level:
                stack.pop()
            if not stack or level <= title_level:
                stack.clear()
                section_index += 1
            stack.append((level, title))
            path = (document.metadata.title, *(item[1] for item in stack))
            pending = start
        if pending < len(self.text):
            self.units.append(MarkdownUnit(pending, len(self.text), path, section_index))

    def unit_at(self, start: int) -> MarkdownUnit:
        return next(unit for unit in self.units if unit.start <= start < unit.end)

    def has_content(self, start: int, end: int) -> bool:
        for block in self.blocks:
            if block.start < end and start < block.end:
                return True
        return any(
            line.strip() and not HEADING.match(line) and not re.fullmatch(r"\s*[-*_]{3,}\s*", line)
            for line in self.text[start:end].splitlines()
        )

    def context(self, start: int, end: int) -> tuple[str, str]:
        prefix = suffix = ""
        for block in self.blocks:
            if block.body_start <= start < block.body_end:
                prefix = block.prefix
            if block.suffix and block.body_start <= end <= block.body_end:
                suffix = block.suffix
        return prefix, suffix

    def atomic_spans(self, start: int, end: int) -> list[tuple[int, int]]:
        """Paragraphs/list items, keeping fitting tables and fenced blocks intact."""
        boundaries = {start, end}
        position = start
        for line in self.text[start:end].splitlines(keepends=True):
            if not line.strip():
                boundaries.add(position + len(line))
            elif LIST_ITEM.match(line):
                boundaries.add(position)
            position += len(line)
        for block in self.blocks:
            if block.start < end and start < block.end:
                boundaries = {p for p in boundaries if not block.start < p < block.end}
                boundaries.update((max(start, block.start), min(end, block.end)))
        points = sorted(boundaries)
        spans: list[tuple[int, int]] = []
        pending = start
        for right in points[1:]:
            # Attach leading headings and blank lines to a following body block.
            if self.has_content(pending, right):
                spans.append((pending, right))
                pending = right
        if pending < end and spans:
            spans[-1] = (spans[-1][0], end)
        return spans
