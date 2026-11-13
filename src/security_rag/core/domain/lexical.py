"""Shared lexical input and identifier normalization for indexing and retrieval."""

import re

from security_rag.core.domain.chunks import ChildChunk, sha256

IDENTIFIER = re.compile(
    r"(?<![\w-])(?:LLM\d{2}|CWE-\d{1,8}|CVE-\d{4}-\d{4,10}|"
    r"ERR_[A-Z0-9]+(?:_[A-Z0-9]+)*|0x[0-9a-f]{1,16})(?![\w-])",
    re.IGNORECASE,
)


def identifiers(text: str) -> list[str]:
    return sorted(
        {
            "0x" + match[2:].upper() if match.lower().startswith("0x") else match.upper()
            for match in IDENTIFIER.findall(text)
        }
    )


def preprocess(text: str) -> str:
    aliases = [
        "code" + sha256(code)[:32] for code in identifiers(text) if "-" in code or "_" in code
    ]
    return "\n".join([text.lower(), *aliases])


def lexical_text(child: ChildChunk) -> str:
    headings = list(dict.fromkeys((child.title, *child.heading_path)))
    # The source slice can start with a heading already included in metadata.
    body = child.markdown_prefix + child.text + child.markdown_suffix
    lines = []
    fence = None
    for line in body.splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line.rstrip("\r\n"))
        if marker:
            if fence is None:
                fence = marker[1]
            elif (
                marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip()
            ):
                fence = None
            lines.append(line)
            continue
        heading = re.match(r"^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        if fence is not None or heading is None or heading[1] not in headings:
            lines.append(line)
    return "\n".join([child.document_code, *headings, "".join(lines)])
