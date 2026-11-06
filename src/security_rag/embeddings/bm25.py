"""Shared lexical analysis and local BM25 weighting; Qdrant supplies IDF."""

import json
import math
import re
from importlib.metadata import version
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import models

from security_rag.config import Settings
from security_rag.ingestion.chunk_models import ChildChunk, sha256

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


def tokens(text: str, token_max_length: int) -> list[str]:
    from fastembed.sparse.bm25 import SimpleTokenizer, remove_non_alphanumeric

    result = SimpleTokenizer.tokenize(remove_non_alphanumeric(preprocess(text)))
    if any(len(token) > token_max_length for token in result):
        raise ValueError("Lexical token exceeds BM25_TOKEN_MAX_LENGTH")
    return result


class SparseConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = "Qdrant/bm25"
    fastembed_version: str = Field(default_factory=lambda: version("fastembed"))
    analyzer: str = "technical-v1-code-sha256-alias-32"
    lexical_format: str = "code-unique-headings-rendered-body-v1"
    language: str = "english"
    disable_stemmer: bool = True
    stopwords: tuple[str, ...] = ()
    token_max_length: int = Field(default=512, ge=40, le=8192)
    k: float = Field(default=1.2, gt=0, allow_inf_nan=False)
    b: float = Field(default=0.75, ge=0, le=1, allow_inf_nan=False)
    avg_len: float = Field(gt=0, allow_inf_nan=False)

    @classmethod
    def from_settings(cls, settings: Settings, avg_len: float) -> "SparseConfig":
        return cls(
            k=settings.bm25_k,
            b=settings.bm25_b,
            token_max_length=settings.bm25_token_max_length,
            avg_len=avg_len,
        )

    @property
    def fingerprint(self) -> str:
        return sha256(json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":")))


def validate_sparse(vector: models.SparseVector, *, allow_empty: bool = False) -> None:
    if (
        len(vector.indices) != len(vector.values)
        or (not vector.indices and not allow_empty)
        or len(set(vector.indices)) != len(vector.indices)
        or any(index < 0 or index > 2**32 - 1 for index in vector.indices)
        or any(not math.isfinite(value) or value <= 0 for value in vector.values)
    ):
        raise ValueError("Expected unique nonnegative sparse indices and finite positive weights")


class BM25Encoder:
    def __init__(self, config: SparseConfig, cache_dir: Path) -> None:
        from fastembed import SparseTextEmbedding

        self.config = config
        self.model = SparseTextEmbedding(
            model_name=config.model,
            cache_dir=str(cache_dir),
            k=config.k,
            b=config.b,
            avg_len=config.avg_len,
            language=config.language,
            disable_stemmer=config.disable_stemmer,
            stopwords=set(config.stopwords),
            token_max_length=config.token_max_length,
        )

    def encode(self, texts: list[str], *, query: bool = False) -> list[models.SparseVector]:
        for text in texts:
            if not tokens(text, self.config.token_max_length) and not query:
                raise ValueError("Child contains no lexical terms")
        prepared = [preprocess(text) for text in texts]
        encoded = (
            self.model.query_embed(prepared)
            if query
            else self.model.embed(prepared, batch_size=max(1, len(prepared)))
        )
        vectors = []
        for result in encoded:
            vector = models.SparseVector(
                indices=result.indices.tolist(), values=result.values.tolist()
            )
            validate_sparse(vector, allow_empty=query)
            vectors.append(vector)
        if len(vectors) != len(texts):
            raise ValueError("Sparse embedding count does not match input count")
        return vectors
