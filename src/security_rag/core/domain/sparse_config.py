"""Persisted configuration and fingerprint for sparse indexing."""

import json

from pydantic import BaseModel, ConfigDict, Field

from security_rag.core.domain.chunks import sha256


class SparseConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = "Qdrant/bm25"
    fastembed_version: str
    analyzer: str = "technical-v1-code-sha256-alias-32"
    lexical_format: str = "code-unique-headings-rendered-body-v1"
    language: str = "english"
    disable_stemmer: bool = True
    stopwords: tuple[str, ...] = ()
    token_max_length: int = Field(default=512, ge=40, le=8192)
    k: float = Field(default=1.2, gt=0, allow_inf_nan=False)
    b: float = Field(default=0.75, ge=0, le=1, allow_inf_nan=False)
    avg_len: float = Field(gt=0, allow_inf_nan=False)

    @property
    def fingerprint(self) -> str:
        return sha256(json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":")))
