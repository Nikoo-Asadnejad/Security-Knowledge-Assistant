"""Environment-backed application settings; loading settings performs no I/O beyond .env."""

from pathlib import Path
from typing import Annotated

from pydantic import Field, HttpUrl, SecretStr, StringConstraints, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from security_rag.core.domain.chunks import DEFAULT_CHILD_MAX_TOKENS, EMBEDDING_MAX_TOKENS

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )

    data_dir: Path = Path("data/owasp")
    download_timeout_seconds: float = Field(default=30.0, gt=0, allow_inf_nan=False)
    download_retries: int = Field(default=2, ge=0, le=5)
    qdrant_url: HttpUrl = HttpUrl("http://localhost:6333")
    qdrant_collection: NonEmptyString = "owasp_security_2026"
    database_url: SecretStr = SecretStr("")
    database_pool_size: int = Field(default=4, ge=1, le=16)
    database_timeout_seconds: float = Field(default=10, gt=0, allow_inf_nan=False)
    ingestion_batch_size: int = Field(default=32, ge=1, le=256)
    child_max_tokens: int = Field(default=DEFAULT_CHILD_MAX_TOKENS, gt=60, le=EMBEDDING_MAX_TOKENS)
    reranker_model: NonEmptyString = "Xenova/ms-marco-MiniLM-L-6-v2"
    embedding_cache_dir: Path = Path("data/models")
    embedding_model: NonEmptyString = "BAAI/bge-small-en-v1.5"
    bm25_k: float = Field(default=1.2, gt=0, allow_inf_nan=False)
    bm25_b: float = Field(default=0.75, ge=0, le=1, allow_inf_nan=False)
    bm25_token_max_length: int = Field(default=512, ge=40, le=8192)
    tokenizer_path: Path | None = None
    tokenizer_revision: NonEmptyString = "main"
    llm_base_url: HttpUrl = HttpUrl("http://localhost:11434")
    llm_model: str = ""
    llm_context_window: int = Field(default=4096, ge=512)
    llm_max_output_tokens: int = Field(default=512, gt=0)
    llm_timeout_seconds: float = Field(default=120, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_generation_budget(self) -> "Settings":
        if self.llm_max_output_tokens >= self.llm_context_window:
            raise ValueError("LLM_MAX_OUTPUT_TOKENS must be smaller than LLM_CONTEXT_WINDOW")
        return self
