"""Environment-backed application settings; loading settings performs no I/O beyond .env."""

from pathlib import Path
from typing import Annotated

from pydantic import Field, HttpUrl, SecretStr, StringConstraints
from pydantic_settings import BaseSettings, SettingsConfigDict

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
    embedding_cache_dir: Path = Path("data/models")
    embedding_model: NonEmptyString = "BAAI/bge-small-en-v1.5"
    tokenizer_path: Path | None = None
    tokenizer_revision: NonEmptyString = "main"
    llm_base_url: HttpUrl = HttpUrl("https://api.openai.com/v1")
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = ""
