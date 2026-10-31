from pathlib import Path

import pytest
from pydantic import ValidationError

from security_rag.config import Settings


def test_defaults_need_no_llm_credentials(settings):
    assert settings.qdrant_collection == "owasp_security_2026"
    assert str(settings.qdrant_url) == "http://localhost:6333/"
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.tokenizer_path is None
    assert settings.tokenizer_revision == "main"
    assert settings.llm_api_key.get_secret_value() == ""
    assert settings.llm_model == ""
    assert Settings().data_dir == Path("data/owasp")


def test_environment_overrides_dotenv(settings, monkeypatch):
    Path(".env").write_text(
        "DATA_DIR=custom/docs\nDOWNLOAD_RETRIES=3\n"
        "QDRANT_COLLECTION=custom_collection\nLLM_API_KEY=local-secret\n"
        "UNRELATED_VARIABLE=ignored\n"
    )
    monkeypatch.setenv("DOWNLOAD_RETRIES", "1")
    loaded = Settings()
    assert loaded.data_dir == Path("custom/docs")
    assert loaded.download_retries == 1
    assert loaded.qdrant_collection == "custom_collection"
    assert loaded.llm_api_key.get_secret_value() == "local-secret"
    assert "local-secret" not in repr(loaded)
    assert "local-secret" not in loaded.model_dump_json()


def test_database_url_is_not_serialized_in_reports(settings, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:private-password@localhost/db")
    loaded = Settings()
    assert "private-password" not in repr(loaded)
    assert "private-password" not in loaded.model_dump_json()


@pytest.mark.parametrize(
    "values",
    [
        {"download_timeout_seconds": 0},
        {"download_timeout_seconds": -1},
        {"download_timeout_seconds": float("inf")},
        {"download_retries": -1},
        {"download_retries": 6},
        {"qdrant_url": "not-a-url"},
        {"llm_base_url": "ftp://example.com"},
        {"qdrant_collection": " "},
        {"embedding_model": ""},
        {"tokenizer_revision": " "},
        {"database_pool_size": 0},
        {"database_pool_size": 17},
        {"database_timeout_seconds": 0},
        {"database_timeout_seconds": float("inf")},
        {"ingestion_batch_size": 0},
        {"ingestion_batch_size": 257},
    ],
)
def test_invalid_settings_are_rejected(settings, values):
    with pytest.raises(ValidationError):
        Settings(**values)
