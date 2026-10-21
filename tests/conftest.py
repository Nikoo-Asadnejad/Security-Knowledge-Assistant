import pytest

from security_rag.config import Settings


@pytest.fixture
def settings(tmp_path, monkeypatch):
    # Isolate both .env lookup and inherited application settings.
    monkeypatch.chdir(tmp_path)
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    return Settings(data_dir=tmp_path / "owasp", download_retries=0)


@pytest.fixture
def markdown():
    def content(name, body="Original content.\n"):
        code, title = name.removesuffix(".md").split("_", 1)
        return f"## {code}:2026 {title}\n\n### Description\n{body}".encode()

    return content
