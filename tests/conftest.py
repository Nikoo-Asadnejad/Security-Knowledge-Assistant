from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors

from security_rag.config import Settings
from security_rag.ingestion.loader import parse_document
from security_rag.ingestion.tokenizer import ChunkTokenizer


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


@pytest.fixture
def raw_tokenizer():
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "alpha", "beta", "gamma", "delta"]
    tokenizer = Tokenizer(models.WordPiece({word: i for i, word in enumerate(words)}))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    return tokenizer


@pytest.fixture
def tokenizer(raw_tokenizer):
    return ChunkTokenizer(raw_tokenizer, "BAAI/bge-small-en-v1.5")


@pytest.fixture
def make_document():
    def make(body, title_level=2, title="Prompt Injection"):
        text = f"{'#' * title_level} LLM01:2026 {title}\n\n{body}"
        return parse_document(Path("LLM01_PromptInjection.md"), text.encode())

    return make
