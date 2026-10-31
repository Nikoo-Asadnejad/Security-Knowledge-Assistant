from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from security_rag.embeddings.embedder import Embedder, validate_vectors
from security_rag.ingestion.chunker import MarkdownChunker
from security_rag.ingestion.tokenizer import ChunkTokenizer


def test_model_is_lazy_reused_cpu_only_and_does_not_truncate(settings, tokenizer, make_document):
    children = MarkdownChunker(tokenizer).chunk(make_document("### Test\nalpha beta.\n")).children
    embedder = Embedder(settings, tokenizer)
    runtime = tokenizer._tokenizer
    model = MagicMock()
    model.model.tokenizer = runtime
    model.embed.return_value = [np.ones(384)]
    with patch("fastembed.TextEmbedding", return_value=model) as factory:
        assert not factory.called
        assert len(embedder.embed(children)[0]) == 384
        embedder.embed(children)
        factory.assert_called_once()
        assert factory.call_args.kwargs["providers"] == ["CPUExecutionProvider"]
        assert runtime.truncation is None
        assert model.embed.call_args.args[0] == [children[0].embedding_text]


def test_runtime_tokenizer_mismatch_is_rejected(settings, tokenizer, raw_tokenizer, make_document):
    children = MarkdownChunker(tokenizer).chunk(make_document("### Test\nalpha.\n")).children
    raw_tokenizer.add_tokens(["extra"])
    assert (
        ChunkTokenizer(raw_tokenizer, settings.embedding_model).fingerprint != tokenizer.fingerprint
    )
    model = MagicMock()
    model.model.tokenizer = raw_tokenizer
    with patch("fastembed.TextEmbedding", return_value=model):
        with pytest.raises(ValueError, match="differs"):
            Embedder(settings, tokenizer).embed(children)
    model.embed.assert_not_called()


@pytest.mark.parametrize(
    "vectors,expected",
    [
        ([], 1),
        ([[0.0] * 384], 1),
        ([[1.0] * 383], 1),
        ([[float("nan")] * 384], 1),
        ([[float("inf")] * 384], 1),
    ],
)
def test_invalid_embedding_output_is_rejected(vectors, expected):
    with pytest.raises(ValueError):
        validate_vectors(vectors, expected)


def test_unsupported_model_and_oversized_batch_rejected(settings, tokenizer, make_document):
    with pytest.raises(ValueError, match="supports only"):
        Embedder(settings.model_copy(update={"embedding_model": "other"}), tokenizer)
    children = MarkdownChunker(tokenizer).chunk(make_document("### Test\nalpha.\n")).children
    embedder = Embedder(settings.model_copy(update={"ingestion_batch_size": 1}), tokenizer)
    with pytest.raises(ValueError, match="bounded batch"):
        embedder.embed(children * 2)
    long_child = children[0].model_copy(update={"text": "alpha " * 500})
    with pytest.raises(ValueError, match="450 tokens"):
        embedder.embed((long_child,))
