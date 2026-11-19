import json
from functools import partial
from unittest.mock import MagicMock

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama
from ollama import Client, ResponseError
from pydantic import ValidationError

from security_rag.config import Settings
from security_rag.core.application.answer import GenerationUnavailable
from security_rag.infrastructure.generation.answer_generator import LangChainGenerator
from security_rag.infrastructure.generation.model_factory import create_local_chat_model


def generator_with_local_model(settings):
    return LangChainGenerator(settings, partial(create_local_chat_model, settings=settings))


def completion(content, *, done_reason="stop"):
    return httpx.Response(
        200,
        headers={"content-type": "application/x-ndjson"},
        text=json.dumps(
            {
                "model": "qwen3:4b",
                "created_at": "2026-11-19T00:00:00Z",
                "message": {"role": "assistant", "content": content},
                "done": True,
                "done_reason": done_reason,
                "prompt_eval_count": 123,
                "eval_count": 12,
            }
        )
        + "\n",
    )


def mock_ollama(monkeypatch, serve):
    monkeypatch.setattr(
        "langchain_ollama.chat_models.Client",
        partial(Client, transport=httpx.MockTransport(serve)),
    )


def test_langchain_uses_native_ollama_chat_and_configured_budgets(settings, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-cloud-key")
    requests = []

    def serve(request):
        requests.append((request, json.loads(request.content)))
        return completion('{"answer":"Use isolation [S1].","insufficient_evidence":false}')

    mock_ollama(monkeypatch, serve)
    settings = Settings(llm_model="qwen3:4b")
    generator = generator_with_local_model(settings)
    estimate = generator.estimate_input_tokens("Question {x}", "[S1]\nReference")
    assert not requests
    result = generator.generate("Question {x}", "[S1]\nReference")
    generator.generate("Another question", "[S1]\nReference")
    assert result.answer == "Use isolation [S1]."
    assert len(requests) == 2
    request, body = requests[0]
    assert str(request.url) == "http://localhost:11434/api/chat"
    assert "ambient-cloud-key" not in str(request.headers)
    assert body["model"] == "qwen3:4b"
    assert body["options"]["num_predict"] == settings.llm_max_output_tokens
    assert body["options"]["num_ctx"] == settings.llm_context_window
    assert body["think"] is False
    assert body["format"] == "json"
    assert "untrusted data" in body["messages"][0]["content"]
    assert "Question {x}" in body["messages"][1]["content"]
    assert estimate > sum(len(message["content"].encode("utf-8")) for message in body["messages"])


def test_factory_returns_chat_ollama_without_startup_requests(settings, monkeypatch):
    def unexpected(request):
        pytest.fail("Constructing the model must not call Ollama")

    mock_ollama(monkeypatch, unexpected)
    model = create_local_chat_model("qwen3:4b", settings=settings)
    assert isinstance(model, ChatOllama)
    assert model.client_kwargs["timeout"] == settings.llm_timeout_seconds


def test_prompt_estimate_is_local_conservative_and_counts_utf8_bytes(settings):
    factory = MagicMock()
    generator = LangChainGenerator(settings, factory)
    baseline = generator.estimate_input_tokens("Question", "")
    assert baseline > 256
    for context in ("plain text", "سؤال امنیتی", "🔐" * 20):
        assert generator.estimate_input_tokens("Question", context) >= (
            baseline + len(context.encode("utf-8"))
        )
    factory.assert_not_called()


def test_generator_requires_explicit_model_and_does_no_startup_io(settings):
    factory = MagicMock()
    generator = LangChainGenerator(settings, factory)
    with pytest.raises(GenerationUnavailable, match="LLM_MODEL"):
        generator.generate("Question", "Context")
    factory.assert_not_called()


@pytest.mark.parametrize("done_reason", ["length", "load", None])
def test_truncated_or_incomplete_ollama_answers_are_rejected(settings, monkeypatch, done_reason):
    mock_ollama(
        monkeypatch,
        lambda request: completion(
            '{"answer":"Partial [S1].","insufficient_evidence":false}', done_reason=done_reason
        ),
    )
    generator = generator_with_local_model(Settings(llm_model="qwen3:4b"))
    with pytest.raises(GenerationUnavailable, match="complete answer"):
        generator.generate("Question", "Context")


@pytest.mark.parametrize("status", [404, 500])
def test_ollama_dependency_errors_propagate_without_fallback(settings, monkeypatch, status):
    mock_ollama(monkeypatch, lambda request: httpx.Response(status, json={"error": "unavailable"}))
    generator = generator_with_local_model(Settings(llm_model="qwen3:4b"))
    with pytest.raises(ResponseError, match="unavailable"):
        generator.generate("Question", "Context")


def test_generator_accepts_a_generic_chat_model_and_reuses_it(settings):
    model = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content='{"answer":"Use isolation [S1].","insufficient_evidence":false}',
                response_metadata={"finish_reason": "stop"},
            )
        ]
    )
    factory = MagicMock(return_value=model)
    generator = LangChainGenerator(Settings(llm_model="another-model"), factory)
    factory.assert_not_called()
    assert generator.generate("Question", "[S1]\nReference").answer == "Use isolation [S1]."
    assert generator.generate("Another question", "[S1]\nReference").answer
    factory.assert_called_once_with("another-model")


@pytest.mark.parametrize(
    "values",
    [
        {"llm_context_window": 0},
        {"llm_max_output_tokens": 0},
        {"llm_context_window": 512, "llm_max_output_tokens": 512},
        {"llm_timeout_seconds": float("inf")},
    ],
)
def test_invalid_generation_settings(settings, values):
    with pytest.raises(ValidationError):
        Settings(**values)
