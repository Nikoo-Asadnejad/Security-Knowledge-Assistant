"""Build the concrete local connector behind LangChain's generic model interface."""

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

from security_rag.config import Settings


def create_local_chat_model(model_name: str, *, settings: Settings) -> BaseChatModel:
    return ChatOllama(
        model=model_name,
        base_url=str(settings.llm_base_url),
        client_kwargs={"timeout": settings.llm_timeout_seconds},
        temperature=0,
        num_ctx=settings.llm_context_window,
        num_predict=settings.llm_max_output_tokens,
        reasoning=False,
        format="json",
        validate_model_on_init=False,
    )
