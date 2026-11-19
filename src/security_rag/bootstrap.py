"""Compose CLI use cases with their concrete adapters and manage resources."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from functools import partial
from threading import Lock
from typing import TYPE_CHECKING

from security_rag.config import Settings
from security_rag.core.application.ingest import IngestionPipeline, IngestionReport
from security_rag.core.application.preview import preview_chunks
from security_rag.core.application.sparse_config import configure_sparse
from security_rag.core.domain.chunks import ChunkingConfig
from security_rag.core.domain.documents import Document
from security_rag.core.domain.manifest import DOCUMENT_FILES
from security_rag.infrastructure.download import download_documents as download_documents
from security_rag.infrastructure.files import load_document, load_documents

if TYPE_CHECKING:
    from uuid import UUID

    from security_rag.core.application.answer import AnswerResult
    from security_rag.core.application.ports import SearchFilters
    from security_rag.core.application.search import RetrievalResult, SearchResult
    from security_rag.core.domain.chunks import ParentChunk
    from security_rag.infrastructure.embeddings.fastembed.tokenizer import ChunkTokenizer
    from security_rag.infrastructure.markdown_chunker import MarkdownChunker


def migrate_database(settings: Settings) -> dict:
    from security_rag.infrastructure.persistence.postgres.postgres_store import PostgresStore

    with closing(PostgresStore.from_settings(settings)) as parents:
        return {"applied": parents.migrate()}


def storage_statistics(settings: Settings) -> dict:
    from qdrant_client import QdrantClient

    from security_rag.infrastructure.persistence.postgres.postgres_store import PostgresStore
    from security_rag.infrastructure.persistence.qdrant import QdrantStore

    with (
        closing(PostgresStore.from_settings(settings)) as parents,
        closing(QdrantClient(url=str(settings.qdrant_url), timeout=30)) as client,
    ):
        sparse_config = parents.sparse_config()
        return {
            "ingestions": parents.statistics(),
            "sparse_configuration": sparse_config.model_dump() if sparse_config else None,
            "qdrant": QdrantStore(client, settings.qdrant_collection).statistics(),
        }


def chunk_documents(settings: Settings) -> dict:
    documents = load_documents(settings.data_dir)
    return preview_chunks(documents, _chunker(settings))


def _tokenizer(settings: Settings) -> ChunkTokenizer:
    from security_rag.infrastructure.embeddings.fastembed.tokenizer import ChunkTokenizer

    return ChunkTokenizer.load(
        settings.embedding_model, path=settings.tokenizer_path, revision=settings.tokenizer_revision
    )


def _chunker(settings: Settings) -> MarkdownChunker:
    from security_rag.infrastructure.markdown_chunker import MarkdownChunker

    config = ChunkingConfig(
        child_max_tokens=settings.child_max_tokens, embedding_model=settings.embedding_model
    )
    return MarkdownChunker(_tokenizer(settings), config)


def ingest_documents(settings: Settings) -> IngestionReport:
    from qdrant_client import QdrantClient

    from security_rag.infrastructure.embeddings.fastembed.bm25 import (
        BM25Encoder,
        config_for_average,
        tokens,
    )
    from security_rag.infrastructure.embeddings.fastembed.embedder import Embedder
    from security_rag.infrastructure.persistence.postgres.postgres_store import PostgresStore
    from security_rag.infrastructure.persistence.qdrant import QdrantStore

    for name in DOCUMENT_FILES:
        if not (settings.data_dir / name).is_file():
            raise ValueError(f"Missing document {name}; run download first")

    chunker = _chunker(settings)
    dense = Embedder(settings, chunker.tokenizer)

    def load(name: str) -> Document:
        return load_document(settings.data_dir / name)

    lexical_children = (
        child for name in DOCUMENT_FILES for child in chunker.chunk(load(name)).children
    )
    with (
        closing(PostgresStore.from_settings(settings)) as parents,
        parents.single_writer(),
    ):
        sparse = configure_sparse(
            parents,
            lexical_children,
            count_terms=lambda text: len(tokens(text, settings.bm25_token_max_length)),
            build_config=partial(
                config_for_average,
                k=settings.bm25_k,
                b=settings.bm25_b,
                token_max_length=settings.bm25_token_max_length,
            ),
            build_encoder=lambda config: BM25Encoder(config, settings.embedding_cache_dir),
        )
        with closing(QdrantClient(url=str(settings.qdrant_url), timeout=30)) as client:
            return IngestionPipeline(
                chunker,
                parents,
                QdrantStore(client, settings.qdrant_collection, settings.ingestion_batch_size),
                dense,
                sparse,
                batch_size=settings.ingestion_batch_size,
            ).ingest(DOCUMENT_FILES, load)


@contextmanager
def search_runtime(settings: Settings) -> Iterator[Callable[[str, SearchFilters], SearchResult]]:
    with _retrieval_runtime(settings) as (retrieve, _):
        yield lambda query, filters: retrieve(query, filters).search_result()


@dataclass(frozen=True)
class ApplicationServices:
    search: Callable[[str, SearchFilters], SearchResult]
    ask: Callable[[str, SearchFilters], AnswerResult]


@contextmanager
def application_runtime(settings: Settings) -> Iterator[ApplicationServices]:
    from security_rag.core.application.answer import Ask
    from security_rag.infrastructure.generation.answer_generator import LangChainGenerator
    from security_rag.infrastructure.generation.model_factory import create_local_chat_model

    with _retrieval_runtime(settings) as (retrieve, fetch_parents):
        ask = Ask(
            retrieve,
            fetch_parents,
            LangChainGenerator(
                settings,
                partial(create_local_chat_model, settings=settings),
            ),
            context_window=settings.llm_context_window,
            max_output_tokens=settings.llm_max_output_tokens,
        )
        yield ApplicationServices(
            search=lambda query, filters: retrieve(query, filters).search_result(), ask=ask.run
        )


@contextmanager
def _retrieval_runtime(
    settings: Settings,
) -> Iterator[
    tuple[
        Callable[[str, SearchFilters], RetrievalResult],
        Callable[[list[UUID]], dict[UUID, ParentChunk]],
    ]
]:
    from qdrant_client import QdrantClient

    from security_rag.core.application.search import Search, SearchUnavailable, completed_corpus
    from security_rag.infrastructure.embeddings.fastembed.bm25 import config_for_average
    from security_rag.infrastructure.embeddings.fastembed.embedder import Embedder
    from security_rag.infrastructure.embeddings.fastembed.reranker import CrossEncoder
    from security_rag.infrastructure.embeddings.fastembed.search_encoder import SearchEncoder
    from security_rag.infrastructure.persistence.postgres.postgres_store import PostgresStore
    from security_rag.infrastructure.persistence.qdrant import QdrantStore

    with (
        closing(PostgresStore.from_settings(settings, check_connection=False)) as parents,
        closing(QdrantClient(url=str(settings.qdrant_url), timeout=30)) as client,
    ):
        store = QdrantStore(client, settings.qdrant_collection)
        reranker = CrossEncoder(settings)
        encoder = None
        chunking_hash = None
        model_lock = Lock()

        def run(query: str, filters: SearchFilters) -> RetrievalResult:
            nonlocal encoder, chunking_hash
            with parents.search_snapshot() as (config, states):
                if config is None or config != config_for_average(
                    config.avg_len,
                    k=settings.bm25_k,
                    b=settings.bm25_b,
                    token_max_length=settings.bm25_token_max_length,
                ):
                    raise SearchUnavailable("Sparse configuration is missing or incompatible")
                with model_lock:
                    if encoder is None:
                        chunker = _chunker(settings)
                        encoder = SearchEncoder(Embedder(settings, chunker.tokenizer), settings)
                        chunking_hash = chunker.config_hash
                config_hash = config.fingerprint
                revisions, count = completed_corpus(states, config_hash, chunking_hash)
                index = store.search_index(revisions, config_hash, count)
                return Search(encoder.for_config(config), index, reranker).retrieve(query, filters)

        yield run, parents.get_parents
