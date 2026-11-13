"""Compose CLI use cases with their concrete adapters and manage resources."""

from __future__ import annotations

from contextlib import closing
from functools import partial
from typing import TYPE_CHECKING

from security_rag.config import Settings
from security_rag.core.application.ingest import IngestionPipeline, IngestionReport
from security_rag.core.application.preview import preview_chunks
from security_rag.core.application.sparse_config import configure_sparse
from security_rag.core.domain.documents import Document
from security_rag.core.domain.manifest import DOCUMENT_FILES
from security_rag.infrastructure.download import download_documents as download_documents
from security_rag.infrastructure.files import load_document, load_documents

if TYPE_CHECKING:
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


def _chunker(settings: Settings) -> MarkdownChunker:
    from security_rag.infrastructure.embeddings.fastembed.tokenizer import ChunkTokenizer
    from security_rag.infrastructure.markdown_chunker import MarkdownChunker

    tokenizer = ChunkTokenizer.load(
        settings.embedding_model, path=settings.tokenizer_path, revision=settings.tokenizer_revision
    )
    return MarkdownChunker(tokenizer)


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

    def load(name: str) -> Document:
        return load_document(settings.data_dir / name)

    lexical_children = (
        child for name in DOCUMENT_FILES for child in chunker.chunk(load(name)).children
    )
    with closing(PostgresStore.from_settings(settings)) as parents:
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
                Embedder(settings, chunker.tokenizer),
                sparse,
                batch_size=settings.ingestion_batch_size,
            ).ingest(DOCUMENT_FILES, load)
