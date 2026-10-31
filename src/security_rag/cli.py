"""Command-line entry point for the project."""

import argparse
import json
import logging
from contextlib import closing
from importlib.metadata import version

from security_rag.config import Settings
from security_rag.ingestion.downloader import DownloadError, download_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Security Knowledge Assistant")
    parser.add_argument("--version", action="version", version=version("security-rag"))
    commands = parser.add_subparsers(dest="command")
    download = commands.add_parser("download", help="Download the ten OWASP Markdown documents")
    download.add_argument("--refresh", action="store_true", help="Replace existing documents")
    commands.add_parser("chunk", help="Validate local document chunks and print statistics")
    commands.add_parser("migrate", help="Apply versioned PostgreSQL schema migrations")
    commands.add_parser(
        "ingest", help="Persist parents and embed/index children (API must be stopped)"
    )
    commands.add_parser("stats", help="Print persisted ingestion state and collection counts")
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    try:
        settings = Settings()
        if args.command in {"migrate", "ingest", "stats"}:
            logging.basicConfig(level=logging.INFO, format="%(message)s")
            logging.getLogger("httpx").setLevel(logging.WARNING)
            print(json.dumps(storage_command(args.command, settings), indent=2, default=str))
        elif args.command == "chunk":
            print(json.dumps(chunk_documents(settings), indent=2))
        else:
            report = download_documents(settings, refresh=args.refresh)
            print(report.model_dump_json(indent=2))
    except (DownloadError, OSError, ValueError) as exc:
        label = {"chunk": "Chunking", "download": "Download"}.get(
            args.command, args.command.title()
        )
        if args.command in {"migrate", "ingest", "stats"}:
            # Driver messages and configuration values must not disclose credentials.
            parser.exit(1, f"{label} failed ({type(exc).__name__}); check settings and services.\n")
        parser.exit(1, f"{label} failed: {exc}\n")
    except Exception as exc:
        from security_rag.ingestion.pipeline import IngestionError

        if isinstance(exc, IngestionError):
            print(exc.report.model_dump_json(indent=2))
        parser.exit(
            1,
            f"{args.command.title()} failed ({type(exc).__name__}); "
            "check services and rerun the command.\n",
        )


def storage_command(command: str, settings: Settings) -> dict:
    from security_rag.parent_store.postgres_store import PostgresStore

    if command == "ingest":
        from security_rag.ingestion.pipeline import ingest_documents

        return ingest_documents(settings).model_dump()
    parents = PostgresStore.from_settings(settings)
    try:
        if command == "migrate":
            return {"applied": parents.migrate()}
        from qdrant_client import QdrantClient

        from security_rag.vector_store.qdrant_store import QdrantStore

        with closing(QdrantClient(url=str(settings.qdrant_url), timeout=30)) as client:
            return {
                "ingestions": parents.statistics(),
                "qdrant": QdrantStore(client, settings.qdrant_collection).statistics(),
            }
    finally:
        parents.close()


def chunk_documents(settings: Settings) -> dict:
    # Tokenizer/model loading happens only when the command is explicitly run.
    from security_rag.ingestion.chunker import MarkdownChunker
    from security_rag.ingestion.loader import load_documents
    from security_rag.ingestion.tokenizer import ChunkTokenizer

    documents = load_documents(settings.data_dir)
    tokenizer = ChunkTokenizer.load(
        settings.embedding_model,
        path=settings.tokenizer_path,
        revision=settings.tokenizer_revision,
    )
    chunker = MarkdownChunker(tokenizer)
    reports = []
    for document in documents:
        chunks = chunker.chunk(document)
        children_by_parent = {}
        for child in chunks.children:
            children_by_parent.setdefault(child.parent_id, []).append(child)
        parent_chunks = [
            {
                "id": str(parent.id),
                "text": parent.text,
                "children": [
                    {"id": str(child.id), "text": child.text}
                    for child in children_by_parent[parent.id]
                ],
            }
            for parent in chunks.parents
        ]
        reports.append(
            {
                "document_id": document.metadata.document_id,
                "document_hash": document.metadata.document_hash,
                "chunking_config_hash": chunker.config_hash,
                "parents": len(chunks.parents),
                "children": len(chunks.children),
                "max_parent_tokens": max(tokenizer.count(p.context_text) for p in chunks.parents),
                "max_child_tokens": max(tokenizer.count(c.embedding_text) for c in chunks.children),
                "parent_chunks": parent_chunks,
            }
        )
    return {
        "parents": sum(report["parents"] for report in reports),
        "children": sum(report["children"] for report in reports),
        "max_parent_tokens": max(report["max_parent_tokens"] for report in reports),
        "max_child_tokens": max(report["max_child_tokens"] for report in reports),
        "documents": reports,
    }


if __name__ == "__main__":
    main()
