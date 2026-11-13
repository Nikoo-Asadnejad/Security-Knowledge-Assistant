"""Command-line entry point for the project."""

import argparse
import json
import logging
from importlib.metadata import version

from security_rag.bootstrap import (
    chunk_documents,
    download_documents,
    ingest_documents,
    migrate_database,
    storage_statistics,
)
from security_rag.config import Settings
from security_rag.core.application.ingest import IngestionBusy, IngestionError
from security_rag.infrastructure.download import DownloadError


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
            if args.command == "ingest":
                result = ingest_documents(settings).model_dump()
            elif args.command == "migrate":
                result = migrate_database(settings)
            else:
                result = storage_statistics(settings)
            print(json.dumps(result, indent=2, default=str))
        elif args.command == "chunk":
            print(json.dumps(chunk_documents(settings), indent=2))
        else:
            report = download_documents(settings, refresh=args.refresh)
            print(report.model_dump_json(indent=2))
    except IngestionBusy as exc:
        parser.exit(1, f"Ingest failed: {exc}\n")
    except (DownloadError, OSError, ValueError) as exc:
        label = {"chunk": "Chunking", "download": "Download"}.get(
            args.command, args.command.title()
        )
        if args.command in {"migrate", "ingest", "stats"}:
            # Driver messages and configuration values must not disclose credentials.
            parser.exit(1, f"{label} failed ({type(exc).__name__}); check settings and services.\n")
        parser.exit(1, f"{label} failed: {exc}\n")
    except Exception as exc:
        if isinstance(exc, IngestionError):
            print(exc.report.model_dump_json(indent=2))
        parser.exit(
            1,
            f"{args.command.title()} failed ({type(exc).__name__}); "
            "check services and rerun the command.\n",
        )


if __name__ == "__main__":
    main()
