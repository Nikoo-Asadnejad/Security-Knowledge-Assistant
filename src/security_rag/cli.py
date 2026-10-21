"""Command-line entry point for the project."""

import argparse
from importlib.metadata import version

from pydantic import ValidationError

from security_rag.config import Settings
from security_rag.ingestion.downloader import DownloadError, download_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Security Knowledge Assistant")
    parser.add_argument("--version", action="version", version=version("security-rag"))
    commands = parser.add_subparsers(dest="command")
    download = commands.add_parser("download", help="Download the ten OWASP Markdown documents")
    download.add_argument("--refresh", action="store_true", help="Replace existing documents")
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    try:
        report = download_documents(Settings(), refresh=args.refresh)
    except (DownloadError, OSError, ValidationError) as exc:
        parser.exit(1, f"Download failed: {exc}\n")
    print(report.model_dump_json(indent=2))
