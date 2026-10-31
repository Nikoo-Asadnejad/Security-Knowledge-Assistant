"""Command-line entry point for the project."""

import argparse
import json
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
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    try:
        settings = Settings()
        if args.command == "chunk":
            print(json.dumps(chunk_documents(settings), indent=2))
        else:
            report = download_documents(settings, refresh=args.refresh)
            print(report.model_dump_json(indent=2))
    except (DownloadError, OSError, ValueError) as exc:
        label = "Chunking" if args.command == "chunk" else "Download"
        parser.exit(1, f"{label} failed: {exc}\n")


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
