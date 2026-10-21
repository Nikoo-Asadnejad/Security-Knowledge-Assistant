"""Download the explicit manifest independently of application startup."""

import logging
import time
from contextlib import nullcontext
from pathlib import Path
from tempfile import NamedTemporaryFile

import httpx
from pydantic import BaseModel

from security_rag.config import Settings
from security_rag.ingestion.loader import DocumentMetadata, load_document, parse_document
from security_rag.ingestion.manifest import DOCUMENT_FILES, RAW_BASE_URL

logger = logging.getLogger(__name__)


class DownloadError(RuntimeError):
    """A document could not be safely downloaded and saved."""


class DownloadReport(BaseModel):
    downloaded: int
    skipped: int
    documents: list[DocumentMetadata]


def _fetch(client: httpx.Client, path: Path, settings: Settings) -> tuple[bytes, DocumentMetadata]:
    url = f"{RAW_BASE_URL}{path.name}"
    for attempt in range(settings.download_retries + 1):
        try:
            response = client.get(
                url,
                timeout=settings.download_timeout_seconds,
                follow_redirects=False,
                headers={"Accept": "text/plain", "Accept-Encoding": "identity"},
            )
            response.raise_for_status()
            if response.status_code != 200:
                raise ValueError(f"Expected HTTP 200, received {response.status_code}")
            media_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
            if media_type not in {"text/plain", "text/markdown", "text/x-markdown"}:
                raise ValueError(f"Unexpected content type: {media_type or 'missing'}")
            content = response.content
            length = response.headers.get("content-length")
            if (
                length is not None
                and response.headers.get("content-encoding", "identity") == "identity"
            ):
                if int(length) != len(content):
                    raise ValueError("Incomplete response: content-length does not match")
            document = parse_document(path, content)
            return content, document.metadata
        except (httpx.TransportError, httpx.HTTPStatusError, ValueError) as exc:
            retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                exc.response.status_code in {408, 429} or exc.response.status_code >= 500
            )
            if not retryable or attempt == settings.download_retries:
                raise DownloadError(f"Could not download {path.name}: {exc}") from exc
            logger.warning(
                "document_download_retry",
                extra={"file_name": path.name, "attempt": attempt + 1, "error": str(exc)},
            )
            time.sleep(0.5 * 2**attempt)
    raise AssertionError("Retry loop must return or raise")


def _save(path: Path, content: bytes) -> None:
    """Replace only after a complete, validated download; retain old files on failure."""
    temporary_path = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as file:
            temporary_path = Path(file.name)
            file.write(content)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def download_documents(
    settings: Settings,
    *,
    refresh: bool = False,
    client: httpx.Client | None = None,
) -> DownloadReport:
    """Download only manifest files; an injected client remains owned by its caller."""
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    documents = []
    with nullcontext(client) if client is not None else httpx.Client() as http:
        for name in DOCUMENT_FILES:
            path = settings.data_dir / name
            if path.exists() and not refresh:
                try:
                    document = load_document(path)
                except ValueError as exc:
                    raise DownloadError(
                        f"Invalid existing file {name}; run download --refresh"
                    ) from exc
                documents.append(document.metadata)
                logger.info("document_download_skipped", extra={"file_name": name})
                continue
            content, metadata = _fetch(http, path, settings)
            _save(path, content)
            documents.append(metadata)
            downloaded += 1
            logger.info(
                "document_downloaded",
                extra={"file_name": name, "document_hash": metadata.document_hash},
            )
    return DownloadReport(
        downloaded=downloaded, skipped=len(documents) - downloaded, documents=documents
    )
