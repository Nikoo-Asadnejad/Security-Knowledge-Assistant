from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from security_rag.ingestion.downloader import DownloadError, download_documents
from security_rag.ingestion.loader import load_documents
from security_rag.ingestion.manifest import DOCUMENT_FILES, RAW_BASE_URL


def test_download_skip_and_refresh(settings, markdown):
    requests = []
    body = "Original content.\n"

    def handle(request):
        requests.append(request)
        name = request.url.path.rsplit("/", 1)[-1]
        return httpx.Response(
            200, content=markdown(name, body), headers={"Content-Type": "text/plain; charset=utf-8"}
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        report = download_documents(settings, client=client)
        assert report.downloaded == 10
        assert report.skipped == 0
        assert {str(request.url) for request in requests} == {
            f"{RAW_BASE_URL}{name}" for name in DOCUMENT_FILES
        }
        assert len(requests) == 10
        assert requests[0].extensions["timeout"]["read"] == settings.download_timeout_seconds
        assert {path.name for path in settings.data_dir.iterdir()} == set(DOCUMENT_FILES)
        for name in DOCUMENT_FILES:
            assert (settings.data_dir / name).read_bytes() == markdown(name)

        report = download_documents(settings, client=client)
        assert report.downloaded == 0
        assert report.skipped == 10
        assert len(requests) == 10

        previous_hash = report.documents[0].document_hash
        body = "Updated content.\n"
        report = download_documents(settings, refresh=True, client=client)
        assert report.downloaded == 10
        assert report.skipped == 0
        assert len(requests) == 20
        assert report.documents[0].document_hash != previous_hash
        assert report.documents == [
            document.metadata for document in load_documents(settings.data_dir)
        ]
        assert not client.is_closed


@pytest.mark.parametrize("failure", ["timeout", "disconnect", 408, 429, 500, 503])
def test_transient_errors_are_retried(settings, markdown, failure):
    settings.download_retries = 2
    calls = 0

    def handle(request):
        nonlocal calls
        calls += 1
        if calls <= 2:
            if failure == "timeout":
                raise httpx.ReadTimeout("Timed out", request=request)
            if failure == "disconnect":
                raise httpx.RemoteProtocolError("Incomplete body", request=request)
            return httpx.Response(failure)
        return httpx.Response(
            200,
            content=markdown(Path(request.url.path).name),
            headers={"Content-Type": "text/plain"},
        )

    with (
        httpx.Client(transport=httpx.MockTransport(handle)) as client,
        patch("security_rag.ingestion.downloader.time.sleep") as sleep,
    ):
        assert download_documents(settings, client=client).downloaded == 10
    assert calls == 12
    assert [call.args[0] for call in sleep.call_args_list] == [0.5, 1.0]


@pytest.mark.parametrize("status", [301, 302, 403, 404])
def test_permanent_errors_and_redirects_are_not_retried(settings, status):
    settings.download_retries = 2
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(status, headers={"Location": "https://other.example/file.md"})

    with httpx.Client(transport=httpx.MockTransport(handle), follow_redirects=True) as client:
        with pytest.raises(DownloadError, match=DOCUMENT_FILES[0]):
            download_documents(settings, client=client)
    assert len(calls) == 1
    assert list(settings.data_dir.iterdir()) == []


@pytest.mark.parametrize(
    ("status", "content", "headers"),
    [
        (200, b"", {"Content-Type": "text/plain"}),
        (200, b" \n", {"Content-Type": "text/plain"}),
        (200, b"\xff", {"Content-Type": "text/plain"}),
        (200, b"<html>error</html>", {"Content-Type": "text/html"}),
        (200, b"not Markdown", {"Content-Type": "text/plain"}),
        (200, b"## LLM01:2026 Title\n", {"Content-Type": "text/plain"}),
        (200, b"## LLM01:2026 Title\nBody", {"Content-Type": "application/json"}),
        (
            200,
            b"## LLM01:2026 Title\nBody",
            {"Content-Length": "200", "Content-Type": "text/plain"},
        ),
        (206, b"## LLM01:2026 Title\nPartial body", {"Content-Type": "text/plain"}),
    ],
)
def test_invalid_download_never_overwrites_existing_file(
    settings, markdown, status, content, headers
):
    settings.data_dir.mkdir()
    path = settings.data_dir / DOCUMENT_FILES[0]
    original = markdown(path.name)
    path.write_bytes(original)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(status, content=content, headers=headers)
    )
    with httpx.Client(transport=transport) as client:
        with pytest.raises(DownloadError):
            download_documents(settings, refresh=True, client=client)
    assert path.read_bytes() == original
    assert list(settings.data_dir.iterdir()) == [path]


def test_exhausted_retries_leave_no_partial_files(settings):
    settings.download_retries = 1
    transport = httpx.MockTransport(lambda request: httpx.Response(503))
    with (
        httpx.Client(transport=transport) as client,
        patch("security_rag.ingestion.downloader.time.sleep") as sleep,
        pytest.raises(DownloadError),
    ):
        download_documents(settings, client=client)
    assert sleep.call_count == 1
    assert list(settings.data_dir.iterdir()) == []


def test_invalid_existing_file_requires_refresh(settings):
    settings.data_dir.mkdir()
    (settings.data_dir / DOCUMENT_FILES[0]).write_bytes(b"")
    transport = httpx.MockTransport(lambda request: pytest.fail("Must not download existing files"))
    with httpx.Client(transport=transport) as client:
        with pytest.raises(DownloadError, match="download --refresh"):
            download_documents(settings, client=client)


def test_failed_replace_retains_old_file_and_removes_temporary_file(settings, markdown):
    settings.data_dir.mkdir()
    path = settings.data_dir / DOCUMENT_FILES[0]
    original = markdown(path.name)
    path.write_bytes(original)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, content=markdown(path.name, "Updated"), headers={"Content-Type": "text/plain"}
        )
    )
    with (
        httpx.Client(transport=transport) as client,
        patch("pathlib.Path.replace", side_effect=OSError("Disk failure")),
        pytest.raises(OSError, match="Disk failure"),
    ):
        download_documents(settings, refresh=True, client=client)
    assert path.read_bytes() == original
    assert list(settings.data_dir.iterdir()) == [path]
