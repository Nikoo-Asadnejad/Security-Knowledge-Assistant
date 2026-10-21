import json
import sys
from unittest.mock import patch

import pytest

from security_rag.cli import main
from security_rag.ingestion.downloader import DownloadError, DownloadReport


@pytest.mark.parametrize("arguments", [[], ["--help"], ["download", "--help"]])
def test_help_needs_no_settings_or_network(settings, monkeypatch, capsys, arguments):
    monkeypatch.setenv("QDRANT_URL", "invalid")
    monkeypatch.setattr(sys, "argv", ["security-rag", *arguments])
    if arguments:
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
    else:
        main()
    assert "download" in capsys.readouterr().out


@pytest.mark.parametrize("refresh", [False, True])
def test_download_outputs_metadata_report(settings, monkeypatch, capsys, refresh):
    monkeypatch.setattr(
        sys, "argv", ["security-rag", "download", *(["--refresh"] if refresh else [])]
    )
    report = DownloadReport(downloaded=0, skipped=10, documents=[])
    with patch("security_rag.cli.download_documents", return_value=report) as download:
        main()
    assert download.call_args.kwargs == {"refresh": refresh}
    assert json.loads(capsys.readouterr().out) == report.model_dump()


def test_download_failure_exits_without_traceback(settings, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["security-rag", "download"])
    with patch(
        "security_rag.cli.download_documents", side_effect=DownloadError("Source unavailable")
    ):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "Download failed: Source unavailable" in output.err


def test_invalid_configuration_exits_without_traceback(settings, monkeypatch, capsys):
    monkeypatch.setenv("DOWNLOAD_RETRIES", "-1")
    monkeypatch.setattr(sys, "argv", ["security-rag", "download"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert "Download failed:" in capsys.readouterr().err
