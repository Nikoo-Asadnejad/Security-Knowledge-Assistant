import json
import sys
from unittest.mock import patch

import pytest

from security_rag.cli import main
from security_rag.ingestion.downloader import DownloadError, DownloadReport


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--help"],
        ["download", "--help"],
        ["chunk", "--help"],
        ["migrate", "--help"],
        ["ingest", "--help"],
        ["stats", "--help"],
    ],
)
def test_help_needs_no_settings_or_network(settings, monkeypatch, capsys, arguments):
    monkeypatch.setenv("QDRANT_URL", "invalid")
    monkeypatch.setattr(sys, "argv", ["security-rag", *arguments])
    if arguments:
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
    else:
        main()
    command = arguments[0] if arguments and arguments[0] != "--help" else "download"
    assert command in capsys.readouterr().out


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


def test_chunk_command_prints_statistics_and_parent_child_texts_without_writing_files(
    settings, monkeypatch, capsys, markdown, tokenizer, tmp_path
):
    from security_rag.ingestion.manifest import DOCUMENT_FILES

    settings.data_dir.mkdir()
    body = (
        "Original content.\n\n#### Details\nChild details.\n\n### Mitigation\nMitigation content.\n"
    )
    for name in DOCUMENT_FILES:
        (settings.data_dir / name).write_bytes(markdown(name, body))
    monkeypatch.setenv("DATA_DIR", str(settings.data_dir))
    monkeypatch.setattr(sys, "argv", ["security-rag", "chunk"])
    existing_files = set(tmp_path.rglob("*"))
    with patch("security_rag.ingestion.tokenizer.ChunkTokenizer.load", return_value=tokenizer):
        main()
    report = json.loads(capsys.readouterr().out)
    assert len(report["documents"]) == 10
    assert report["parents"] == 20
    assert report["children"] == 30
    assert report["max_parent_tokens"] <= 1200
    assert report["max_child_tokens"] <= 450
    assert all("parent_file" not in document for document in report["documents"])
    for document in report["documents"]:
        description, mitigation = document["parent_chunks"]
        assert description["id"] != mitigation["id"]
        assert "Original content." in description["text"]
        assert "Child details." in description["text"]
        intro, details = description["children"]
        assert intro["id"] != details["id"]
        assert "Original content." in intro["text"]
        assert "Child details." in details["text"]
        assert intro["text"] in description["text"]
        assert details["text"] in description["text"]
        assert "Mitigation content." in mitigation["text"]
        assert len(mitigation["children"]) == 1
        assert mitigation["children"][0]["text"] == mitigation["text"]
    assert set(tmp_path.rglob("*")) == existing_files


def test_missing_documents_fail_before_tokenizer_download(settings, monkeypatch, capsys):
    monkeypatch.setenv("DATA_DIR", str(settings.data_dir))
    monkeypatch.setattr(sys, "argv", ["security-rag", "chunk"])
    with patch("security_rag.ingestion.tokenizer.ChunkTokenizer.load") as load:
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1
    assert "Chunking failed:" in capsys.readouterr().err
    load.assert_not_called()


def test_invalid_tokenizer_exits_without_traceback(
    settings, monkeypatch, capsys, markdown, tmp_path
):
    from security_rag.ingestion.manifest import DOCUMENT_FILES

    settings.data_dir.mkdir()
    for name in DOCUMENT_FILES:
        (settings.data_dir / name).write_bytes(markdown(name))
    path = tmp_path / "tokenizer.json"
    path.write_text("invalid JSON")
    monkeypatch.setenv("DATA_DIR", str(settings.data_dir))
    monkeypatch.setenv("TOKENIZER_PATH", str(path))
    monkeypatch.setattr(sys, "argv", ["security-rag", "chunk"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert "Chunking failed: Could not load tokenizer" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["migrate", "ingest", "stats"])
def test_storage_commands_print_json(settings, monkeypatch, capsys, command):
    monkeypatch.setattr(sys, "argv", ["security-rag", command])
    with patch("security_rag.cli.storage_command", return_value={"ok": True}) as run:
        main()
    assert run.call_args.args[0] == command
    assert json.loads(capsys.readouterr().out) == {"ok": True}


def test_ingestion_failure_prints_report_and_nonzero_status(settings, monkeypatch, capsys):
    from security_rag.ingestion.pipeline import IngestionError, IngestionReport

    monkeypatch.setattr(sys, "argv", ["security-rag", "ingest"])
    report = IngestionReport(failed=1, failures=[{"file_name": "bad.md", "error": "ValueError"}])
    with patch("security_rag.cli.storage_command", side_effect=IngestionError(report)):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1
    output = capsys.readouterr()
    assert json.loads(output.out)["failed"] == 1
    assert "IngestionError" in output.err


def test_database_error_does_not_disclose_credentials(settings, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["security-rag", "migrate"])
    with patch("security_rag.cli.storage_command", side_effect=RuntimeError("secret-password")):
        with pytest.raises(SystemExit):
            main()
    assert "secret-password" not in capsys.readouterr().err
