from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from typer.testing import CliRunner

from jobpilot.cli import app


def test_ingest_does_not_build_llm_by_default(
    monkeypatch,
    tmp_path: Path,
) -> None:
    profile = tmp_path / "profile"
    profile.mkdir()
    monkeypatch.setenv("PROFILE_DIR", str(profile))
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr("jobpilot.cli._build_store", lambda: object())
    ingest = MagicMock(return_value=0)
    monkeypatch.setattr("jobpilot.cli.ingest_profile_dir", ingest)
    build_llm_mock = MagicMock()
    monkeypatch.setattr("jobpilot.cli.build_llm", build_llm_mock)

    result = CliRunner().invoke(app, ["ingest"])

    assert result.exit_code == 0, result.output
    build_llm_mock.assert_not_called()
    assert ingest.call_args.kwargs["smart_docx"] is False
    assert ingest.call_args.kwargs["llm"] is None


def test_ingest_smart_docx_builds_llm_and_passes_flag(
    monkeypatch,
    tmp_path: Path,
) -> None:
    profile = tmp_path / "profile"
    profile.mkdir()
    monkeypatch.setenv("PROFILE_DIR", str(profile))
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr("jobpilot.cli._build_store", lambda: object())
    ingest = MagicMock(return_value=0)
    monkeypatch.setattr("jobpilot.cli.ingest_profile_dir", ingest)
    llm = object()
    build_llm_mock = MagicMock(return_value=llm)
    monkeypatch.setattr("jobpilot.cli.build_llm", build_llm_mock)

    result = CliRunner().invoke(app, ["ingest", "--smart-docx"])

    assert result.exit_code == 0, result.output
    build_llm_mock.assert_called_once()
    assert ingest.call_args.kwargs["smart_docx"] is True
    assert ingest.call_args.kwargs["llm"] is llm
