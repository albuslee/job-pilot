from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from jobpilot.cli import app
from jobpilot.models.schemas import AnswerFeedback, CapturedAnswer

runner = CliRunner()


def _fake_coach() -> MagicMock:
    coach = MagicMock()
    coach.run.return_value = AnswerFeedback(
        answered_question=True, structure_notes="ok", strengths=["s"], improvements=["i"]
    )
    return coach


def test_interview_drill_text_mode_writes_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    with (
        patch("jobpilot.cli._build_store", return_value=MagicMock()),
        patch("jobpilot.cli.LLMClient", return_value=MagicMock()),
        patch("jobpilot.cli.InterviewCoachAgent", return_value=_fake_coach()),
        patch("jobpilot.cli.TypedAnswerSource") as typed_src,
    ):
        typed_src.return_value.capture.return_value = CapturedAnswer(
            transcript="my answer", source="text"
        )
        result = runner.invoke(app, ["interview", "--mode", "drill", "--num", "1", "--text"])

    assert result.exit_code == 0, result.output
    reports = list(tmp_path.glob("interview_*.md"))
    assert len(reports) == 1
    assert "my answer" in reports[0].read_text(encoding="utf-8")


def test_interview_no_save_skips_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    with (
        patch("jobpilot.cli._build_store", return_value=MagicMock()),
        patch("jobpilot.cli.LLMClient", return_value=MagicMock()),
        patch("jobpilot.cli.InterviewCoachAgent", return_value=_fake_coach()),
        patch("jobpilot.cli.TypedAnswerSource") as typed_src,
    ):
        typed_src.return_value.capture.return_value = CapturedAnswer(
            transcript="x", source="text"
        )
        result = runner.invoke(
            app, ["interview", "--mode", "drill", "--num", "1", "--text", "--no-save"]
        )

    assert result.exit_code == 0, result.output
    assert list(tmp_path.glob("interview_*.md")) == []
