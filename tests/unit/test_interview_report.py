from __future__ import annotations

from pathlib import Path

from jobpilot.interview.report import default_report_path, write_interview_report
from jobpilot.models.schemas import (
    AnswerFeedback,
    DeliveryMetrics,
    InterviewQuestion,
    InterviewTurn,
    SessionSummary,
)


def _turn() -> InterviewTurn:
    q = InterviewQuestion(id="why_leave", category="motivational", text="Why leave?")
    m = DeliveryMetrics(
        word_count=40, duration_s=20.0, wpm=120.0, filler_count=3, top_fillers=[],
        long_pause_count=1, longest_pause_s=2.0, mean_pitch_hz=130.0,
        pitch_range_hz=60.0, pitch_std_semitones=2.4, monotone=False,
    )
    fb = AnswerFeedback(
        answered_question=True, structure_notes="Good arc.",
        strengths=["Concrete"], improvements=["End with impact"],
        unsupported_claims=["Claimed FAANG scale"], missed_experiences=["GCP migration"],
        cited_chunk_ids=["cv-1"],
    )
    return InterviewTurn(question=q, transcript="I led a RAG build.", metrics=m, feedback=fb)


def test_write_report_contains_turn_and_metrics(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    write_interview_report([_turn()], out, mode="drill")
    text = out.read_text(encoding="utf-8")
    assert "Why leave?" in text
    assert "I led a RAG build." in text
    assert "120.0" in text  # wpm
    assert "2.4 semitones" in text
    assert "GCP migration" in text  # missed experience
    assert "cv-1" in text  # citation


def test_mock_report_appends_summary(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    summary = SessionSummary(
        overall="Solid session.", top_strengths=["Concrete"], top_improvements=["Slow down"]
    )
    write_interview_report([_turn()], out, mode="mock", summary=summary)
    text = out.read_text(encoding="utf-8")
    assert "Session Summary" in text
    assert "Solid session." in text
    assert "Slow down" in text


def test_default_report_path_uses_output_dir(tmp_path: Path) -> None:
    p = default_report_path(tmp_path)
    assert p.parent == tmp_path
    assert p.name.startswith("interview_")
    assert p.suffix == ".md"
