from __future__ import annotations

import pytest
from pydantic import ValidationError

from jobpilot.models.schemas import (
    AnswerFeedback,
    CapturedAnswer,
    DeliveryMetrics,
    FollowUp,
    InterviewQuestion,
    InterviewTurn,
    SessionSummary,
    Transcript,
    Word,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metrics() -> DeliveryMetrics:
    return DeliveryMetrics(
        word_count=3,
        duration_s=2.0,
        wpm=90.0,
        filler_count=0,
        top_fillers=[],
        long_pause_count=0,
        longest_pause_s=None,
        mean_pitch_hz=120.0,
        pitch_range_hz=40.0,
        pitch_std_semitones=2.1,
        monotone=False,
    )


def _make_feedback() -> AnswerFeedback:
    return AnswerFeedback(answered_question=True, structure_notes="Clear.")


# ---------------------------------------------------------------------------
# InterviewQuestion
# ---------------------------------------------------------------------------


def test_interview_question_requires_text() -> None:
    q = InterviewQuestion(id="why_leave", category="motivational", text="Why leave?")
    assert q.guidance is None
    with pytest.raises(ValidationError):
        InterviewQuestion(id="x", category="c", text="")


def test_interview_question_rejects_empty_id_and_category() -> None:
    with pytest.raises(ValidationError):
        InterviewQuestion(id="", category="c", text="ok")
    with pytest.raises(ValidationError):
        InterviewQuestion(id="x", category="", text="ok")


# ---------------------------------------------------------------------------
# CapturedAnswer
# ---------------------------------------------------------------------------


def test_captured_answer_defaults() -> None:
    ca = CapturedAnswer(transcript="hello world", source="text")
    assert ca.words == []
    assert ca.pitch_hz == []
    assert ca.duration_s is None


def test_captured_answer_rejects_unknown_source() -> None:
    with pytest.raises(ValidationError):
        CapturedAnswer(transcript="hi", source="keyboard")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------


def test_transcript_defaults() -> None:
    t = Transcript(text="hi")
    assert t.words == []
    assert t.duration_s is None


# ---------------------------------------------------------------------------
# DeliveryMetrics
# ---------------------------------------------------------------------------


def test_delivery_metrics_allows_none_pitch() -> None:
    m = DeliveryMetrics(
        word_count=2,
        duration_s=None,
        wpm=None,
        filler_count=0,
        top_fillers=[],
        long_pause_count=0,
        longest_pause_s=None,
        mean_pitch_hz=None,
        pitch_range_hz=None,
        pitch_std_semitones=None,
        monotone=None,
    )
    assert m.monotone is None


# ---------------------------------------------------------------------------
# InterviewTurn
# ---------------------------------------------------------------------------


def test_interview_turn_constructs() -> None:
    q = InterviewQuestion(id="q1", category="fit", text="Tell me about yourself.")
    turn = InterviewTurn(
        question=q,
        transcript="a b c",
        metrics=_make_metrics(),
        feedback=_make_feedback(),
    )
    assert turn.feedback is not None


# ---------------------------------------------------------------------------
# SessionSummary
# ---------------------------------------------------------------------------


def test_session_summary_defaults() -> None:
    assert SessionSummary(overall="Good.").top_strengths == []


# ---------------------------------------------------------------------------
# FollowUp
# ---------------------------------------------------------------------------


def test_followup_optional_question() -> None:
    assert FollowUp(should_continue=False).question is None


# ---------------------------------------------------------------------------
# Word
# ---------------------------------------------------------------------------


def test_word_fields() -> None:
    assert Word(text="hi", start=0.0, end=0.3).end == 0.3
