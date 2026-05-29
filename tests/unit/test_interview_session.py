from __future__ import annotations

from unittest.mock import MagicMock

from jobpilot.config import Settings
from jobpilot.interview.session import (
    TypedAnswerSource,
    build_session_summary,
    run_drill,
    run_mock,
)
from jobpilot.models.schemas import (
    AnswerFeedback,
    CapturedAnswer,
    DeliveryMetrics,
    FollowUp,
    InterviewQuestion,
    InterviewTurn,
)


class _FakeSource:
    """Returns a canned CapturedAnswer per capture() call, recording prompts seen."""

    def __init__(self, transcripts: list[str]) -> None:
        self._transcripts = list(transcripts)
        self.prompts: list[str] = []

    def capture(self, prompt_text: str) -> CapturedAnswer:
        self.prompts.append(prompt_text)
        text = self._transcripts.pop(0) if self._transcripts else "..."
        return CapturedAnswer(transcript=text, source="text")


def _coach() -> MagicMock:
    coach = MagicMock()
    coach.run.return_value = AnswerFeedback(
        answered_question=True, structure_notes="ok", strengths=["s1"], improvements=["i1"]
    )
    return coach


def _questions(n: int = 2) -> list[InterviewQuestion]:
    return [
        InterviewQuestion(id=f"q{i}", category="fit", text=f"Question {i}?")
        for i in range(n)
    ]


def test_run_drill_one_turn_per_question(settings: Settings) -> None:
    source = _FakeSource(["answer 0", "answer 1"])
    coach = _coach()
    turns = run_drill(_questions(2), source=source, coach=coach, settings=settings)
    assert len(turns) == 2
    assert all(t.feedback is not None for t in turns)
    assert turns[0].transcript == "answer 0"
    assert coach.run.call_count == 2
    assert turns[0].metrics.word_count == 2


def test_run_mock_adds_followups_then_stops(settings: Settings) -> None:
    settings = settings.model_copy(update={"interview_followups": 2})
    interviewer = MagicMock()
    interviewer.next_followup.side_effect = [
        FollowUp(should_continue=True, question="Example?"),
        FollowUp(should_continue=False, question=None),
    ]
    source = _FakeSource(["seed answer", "followup answer"])
    coach = _coach()
    turns, summary = run_mock(
        _questions(1), source=source, coach=coach, interviewer=interviewer, settings=settings
    )
    assert len(turns) == 2  # seed + 1 followup
    assert summary.overall  # non-empty
    assert turns[1].question.text == "Example?"


def test_run_mock_respects_followup_cap(settings: Settings) -> None:
    settings = settings.model_copy(update={"interview_followups": 1})
    interviewer = MagicMock()
    interviewer.next_followup.return_value = FollowUp(should_continue=True, question="More?")
    source = _FakeSource(["a", "b", "c", "d"])
    coach = _coach()
    turns, _ = run_mock(
        _questions(1), source=source, coach=coach, interviewer=interviewer, settings=settings
    )
    assert len(turns) == 2  # seed + exactly 1 followup (cap), despite always-continue


def _turn_with(strengths: list[str], improvements: list[str]) -> InterviewTurn:
    q = InterviewQuestion(id="q", category="fit", text="?")
    m = DeliveryMetrics(
        word_count=1, duration_s=None, wpm=None, filler_count=0, top_fillers=[],
        long_pause_count=0, longest_pause_s=None, mean_pitch_hz=None,
        pitch_range_hz=None, pitch_std_semitones=None, monotone=None,
    )
    fb = AnswerFeedback(
        answered_question=True, structure_notes="ok",
        strengths=strengths, improvements=improvements,
    )
    return InterviewTurn(question=q, transcript="x", metrics=m, feedback=fb)


def test_build_session_summary_aggregates_and_dedupes() -> None:
    turns = [
        _turn_with(["clear", "concise"], ["add metrics"]),
        _turn_with(["clear"], ["add metrics", "slow down"]),
    ]
    summary = build_session_summary(turns)
    assert "clear" in summary.top_strengths
    assert summary.top_strengths.count("clear") == 1  # deduped
    assert "add metrics" in summary.top_improvements
    assert summary.overall


def test_typed_answer_source_reads_input() -> None:
    src = TypedAnswerSource(input_fn=lambda _: "typed answer here")
    ca = src.capture("Q?")
    assert ca.transcript == "typed answer here"
    assert ca.source == "text"
    assert ca.pitch_hz == []
