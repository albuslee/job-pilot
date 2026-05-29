from __future__ import annotations

from unittest.mock import MagicMock

from jobpilot.agents.interviewer import InterviewerAgent
from jobpilot.config import Settings
from jobpilot.models.schemas import (
    DeliveryMetrics,
    FollowUp,
    InterviewQuestion,
    InterviewTurn,
)


def _turn(text: str) -> InterviewTurn:
    q = InterviewQuestion(id="seed", category="fit", text="Why us?")
    m = DeliveryMetrics(
        word_count=10, duration_s=5.0, wpm=120.0, filler_count=0, top_fillers=[],
        long_pause_count=0, longest_pause_s=None, mean_pitch_hz=None,
        pitch_range_hz=None, pitch_std_semitones=None, monotone=None,
    )
    return InterviewTurn(question=q, transcript=text, metrics=m, feedback=None)


def test_interviewer_returns_followup(settings: Settings) -> None:
    llm = MagicMock()
    llm.complete_structured.return_value = FollowUp(
        should_continue=True, question="Can you give a specific example?"
    )
    agent = InterviewerAgent(settings=settings, llm=llm)
    q = InterviewQuestion(id="seed", category="fit", text="Why us?")

    fu = agent.next_followup(q, [_turn("I like the mission.")])

    assert fu.should_continue is True
    assert fu.question == "Can you give a specific example?"
    kwargs = llm.complete_structured.call_args.kwargs
    assert kwargs["schema"] is FollowUp
    assert kwargs["tool_name"] == "submit_followup"
    assert "I like the mission." in kwargs["user"]


def test_interviewer_can_stop(settings: Settings) -> None:
    llm = MagicMock()
    llm.complete_structured.return_value = FollowUp(should_continue=False, question=None)
    agent = InterviewerAgent(settings=settings, llm=llm)
    q = InterviewQuestion(id="seed", category="fit", text="Why us?")
    fu = agent.next_followup(q, [_turn("done")])
    assert fu.should_continue is False
