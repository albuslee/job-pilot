from __future__ import annotations

from unittest.mock import MagicMock

from jobpilot.agents.interview_coach import InterviewCoachAgent
from jobpilot.config import Settings
from jobpilot.models.schemas import (
    AnswerFeedback,
    CapturedAnswer,
    DeliveryMetrics,
    InterviewQuestion,
    ProfileChunk,
)


def _metrics() -> DeliveryMetrics:
    return DeliveryMetrics(
        word_count=40, duration_s=20.0, wpm=120.0, filler_count=3,
        top_fillers=[], long_pause_count=1, longest_pause_s=2.0,
        mean_pitch_hz=130.0, pitch_range_hz=60.0, pitch_std_semitones=2.4, monotone=False,
    )


def test_coach_retrieves_and_returns_feedback(settings: Settings) -> None:
    chunks = [ProfileChunk(id="cv-1", source="cv.docx", heading_path=["Exp"], text="Led RAG build.")]
    rag = MagicMock()
    rag.query.return_value = chunks
    llm = MagicMock()
    llm.complete_structured.return_value = AnswerFeedback(
        answered_question=True,
        structure_notes="Strong open, weak close.",
        strengths=["Concrete example"],
        improvements=["End with impact"],
        unsupported_claims=[],
        missed_experiences=["GCP migration"],
        cited_chunk_ids=["cv-1"],
    )

    agent = InterviewCoachAgent(settings=settings, llm=llm, rag=rag)
    q = InterviewQuestion(id="why_leave", category="motivational", text="Why leave?")
    captured = CapturedAnswer(transcript="I led a RAG build.", source="audio")

    fb = agent.run(q, captured, _metrics())

    assert fb.answered_question is True
    assert fb.cited_chunk_ids == ["cv-1"]
    rag.query.assert_called_once_with("I led a RAG build.", k=settings.retrieval_k)
    kwargs = llm.complete_structured.call_args.kwargs
    assert kwargs["schema"] is AnswerFeedback
    assert kwargs["tool_name"] == "submit_answer_feedback"
    assert "120.0" in kwargs["user"] or "120.0" in kwargs["system"]
