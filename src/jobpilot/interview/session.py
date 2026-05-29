"""Session loops for drill and mock modes, plus the AnswerSource seam.

AnswerSource decouples "get the candidate's answer" from how it's captured, so
the loops are testable with a fake source (no mic, no Whisper). RecordingAnswerSource
is the real mic path; TypedAnswerSource is the --text fallback.
"""

from __future__ import annotations

from collections import Counter
from typing import Protocol

from jobpilot.agents.interview_coach import InterviewCoachAgent
from jobpilot.agents.interviewer import InterviewerAgent
from jobpilot.config import Settings
from jobpilot.interview.delivery import compute_delivery_metrics
from jobpilot.logging_setup import get_logger
from jobpilot.models.schemas import (
    CapturedAnswer,
    DeliveryMetrics,
    InterviewQuestion,
    InterviewTurn,
    SessionSummary,
)

log = get_logger(__name__)

_MAX_SUMMARY_ITEMS = 5


class AnswerSource(Protocol):
    def capture(self, prompt_text: str) -> CapturedAnswer: ...


class TypedAnswerSource:
    """--text fallback: read a typed answer from stdin. No audio metrics."""

    def __init__(self, input_fn=input) -> None:
        self._input = input_fn

    def capture(self, prompt_text: str) -> CapturedAnswer:
        text = self._input("Your answer (type, then Enter): ")
        return CapturedAnswer(transcript=text.strip(), source="text")


class RecordingAnswerSource:
    """Real mic path: record -> transcribe -> pitch. Imports adapters lazily."""

    def __init__(self, *, settings: Settings) -> None:
        self._settings = settings

    def capture(self, prompt_text: str) -> CapturedAnswer:
        from jobpilot.tools.audio import extract_pitch, record_audio
        from jobpilot.tools.transcribe import transcribe

        wav_path, duration_s = record_audio()
        transcript = transcribe(wav_path, settings=self._settings)
        pitch_hz = extract_pitch(wav_path)
        return CapturedAnswer(
            transcript=transcript.text,
            words=transcript.words,
            duration_s=transcript.duration_s or duration_s,
            pitch_hz=pitch_hz,
            source="audio",
        )


def _metrics_for(captured: CapturedAnswer, settings: Settings) -> DeliveryMetrics:
    return compute_delivery_metrics(
        captured.transcript,
        captured.words,
        captured.duration_s,
        captured.pitch_hz,
        pause_threshold_s=settings.pause_threshold_s,
        monotone_std_threshold_semitones=settings.monotone_std_threshold_semitones,
    )


def _coach_turn(
    question: InterviewQuestion,
    source: AnswerSource,
    coach: InterviewCoachAgent,
    settings: Settings,
) -> InterviewTurn:
    print(f"\n[{question.category}] {question.text}")
    captured = source.capture(question.text)
    metrics = _metrics_for(captured, settings)
    feedback = coach.run(question, captured, metrics)
    return InterviewTurn(
        question=question, transcript=captured.transcript, metrics=metrics, feedback=feedback
    )


def run_drill(
    questions: list[InterviewQuestion],
    *,
    source: AnswerSource,
    coach: InterviewCoachAgent,
    settings: Settings,
) -> list[InterviewTurn]:
    turns: list[InterviewTurn] = []
    for q in questions:
        turns.append(_coach_turn(q, source, coach, settings))
    return turns


def run_mock(
    seed_questions: list[InterviewQuestion],
    *,
    source: AnswerSource,
    coach: InterviewCoachAgent,
    interviewer: InterviewerAgent,
    settings: Settings,
) -> tuple[list[InterviewTurn], SessionSummary]:
    turns: list[InterviewTurn] = []
    for seed in seed_questions:
        turns.append(_coach_turn(seed, source, coach, settings))
        for i in range(settings.interview_followups):
            fu = interviewer.next_followup(seed, turns)
            if not fu.should_continue or not fu.question:
                break
            followup_q = InterviewQuestion(
                id=f"{seed.id}-followup-{i + 1}",
                category=seed.category,
                text=fu.question,
            )
            turns.append(_coach_turn(followup_q, source, coach, settings))
    return turns, build_session_summary(turns)


def _top_items(counts: Counter[str]) -> list[str]:
    """Most frequent first, ties broken alphabetically for deterministic output."""
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [item for item, _ in ordered[:_MAX_SUMMARY_ITEMS]]


def build_session_summary(turns: list[InterviewTurn]) -> SessionSummary:
    """Aggregate per-turn grounded feedback into a session-level summary.

    Deterministic: dedupe strengths/improvements, keep the most frequent first,
    cap the list. `overall` is a short factual recap.
    """
    strengths: Counter[str] = Counter()
    improvements: Counter[str] = Counter()
    for t in turns:
        if t.feedback is None:
            continue
        strengths.update(t.feedback.strengths)
        improvements.update(t.feedback.improvements)

    top_strengths = _top_items(strengths)
    top_improvements = _top_items(improvements)
    answered = sum(
        1 for t in turns if t.feedback is not None and t.feedback.answered_question
    )
    overall = (
        f"Answered {answered}/{len(turns)} questions on-topic. "
        f"{len(top_strengths)} notable strength(s), "
        f"{len(top_improvements)} area(s) to improve."
    )
    return SessionSummary(
        overall=overall, top_strengths=top_strengths, top_improvements=top_improvements
    )
