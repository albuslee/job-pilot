"""Interview coach agent: RAG over the CV → structured, grounded AnswerFeedback.

Sync (not async): unlike evaluator/tailor it does not run inside a LangGraph.
"""

from __future__ import annotations

from pathlib import Path

from jobpilot.config import Settings
from jobpilot.llm.client import LLMClient
from jobpilot.logging_setup import get_logger
from jobpilot.models.schemas import (
    AnswerFeedback,
    CapturedAnswer,
    DeliveryMetrics,
    InterviewQuestion,
    ProfileChunk,
)
from jobpilot.prompts.loader import PromptLoader
from jobpilot.rag.store import RagStore

log = get_logger(__name__)

_TOOL_NAME = "submit_answer_feedback"
_TOOL_DESCRIPTION = (
    "Submit grounded feedback on the candidate's interview answer. "
    "cited_chunk_ids MUST be a subset of the profile chunk IDs you were shown; "
    "do not invent claims, experiences, or delivery numbers."
)


def _format_chunks_for_prompt(chunks: list[ProfileChunk]) -> str:
    parts: list[str] = []
    for c in chunks:
        heading = " > ".join(c.heading_path) or "(no heading)"
        parts.append(f"[{c.id}] {heading}\n{c.text}")
    return "\n\n".join(parts)


def _format_metrics_for_prompt(m: DeliveryMetrics) -> str:
    wpm = "n/a" if m.wpm is None else f"{m.wpm} wpm"
    pitch = (
        "n/a"
        if m.pitch_std_semitones is None
        else f"mean {m.mean_pitch_hz} Hz, range {m.pitch_range_hz} Hz, "
        f"variation {m.pitch_std_semitones} semitones "
        f"({'monotone' if m.monotone else 'varied'})"
    )
    return (
        f"- pace: {wpm} ({m.word_count} words)\n"
        f"- fillers: {m.filler_count} total\n"
        f"- long pauses: {m.long_pause_count}"
        f"{'' if m.longest_pause_s is None else f' (longest {m.longest_pause_s}s)'}\n"
        f"- pitch: {pitch}"
    )


class InterviewCoachAgent:
    def __init__(
        self,
        *,
        settings: Settings,
        llm: LLMClient,
        rag: RagStore,
        prompts: PromptLoader | None = None,
        prompt_version: str = "v1",
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._rag = rag
        self._prompts = prompts or PromptLoader(Path(__file__).parent.parent / "prompts")
        self._version = prompt_version

    def run(
        self,
        question: InterviewQuestion,
        captured: CapturedAnswer,
        metrics: DeliveryMetrics,
    ) -> AnswerFeedback:
        chunks = self._rag.query(captured.transcript, k=self._settings.retrieval_k)
        log.info("coach.retrieved", count=len(chunks))
        formatted_chunks = _format_chunks_for_prompt(chunks)

        prompt = self._prompts.render(
            "interview_coach",
            self._version,
            variables={
                "category": question.category,
                "question": question.text,
                "guidance": question.guidance or "(none provided)",
                "transcript": captured.transcript,
                "metrics": _format_metrics_for_prompt(metrics),
                "k": self._settings.retrieval_k,
                "profile_chunks": formatted_chunks,
            },
        )

        feedback = self._llm.complete_structured(
            system=prompt.system,
            user=prompt.user,
            cached_context=formatted_chunks,
            schema=AnswerFeedback,
            tool_name=_TOOL_NAME,
            tool_description=_TOOL_DESCRIPTION,
        )
        log.info("coach.feedback", answered=feedback.answered_question)
        return feedback
