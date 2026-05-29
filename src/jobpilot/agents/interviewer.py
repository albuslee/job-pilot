"""Mock-mode interviewer agent: generates one natural follow-up, or signals stop.

Sync; no RAG (it only reacts to the conversation, it does not ground in the CV).
"""

from __future__ import annotations

from pathlib import Path

from jobpilot.config import Settings
from jobpilot.llm.client import LLMClient
from jobpilot.logging_setup import get_logger
from jobpilot.models.schemas import FollowUp, InterviewQuestion, InterviewTurn
from jobpilot.prompts.loader import PromptLoader

log = get_logger(__name__)

_TOOL_NAME = "submit_followup"
_TOOL_DESCRIPTION = (
    "Submit one follow-up question that probes the candidate's most recent answer, "
    "or set should_continue=false (question=null) when the thread is complete."
)


def _format_history_for_prompt(history: list[InterviewTurn]) -> str:
    parts: list[str] = []
    for t in history:
        parts.append(f"Q: {t.question.text}\nA: {t.transcript}")
    return "\n\n".join(parts)


class InterviewerAgent:
    def __init__(
        self,
        *,
        settings: Settings,
        llm: LLMClient,
        prompts: PromptLoader | None = None,
        prompt_version: str = "v1",
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._prompts = prompts or PromptLoader(Path(__file__).parent.parent / "prompts")
        self._version = prompt_version

    def next_followup(
        self, seed: InterviewQuestion, history: list[InterviewTurn]
    ) -> FollowUp:
        prompt = self._prompts.render(
            "interviewer",
            self._version,
            variables={
                "category": seed.category,
                "seed_question": seed.text,
                "history": _format_history_for_prompt(history),
            },
        )
        result = self._llm.complete_structured(
            system=prompt.system,
            user=prompt.user,
            cached_context=None,
            schema=FollowUp,
            tool_name=_TOOL_NAME,
            tool_description=_TOOL_DESCRIPTION,
        )
        log.info("interviewer.followup", should_continue=result.should_continue)
        return result
