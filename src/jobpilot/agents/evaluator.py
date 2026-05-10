"""Evaluator agent: RAG over profile → Anthropic structured output → AgentState update."""

from __future__ import annotations

from pathlib import Path

from jobpilot.config import Settings
from jobpilot.llm.client import AnthropicClient
from jobpilot.logging_setup import get_logger
from jobpilot.models.schemas import EvaluationResult, ProfileChunk
from jobpilot.models.state import AgentState
from jobpilot.prompts.loader import PromptLoader
from jobpilot.rag.store import RagStore

log = get_logger(__name__)

_TOOL_NAME = "submit_evaluation"
_TOOL_DESCRIPTION = (
    "Submit your structured evaluation of the candidate-job fit. "
    "All required fields must be populated."
)


def _format_chunks_for_prompt(chunks: list[ProfileChunk]) -> str:
    parts: list[str] = []
    for c in chunks:
        heading = " > ".join(c.heading_path) or "(no heading)"
        parts.append(f"[{c.id}] {heading}\n{c.text}")
    return "\n\n".join(parts)


class EvaluatorAgent:
    def __init__(
        self,
        *,
        settings: Settings,
        llm: AnthropicClient,
        rag: RagStore,
        prompts: PromptLoader | None = None,
        prompt_version: str = "v1",
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._rag = rag
        self._prompts = prompts or PromptLoader(Path(__file__).parent.parent / "prompts")
        self._version = prompt_version

    async def run(self, state: AgentState) -> AgentState:
        job = state["job"]
        chunks = self._rag.query(job.body, k=self._settings.retrieval_k)
        log.info("evaluator.retrieved", count=len(chunks))

        prompt = self._prompts.render(
            "evaluator",
            self._version,
            variables={
                "threshold": self._settings.score_threshold,
                "k": self._settings.retrieval_k,
                "jd_source": job.source,
                "jd_body": job.body,
                "profile_chunks": _format_chunks_for_prompt(chunks),
            },
        )

        result = self._llm.complete_structured(
            system=prompt.system,
            user=prompt.user,
            cached_context=_format_chunks_for_prompt(chunks),
            schema=EvaluationResult,
            tool_name=_TOOL_NAME,
            tool_description=_TOOL_DESCRIPTION,
        )
        log.info("evaluator.scored", score=result.score, decision=result.decision)

        return {**state, "retrieved": chunks, "evaluation": result}
