"""Evaluator agent: RAG over profile → LangChain structured output → AgentState update."""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from jobpilot.config import Settings
from jobpilot.logging_setup import get_logger
from jobpilot.models.schemas import EvaluationResult, ProfileChunk
from jobpilot.models.state import AgentState
from jobpilot.prompts.loader import PromptLoader
from jobpilot.rag.store import RagStore

log = get_logger(__name__)


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
        llm: ChatOpenAI,
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

        cached_context = _format_chunks_for_prompt(chunks)
        prompt = self._prompts.render(
            "evaluator",
            self._version,
            variables={
                "threshold": self._settings.score_threshold,
                "k": self._settings.retrieval_k,
                "jd_source": job.source,
                "jd_body": job.body,
                "profile_chunks": cached_context,
            },
        )

        messages = [SystemMessage(prompt.system)]
        if cached_context:
            messages += [HumanMessage(cached_context), AIMessage("Understood.")]
        messages.append(HumanMessage(prompt.user))

        # method="function_calling": Bedrock rejects json_schema mode when the schema
        # has integer min/max constraints (ge/le on Pydantic fields).
        result: EvaluationResult = self._llm.with_structured_output(
            EvaluationResult, method="function_calling"
        ).invoke(messages)
        log.info("evaluator.scored", score=result.score, decision=result.decision)

        return {**state, "retrieved": chunks, "evaluation": result}
