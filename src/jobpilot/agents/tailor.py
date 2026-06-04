"""Tailor agent: RAG over profile → structured TailoredCV → AgentState update."""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from jobpilot.config import Settings
from jobpilot.logging_setup import get_logger
from jobpilot.models.schemas import ProfileChunk, TailoredCV
from jobpilot.models.state import AgentState
from jobpilot.prompts.loader import PromptLoader
from jobpilot.rag.store import RagStore
from jobpilot.tools.bullet_pool import BulletPool

log = get_logger(__name__)


def _format_chunks_for_prompt(chunks: list[ProfileChunk]) -> str:
    parts: list[str] = []
    for c in chunks:
        heading = " > ".join(c.heading_path) or "(no heading)"
        parts.append(f"[{c.id}] {heading}\n{c.text}")
    return "\n\n".join(parts)


class TailorAgent:
    def __init__(
        self,
        *,
        settings: Settings,
        llm: ChatOpenAI,
        rag: RagStore,
        pool: BulletPool,
        prompts: PromptLoader | None = None,
        prompt_version: str = "v1",
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._rag = rag
        self._pool = pool
        self._prompts = prompts or PromptLoader(Path(__file__).parent.parent / "prompts")
        self._version = prompt_version

    async def run(self, state: AgentState) -> AgentState:
        job = state["job"]
        chunks = state.get("retrieved")
        if chunks is None:
            chunks = self._rag.query(job.body, k=self._settings.retrieval_k)
            log.info("tailor.retrieved", count=len(chunks))
        else:
            log.info("tailor.reused_retrieved", count=len(chunks))

        cached_context = _format_chunks_for_prompt(chunks)
        prompt = self._prompts.render(
            "tailor",
            self._version,
            variables={
                "k": self._settings.retrieval_k,
                "jd_source": job.source,
                "jd_body": job.body,
                "profile_chunks": cached_context,
                "current_role_pool": self._pool.render_for_prompt(),
            },
        )

        messages = [SystemMessage(prompt.system)]
        if cached_context:
            messages += [HumanMessage(cached_context), AIMessage("Understood.")]
        messages.append(HumanMessage(prompt.user))

        result: TailoredCV = self._llm.with_structured_output(TailoredCV).invoke(messages)

        self._pool.resolve_current_role(result.current_role_bullet_ids)

        log.info(
            "tailor.done",
            company=result.target_company,
            role=result.target_role,
            bullets=len(result.current_role_bullet_ids),
        )
        return {**state, "retrieved": chunks, "tailored": result}
