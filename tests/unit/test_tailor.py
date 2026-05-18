from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from jobpilot.agents.tailor import TailorAgent
from jobpilot.config import Settings
from jobpilot.models.schemas import JobDescription, ProfileChunk, TailoredCV
from jobpilot.models.state import AgentState
from jobpilot.tools.bullet_pool import (
    BulletEntry,
    BulletPool,
    InvalidBulletIdError,
)


def _pool() -> BulletPool:
    return BulletPool(
        current_role=[
            BulletEntry(id="entry-a", text="Bullet A."),
            BulletEntry(id="entry-b", text="Bullet B."),
            BulletEntry(id="entry-c", text="Bullet C."),
        ]
    )


@pytest.mark.asyncio
async def test_tailor_populates_state(settings: Settings) -> None:
    chunks = [
        ProfileChunk(
            id="cv-1",
            source="cv.docx",
            heading_path=["CAREER OVERVIEW"],
            text="Senior backend engineer with 7+ years on AWS serverless.",
        ),
        ProfileChunk(
            id="cv-2",
            source="cv.docx",
            heading_path=["SKILLS & EXPERTISE"],
            text="Languages: TypeScript, Python",
        ),
    ]
    rag = MagicMock()
    rag.query.return_value = chunks

    llm = MagicMock()
    llm.complete_structured.return_value = TailoredCV(
        target_company="Canva",
        target_role="Senior Fullstack Engineer",
        summary="Seven years of fullstack delivery on AWS, shipping RAG pipelines.",
        skills_lines=[
            "Languages: TypeScript, Python",
            "AWS Serverless: Lambda, Step Functions, EventBridge",
        ],
        current_role_bullet_ids=["entry-b", "entry-a"],
    )

    agent = TailorAgent(settings=settings, llm=llm, rag=rag, pool=_pool())
    state: AgentState = {
        "job": JobDescription(source="canva.txt", body="Senior fullstack at Canva."),
    }

    out = await agent.run(state)

    assert out["tailored"] is not None
    assert out["tailored"].target_company == "Canva"
    assert out["tailored"].current_role_bullet_ids == ["entry-b", "entry-a"]
    assert out["retrieved"] == chunks
    rag.query.assert_called_once()
    call_kwargs: dict[str, Any] = llm.complete_structured.call_args.kwargs
    assert call_kwargs["schema"] is TailoredCV
    assert call_kwargs["tool_name"] == "submit_tailored_cv"
    # Pool must be rendered into the prompt so the LLM knows which IDs exist.
    assert "[entry-a]" in call_kwargs["user"] or "[entry-a]" in call_kwargs["cached_context"]


@pytest.mark.asyncio
async def test_tailor_reuses_existing_retrieved_chunks(settings: Settings) -> None:
    """If the evaluator already retrieved, the tailor MUST NOT requery."""
    existing = [
        ProfileChunk(id="x", source="cv.docx", heading_path=[], text="prior chunk"),
    ]
    rag = MagicMock()
    llm = MagicMock()
    llm.complete_structured.return_value = TailoredCV(
        target_company="X",
        target_role="Y",
        summary="z",
        skills_lines=["a"],
        current_role_bullet_ids=["entry-a"],
    )

    agent = TailorAgent(settings=settings, llm=llm, rag=rag, pool=_pool())
    state: AgentState = {
        "job": JobDescription(source="x.txt", body="x"),
        "retrieved": existing,
    }

    await agent.run(state)
    rag.query.assert_not_called()


@pytest.mark.asyncio
async def test_tailor_rejects_unknown_bullet_id(settings: Settings) -> None:
    """If the LLM hallucinates a bullet id, the agent MUST raise — not silently render."""
    rag = MagicMock()
    rag.query.return_value = []
    llm = MagicMock()
    llm.complete_structured.return_value = TailoredCV(
        target_company="X",
        target_role="Y",
        summary="z",
        skills_lines=["a"],
        current_role_bullet_ids=["entry-a", "entry-fake-hallucinated"],
    )

    agent = TailorAgent(settings=settings, llm=llm, rag=rag, pool=_pool())
    state: AgentState = {"job": JobDescription(source="x.txt", body="x")}

    with pytest.raises(InvalidBulletIdError):
        await agent.run(state)
