from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from jobpilot.agents.evaluator import EvaluatorAgent
from jobpilot.config import Settings
from jobpilot.models.schemas import (
    EvaluationResult,
    JobDescription,
    ProfileChunk,
)
from jobpilot.models.state import AgentState


@pytest.mark.asyncio
async def test_evaluator_populates_state(settings: Settings) -> None:
    chunks = [
        ProfileChunk(id="cv-1", source="cv.docx", heading_path=["Exp"], text="Built RAG pipeline."),
        ProfileChunk(id="cv-2", source="cv.docx", heading_path=["Exp"], text="GCP migration."),
    ]
    rag = MagicMock()
    rag.query.return_value = chunks

    llm = MagicMock()
    llm.complete_structured.return_value = EvaluationResult(
        score=84,
        decision="apply",
        reasoning="Strong backend + RAG match.",
        cited_chunk_ids=["cv-1"],
        risk_flags=[],
    )

    agent = EvaluatorAgent(settings=settings, llm=llm, rag=rag)
    state: AgentState = {
        "job": JobDescription(source="canva.txt", body="Senior fullstack at Canva."),
    }

    out = await agent.run(state)

    assert out["evaluation"] is not None
    assert out["evaluation"].score == 84
    assert out["retrieved"] == chunks
    rag.query.assert_called_once()
    # The cached_context kwarg should embed both chunks (so prompt caching pays off).
    call_kwargs: dict[str, Any] = llm.complete_structured.call_args.kwargs
    assert "Built RAG pipeline." in call_kwargs["cached_context"]
    assert "GCP migration." in call_kwargs["cached_context"]
    assert call_kwargs["schema"] is EvaluationResult
    assert call_kwargs["tool_name"] == "submit_evaluation"
