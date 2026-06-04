from __future__ import annotations

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


def _make_llm(result: EvaluationResult) -> MagicMock:
    """Return a mock ChatOpenAI whose with_structured_output().invoke() returns result."""
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.return_value = result
    return llm


@pytest.mark.asyncio
async def test_evaluator_populates_state(settings: Settings) -> None:
    chunks = [
        ProfileChunk(id="cv-1", source="cv.docx", heading_path=["Exp"], text="Built RAG pipeline."),
        ProfileChunk(id="cv-2", source="cv.docx", heading_path=["Exp"], text="GCP migration."),
    ]
    rag = MagicMock()
    rag.query.return_value = chunks

    expected = EvaluationResult(
        score=84,
        decision="apply",
        reasoning="Strong backend + RAG match.",
        cited_chunk_ids=["cv-1"],
        risk_flags=[],
    )
    llm = _make_llm(expected)

    agent = EvaluatorAgent(settings=settings, llm=llm, rag=rag)
    state: AgentState = {
        "job": JobDescription(source="canva.txt", body="Senior fullstack at Canva."),
    }

    out = await agent.run(state)

    assert out["evaluation"] is not None
    assert out["evaluation"].score == 84
    assert out["retrieved"] == chunks
    rag.query.assert_called_once()

    # with_structured_output must have been called with the correct schema
    llm.with_structured_output.assert_called_once_with(EvaluationResult, method="function_calling")

    # invoke must have been called; inspect the messages list
    invoke_call = llm.with_structured_output.return_value.invoke.call_args
    messages = invoke_call.args[0]
    # First message is the system prompt
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    assert isinstance(messages[0], SystemMessage)
    # cached_context produces a HumanMessage + AIMessage pair
    assert any(isinstance(m, HumanMessage) and "Built RAG pipeline." in m.content for m in messages)
    assert any(isinstance(m, AIMessage) and m.content == "Understood." for m in messages)
