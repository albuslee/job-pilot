"""Cover the build_graph(tailor=None) path added for Task 11."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from jobpilot.agents.orchestrator import build_graph
from jobpilot.config import Settings
from jobpilot.models.schemas import EvaluationResult, JobDescription
from jobpilot.models.state import AgentState


@pytest.mark.asyncio
async def test_build_graph_without_tailor_routes_directly_to_end(settings: Settings) -> None:
    """When tailor is None, the graph stops after evaluate regardless of score."""
    evaluator = MagicMock()
    # Even with a very high score, no tailor node exists so tailored stays None
    evaluator.run = AsyncMock(return_value=AgentState(
        job=JobDescription(source="a.txt", body="jd"),
        retrieved=[],
        evaluation=EvaluationResult(score=99, decision="apply", reasoning="r",
                                    cited_chunk_ids=[], risk_flags=[]),
        tailored=None,
        output_paths={},
    ))

    graph = build_graph(settings=settings, evaluator=evaluator, tailor=None)
    out = await graph.ainvoke({"job": JobDescription(source="a.txt", body="jd")})

    assert out["evaluation"].score == 99
    assert out.get("tailored") is None


@pytest.mark.asyncio
async def test_build_graph_with_tailor_unchanged(settings: Settings) -> None:
    """Smoke check: tailor is still optional positional behavior in the not-None path."""
    evaluator = MagicMock()
    evaluator.run = AsyncMock(return_value=AgentState(
        job=JobDescription(source="a.txt", body="jd"),
        retrieved=[],
        evaluation=EvaluationResult(score=10, decision="skip", reasoning="r",
                                    cited_chunk_ids=[], risk_flags=[]),
        tailored=None,
        output_paths={},
    ))
    tailor = MagicMock()
    tailor.run = AsyncMock()

    graph = build_graph(settings=settings, evaluator=evaluator, tailor=tailor)
    out = await graph.ainvoke({"job": JobDescription(source="a.txt", body="jd")})

    assert out["evaluation"].score == 10
    # Score below threshold (default 70) — tailor not invoked
    tailor.run.assert_not_awaited()
