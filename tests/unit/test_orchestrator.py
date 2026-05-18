from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from jobpilot.agents.orchestrator import build_graph
from jobpilot.config import Settings
from jobpilot.models.schemas import (
    EvaluationResult,
    JobDescription,
    ProfileChunk,
    TailoredCV,
)


def _chunks() -> list[ProfileChunk]:
    return [
        ProfileChunk(id="cv-1", source="cv.docx", heading_path=["X"], text="x"),
    ]


def _evaluator(score: int) -> Any:
    agent = MagicMock()

    async def run(state: dict[str, Any]) -> dict[str, Any]:
        return {
            **state,
            "retrieved": _chunks(),
            "evaluation": EvaluationResult(
                score=score,
                decision="apply" if score >= 70 else "skip",
                reasoning="r",
                cited_chunk_ids=[],
                risk_flags=[],
            ),
        }

    agent.run = AsyncMock(side_effect=run)
    return agent


def _tailor() -> Any:
    agent = MagicMock()

    async def run(state: dict[str, Any]) -> dict[str, Any]:
        return {
            **state,
            "tailored": TailoredCV(
                target_company="Canva",
                target_role="Senior Fullstack Engineer",
                summary="s",
                skills_lines=["a"],
                current_role_bullet_ids=["entry-a"],
            ),
        }

    agent.run = AsyncMock(side_effect=run)
    return agent


@pytest.mark.asyncio
async def test_graph_skips_tailor_below_threshold(settings: Settings) -> None:
    evaluator = _evaluator(score=40)
    tailor = _tailor()
    graph = build_graph(settings=settings, evaluator=evaluator, tailor=tailor)

    out = await graph.ainvoke({"job": JobDescription(source="x.txt", body="x")})
    assert out["evaluation"].score == 40
    assert out.get("tailored") is None
    tailor.run.assert_not_called()


@pytest.mark.asyncio
async def test_graph_runs_tailor_when_above_threshold(settings: Settings) -> None:
    evaluator = _evaluator(score=85)
    tailor = _tailor()
    graph = build_graph(settings=settings, evaluator=evaluator, tailor=tailor)

    out = await graph.ainvoke({"job": JobDescription(source="x.txt", body="x")})
    assert out["evaluation"].score == 85
    assert out["tailored"].target_company == "Canva"
    tailor.run.assert_awaited_once()
