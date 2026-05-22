from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from jobpilot.config import Settings
from jobpilot.evals.metrics import EvalRecord
from jobpilot.evals.runner import run_batch
from jobpilot.models.schemas import EvaluationResult, JobDescription, ProfileChunk
from jobpilot.models.state import AgentState
from tests.fixtures.evals import make_eval_case


class _StubGraph:
    """Mimics the LangGraph .ainvoke() contract used by orchestrator.build_graph()."""

    def __init__(self, response_by_stem: dict[str, Any]) -> None:
        self._response = response_by_stem

    async def ainvoke(self, state: AgentState) -> AgentState:
        stem = state["job"].source.replace(".txt", "")
        res = self._response[stem]
        if isinstance(res, Exception):
            raise res
        return res


def _state_for(stem: str, *, decision="apply", score=80) -> AgentState:
    chunk = ProfileChunk(id="c1", source="cv.docx", heading_path=["Work"], text="RAG pipeline")
    return AgentState(
        job=JobDescription(source=f"{stem}.txt", body="jd"),
        retrieved=[chunk],
        evaluation=EvaluationResult(score=score, decision=decision, reasoning="r",
                                    cited_chunk_ids=["c1"], risk_flags=[]),
        tailored=None,
        output_paths={},
    )


@pytest.mark.asyncio
async def test_run_batch_assembles_eval_records(settings: Settings) -> None:
    cases = [
        make_eval_case("a", decision="apply", score_band=(70, 90)),
        make_eval_case("b", decision="apply", score_band=(70, 90)),
    ]
    graph = _StubGraph({
        "a": _state_for("a", decision="apply", score=80),
        "b": _state_for("b", decision="maybe", score=60),
    })
    llm = MagicMock()
    llm.record.return_value.__enter__.return_value = []  # no calls recorded
    llm.record.return_value.__exit__.return_value = None

    records = await run_batch(
        cases=cases, graph=graph, llm=llm,
        prompt_version="v1", model="m",
    )

    assert isinstance(records, list) and all(isinstance(r, EvalRecord) for r in records)
    assert [r.stem for r in records] == ["a", "b"]
    assert records[0].actual is not None and records[0].actual.decision == "apply"
    assert records[1].actual is not None and records[1].actual.score == 60
    assert all(r.error is None for r in records)


@pytest.mark.asyncio
async def test_run_batch_captures_error_and_continues(settings: Settings) -> None:
    cases = [make_eval_case("a"), make_eval_case("b")]
    graph = _StubGraph({
        "a": _state_for("a"),
        "b": RuntimeError("graph blew up"),
    })
    llm = MagicMock()
    llm.record.return_value.__enter__.return_value = []
    llm.record.return_value.__exit__.return_value = None

    records = await run_batch(
        cases=cases, graph=graph, llm=llm,
        prompt_version="v1", model="m",
    )

    assert records[0].error is None
    assert records[1].error is not None
    assert records[1].error.type == "RuntimeError"
    assert "graph blew up" in records[1].error.message
    assert records[1].actual is None
    assert records[1].metrics is None
