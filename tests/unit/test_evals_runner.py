from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from jobpilot.config import Settings
from jobpilot.evals.metrics import EvalRecord
from jobpilot.evals.runner import run_batch
from jobpilot.llm.client import CallTelemetry
from jobpilot.models.schemas import EvaluationResult, JobDescription, ProfileChunk
from jobpilot.models.state import AgentState
from tests.fixtures.evals import make_eval_case


class _StubGraph:
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
        evaluation=EvaluationResult(
            score=score,
            decision=decision,
            reasoning="r",
            cited_chunk_ids=["c1"],
            risk_flags=[],
        ),
        tailored=None,
        output_paths={},
    )


def _patch_record(bucket: list[CallTelemetry] | None = None):
    """Patch jobpilot.evals.runner.record to yield the given bucket."""
    actual_bucket: list[CallTelemetry] = bucket if bucket is not None else []

    @contextmanager
    def _fake_record(llm):
        yield actual_bucket

    return patch("jobpilot.evals.runner.record", side_effect=_fake_record)


@pytest.mark.asyncio
async def test_run_batch_assembles_eval_records(settings: Settings) -> None:
    cases = [
        make_eval_case("a", decision="apply", score_band=(70, 90)),
        make_eval_case("b", decision="apply", score_band=(70, 90)),
    ]
    graph = _StubGraph(
        {
            "a": _state_for("a", decision="apply", score=80),
            "b": _state_for("b", decision="maybe", score=60),
        }
    )
    llm = MagicMock()

    with _patch_record():
        records = await run_batch(
            cases=cases,
            graph=graph,
            llm=llm,
            prompt_version="v1",
            model="m",
        )

    assert isinstance(records, list) and all(isinstance(r, EvalRecord) for r in records)
    assert [r.stem for r in records] == ["a", "b"]
    assert records[0].actual is not None and records[0].actual.decision == "apply"
    assert records[1].actual is not None and records[1].actual.score == 60
    assert all(r.error is None for r in records)


@pytest.mark.asyncio
async def test_run_batch_captures_error_and_continues(settings: Settings) -> None:
    cases = [make_eval_case("a"), make_eval_case("b")]
    graph = _StubGraph({"a": _state_for("a"), "b": RuntimeError("graph blew up")})
    llm = MagicMock()

    with _patch_record():
        records = await run_batch(
            cases=cases,
            graph=graph,
            llm=llm,
            prompt_version="v1",
            model="m",
        )

    assert records[0].error is None
    assert records[1].error is not None
    assert records[1].error.type == "RuntimeError"
    assert "graph blew up" in records[1].error.message


@pytest.mark.asyncio
async def test_run_batch_records_error_when_graph_returns_no_evaluation(
    settings: Settings,
) -> None:
    chunk = ProfileChunk(id="c1", source="cv.docx", heading_path=["Work"], text="x")
    state_with_no_eval = AgentState(
        job=JobDescription(source="a.txt", body="jd"),
        retrieved=[chunk],
        evaluation=None,
        tailored=None,
        output_paths={},
    )
    cases = [make_eval_case("a")]
    graph = _StubGraph({"a": state_with_no_eval})
    llm = MagicMock()

    with _patch_record():
        records = await run_batch(
            cases=cases,
            graph=graph,
            llm=llm,
            prompt_version="v1",
            model="m",
        )

    rec = records[0]
    assert rec.error is not None
    assert rec.error.type == "ValueError"
    assert "no evaluation" in rec.error.message


@pytest.mark.asyncio
async def test_run_batch_preserves_telemetry_when_graph_raises_mid_call(
    settings: Settings,
) -> None:
    """Regression: telemetry captured before a mid-call exception must appear in the error record."""
    pre_captured = [CallTelemetry(model="m", input_tokens=100, output_tokens=20, latency_ms=50.0)]
    cases = [make_eval_case("a")]

    class _RaisesGraph:
        async def ainvoke(self, state: AgentState) -> AgentState:
            raise RuntimeError("graph blew up mid-call")

    llm = MagicMock()

    with _patch_record(pre_captured):
        records = await run_batch(
            cases=cases,
            graph=_RaisesGraph(),
            llm=llm,
            prompt_version="v1",
            model="m",
        )

    rec = records[0]
    assert rec.error is not None
    assert rec.telemetry.input_tokens == 100
    assert rec.telemetry.output_tokens == 20
    assert len(rec.telemetry.calls) == 1
