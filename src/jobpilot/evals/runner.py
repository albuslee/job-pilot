"""Batch runner: invoke the production graph per case + capture telemetry."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Protocol

from jobpilot.evals.fixtures import EvalCase
from jobpilot.evals.metrics import (
    ActualOutcome,
    CallTelemetryDTO,
    CaseError,
    CaseTelemetry,
    EvalRecord,
    score_case,
)
from jobpilot.evals.pricing import PriceTable, cost
from jobpilot.llm.client import CallTelemetry, LLMClient
from jobpilot.logging_setup import get_logger
from jobpilot.models.state import AgentState

log = get_logger(__name__)


class _GraphLike(Protocol):
    async def ainvoke(self, state: AgentState) -> AgentState: ...


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


async def run_batch(
    *,
    cases: list[EvalCase],
    graph: _GraphLike,
    llm: LLMClient,
    prompt_version: str,
    model: str,
    prices: PriceTable | None = None,
) -> list[EvalRecord]:
    """Run each case through `graph` and assemble EvalRecord rows."""
    records: list[EvalRecord] = []
    for case in cases:
        ts = _now_iso()
        t0 = time.monotonic()
        calls: list[CallTelemetry] = []
        try:
            with llm.record() as recorded:
                state: AgentState = {"job": case.jd}
                out = await graph.ainvoke(state)
            calls = recorded                       # capture for any post-with exception path
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            evaluation = out["evaluation"]
            if evaluation is None:
                raise ValueError("graph returned no evaluation")
            retrieved = out.get("retrieved") or []
            retrieved_ids = [c.id for c in retrieved]
            cited_texts = {c.id: c.text for c in retrieved if c.id in evaluation.cited_chunk_ids}
            actual = ActualOutcome(
                decision=evaluation.decision,
                score=evaluation.score,
                reasoning=evaluation.reasoning,
                cited_chunk_ids=list(evaluation.cited_chunk_ids),
                risk_flags=list(evaluation.risk_flags),
            )
            metrics = score_case(
                expected=case.expected, actual=actual,
                retrieved_chunk_ids=retrieved_ids, cited_chunk_texts=cited_texts,
            )
            call_dtos = [CallTelemetryDTO(model=c.model, input_tokens=c.input_tokens,
                                          output_tokens=c.output_tokens, latency_ms=c.latency_ms)
                         for c in calls]
            tot_in = sum(c.input_tokens for c in call_dtos)
            tot_out = sum(c.output_tokens for c in call_dtos)
            usd = cost(model, tot_in, tot_out, prices=prices) if prices is not None else None
            telemetry = CaseTelemetry(
                latency_ms=elapsed_ms, calls=call_dtos,
                input_tokens=tot_in, output_tokens=tot_out, estimated_usd=usd,
            )
            records.append(EvalRecord(
                stem=case.stem, ts=ts, prompt_version=prompt_version, model=model,
                expected=case.expected, actual=actual, retrieved_chunk_ids=retrieved_ids,
                metrics=metrics, telemetry=telemetry, error=None,
            ))
            log.info("eval.case_done", stem=case.stem, decision=actual.decision,
                     score=actual.score, latency_ms=elapsed_ms)
        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            # Preserve whatever telemetry was captured before the exception fired.
            call_dtos = [CallTelemetryDTO(model=c.model, input_tokens=c.input_tokens,
                                          output_tokens=c.output_tokens, latency_ms=c.latency_ms)
                         for c in calls]
            tot_in = sum(c.input_tokens for c in call_dtos)
            tot_out = sum(c.output_tokens for c in call_dtos)
            usd = cost(model, tot_in, tot_out, prices=prices) if prices is not None else None
            records.append(EvalRecord(
                stem=case.stem, ts=ts, prompt_version=prompt_version, model=model,
                expected=case.expected, actual=None, retrieved_chunk_ids=[], metrics=None,
                telemetry=CaseTelemetry(latency_ms=elapsed_ms, calls=call_dtos,
                                        input_tokens=tot_in, output_tokens=tot_out, estimated_usd=usd),
                error=CaseError(type=type(exc).__name__, message=str(exc)),
            ))
            log.warning("eval.case_failed", stem=case.stem, error=str(exc), exc_info=True)
    return records
