"""Eval-harness Pydantic DTOs + per-case and aggregate scoring."""

from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

from jobpilot.evals.fixtures import ExpectedOutcome
from jobpilot.models.schemas import Decision


class CallTelemetryDTO(BaseModel):
    model_config = ConfigDict(frozen=True)
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class CaseTelemetry(BaseModel):
    model_config = ConfigDict(frozen=True)
    latency_ms: float
    calls: list[CallTelemetryDTO] = Field(default_factory=list)
    input_tokens: int
    output_tokens: int
    estimated_usd: float | None


class CaseError(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: str
    message: str


class ActualOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)
    decision: Decision
    score: int
    reasoning: str
    cited_chunk_ids: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)


class PerCaseMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)
    decision_correct: bool
    score_in_band: bool | None
    score_abs_error: float | None
    citation_evidence_ok: bool | None
    required_risk_flags_present: bool | None
    disallowed_risk_flags_absent: bool | None
    cited_chunks_retrieved: bool


class EvalRecord(BaseModel):
    """One row in `results.jsonl`."""

    model_config = ConfigDict(frozen=True)

    stem: str
    ts: str
    prompt_version: str
    model: str
    expected: ExpectedOutcome
    actual: ActualOutcome | None
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    metrics: PerCaseMetrics | None
    telemetry: CaseTelemetry
    error: CaseError | None


class BatchConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    run_id: str
    ts: str
    model: str
    prompt_version: str
    settings: dict[str, Any] = Field(default_factory=dict)
    git_sha: str | None = None


class BatchAggregates(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_cases: int
    n_errors: int
    decision_accuracy: float
    decision_correct_count: int
    decision_total: int
    score_mae: float | None
    score_mae_n: int
    score_in_band_rate: float | None
    score_in_band_count: int
    score_band_total: int
    citation_evidence_pass_rate: float | None
    citation_evidence_count: int
    citation_evidence_total: int
    required_risk_flags_pass_rate: float | None
    required_risk_flags_count: int
    required_risk_flags_total: int
    disallowed_risk_flags_pass_rate: float | None
    disallowed_risk_flags_count: int
    disallowed_risk_flags_total: int
    p50_latency_ms: float
    p95_latency_ms: float
    total_input_tokens: int
    total_output_tokens: int
    total_usd: float | None
    mean_usd_per_known_case: float | None


class ScoredBatch(BaseModel):
    model_config = ConfigDict(frozen=True)
    config: BatchConfig
    records: list[EvalRecord]
    aggregates: BatchAggregates


# ---- scoring ----------------------------------------------------------------


def score_case(
    *,
    expected: ExpectedOutcome,
    actual: ActualOutcome,
    retrieved_chunk_ids: list[str],
    cited_chunk_texts: dict[str, str],
) -> PerCaseMetrics:
    """Compute deterministic per-case metrics. See spec §7.1."""
    decision_correct = actual.decision == expected.expected_decision

    if expected.expected_score_band is not None:
        lo, hi = expected.expected_score_band
        midpoint = (lo + hi) / 2.0
        score_in_band: bool | None = lo <= actual.score <= hi
        score_abs_error: float | None = abs(actual.score - midpoint)
    else:
        score_in_band = None
        score_abs_error = None

    if expected.key_evidence_chunk_substrings:
        cited_joined = " ".join(cited_chunk_texts.values()).lower()
        # Lenient: a single matching anchor passes. Use multiple substrings to broaden
        # what counts as 'right evidence', not to require all of them.
        citation_evidence_ok: bool | None = any(
            sub.lower() in cited_joined for sub in expected.key_evidence_chunk_substrings
        )
    else:
        citation_evidence_ok = None

    flags_joined = " ".join(actual.risk_flags).lower()
    if expected.required_risk_flags:
        required_present: bool | None = all(
            r.lower() in flags_joined for r in expected.required_risk_flags
        )
    else:
        required_present = None

    if expected.disallowed_risk_flags:
        disallowed_absent: bool | None = not any(
            d.lower() in flags_joined for d in expected.disallowed_risk_flags
        )
    else:
        disallowed_absent = None

    cited_chunks_retrieved = set(actual.cited_chunk_ids).issubset(set(retrieved_chunk_ids))

    return PerCaseMetrics(
        decision_correct=decision_correct,
        score_in_band=score_in_band,
        score_abs_error=score_abs_error,
        citation_evidence_ok=citation_evidence_ok,
        required_risk_flags_present=required_present,
        disallowed_risk_flags_absent=disallowed_absent,
        cited_chunks_retrieved=cited_chunks_retrieved,
    )


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile. Returns 0.0 for empty input."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * pct
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def aggregate(records: list[EvalRecord]) -> BatchAggregates:
    """Aggregate per-case records into batch-level metrics. See spec §7.2."""
    n_cases = len(records)
    errored = [r for r in records if r.error is not None]
    # ok_metrics: pairs of (record, metrics) where metrics is guaranteed non-None.
    # Typed explicitly so mypy can see the narrowed PerCaseMetrics type.
    ok_raw = [r for r in records if r.error is None and r.metrics is not None]
    ok: list[tuple[EvalRecord, PerCaseMetrics]] = [
        (r, cast(PerCaseMetrics, r.metrics)) for r in ok_raw
    ]
    n_errors = len(errored)

    decision_correct_count = sum(1 for _, m in ok if m.decision_correct)
    decision_total = len(ok)
    decision_accuracy = (decision_correct_count / decision_total) if decision_total else 0.0

    score_pairs = [(r, m) for r, m in ok if m.score_in_band is not None]
    score_in_band_count = sum(1 for _, m in score_pairs if m.score_in_band)
    score_band_total = len(score_pairs)
    score_in_band_rate = (score_in_band_count / score_band_total) if score_band_total else None
    score_mae_values = [m.score_abs_error for _, m in score_pairs if m.score_abs_error is not None]
    score_mae = (sum(score_mae_values) / len(score_mae_values)) if score_mae_values else None

    def _flag_pass_rate(attr: str) -> tuple[float | None, int, int]:
        rs = [(r, m) for r, m in ok if getattr(m, attr) is not None]
        passes = sum(1 for _, m in rs if getattr(m, attr))
        total = len(rs)
        rate = (passes / total) if total else None
        return rate, passes, total

    cit_rate, cit_pass, cit_tot = _flag_pass_rate("citation_evidence_ok")
    req_rate, req_pass, req_tot = _flag_pass_rate("required_risk_flags_present")
    dis_rate, dis_pass, dis_tot = _flag_pass_rate("disallowed_risk_flags_absent")

    latencies = [r.telemetry.latency_ms for r in records]
    p50 = _percentile(latencies, 0.5)
    p95 = _percentile(latencies, 0.95)

    total_input_tokens = sum(r.telemetry.input_tokens for r in records)
    total_output_tokens = sum(r.telemetry.output_tokens for r in records)
    known_costs = [
        r.telemetry.estimated_usd for r in records if r.telemetry.estimated_usd is not None
    ]
    total_usd: float | None
    mean_usd_per_known_case: float | None
    if known_costs:
        total_usd = sum(known_costs)
        mean_usd_per_known_case = total_usd / len(known_costs)
    else:
        total_usd = None
        mean_usd_per_known_case = None

    return BatchAggregates(
        n_cases=n_cases,
        n_errors=n_errors,
        decision_accuracy=decision_accuracy,
        decision_correct_count=decision_correct_count,
        decision_total=decision_total,
        score_mae=score_mae,
        score_mae_n=score_band_total,
        score_in_band_rate=score_in_band_rate,
        score_in_band_count=score_in_band_count,
        score_band_total=score_band_total,
        citation_evidence_pass_rate=cit_rate,
        citation_evidence_count=cit_pass,
        citation_evidence_total=cit_tot,
        required_risk_flags_pass_rate=req_rate,
        required_risk_flags_count=req_pass,
        required_risk_flags_total=req_tot,
        disallowed_risk_flags_pass_rate=dis_rate,
        disallowed_risk_flags_count=dis_pass,
        disallowed_risk_flags_total=dis_tot,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_usd=total_usd,
        mean_usd_per_known_case=mean_usd_per_known_case,
    )
