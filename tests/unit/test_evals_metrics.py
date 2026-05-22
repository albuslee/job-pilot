from __future__ import annotations

import pytest

from jobpilot.evals.fixtures import ExpectedOutcome
from jobpilot.evals.metrics import (
    ActualOutcome,
    BatchAggregates,
    CallTelemetryDTO,
    CaseTelemetry,
    EvalRecord,
    PerCaseMetrics,
    aggregate,
    score_case,
)


def _actual(decision="apply", score=80, cited=("c1",), risk_flags=()) -> ActualOutcome:
    return ActualOutcome(
        decision=decision,
        score=score,
        reasoning="ok",
        cited_chunk_ids=list(cited),
        risk_flags=list(risk_flags),
    )


def test_score_case_decision_correct() -> None:
    expected = ExpectedOutcome(expected_decision="apply")
    m = score_case(
        expected=expected,
        actual=_actual(decision="apply"),
        retrieved_chunk_ids=["c1"],
        cited_chunk_texts={"c1": "anything"},
    )
    assert m.decision_correct is True


def test_score_case_decision_wrong() -> None:
    expected = ExpectedOutcome(expected_decision="apply")
    m = score_case(
        expected=expected,
        actual=_actual(decision="maybe"),
        retrieved_chunk_ids=["c1"],
        cited_chunk_texts={"c1": "x"},
    )
    assert m.decision_correct is False


def test_score_band_in_and_out() -> None:
    expected = ExpectedOutcome(expected_decision="apply", expected_score_band=(70, 90))  # type: ignore[arg-type]
    m_in = score_case(expected=expected, actual=_actual(score=80),
                      retrieved_chunk_ids=["c1"], cited_chunk_texts={"c1": "x"})
    assert m_in.score_in_band is True
    assert m_in.score_abs_error == pytest.approx(0.0)  # midpoint = 80

    m_out = score_case(expected=expected, actual=_actual(score=60),
                       retrieved_chunk_ids=["c1"], cited_chunk_texts={"c1": "x"})
    assert m_out.score_in_band is False
    assert m_out.score_abs_error == pytest.approx(20.0)


def test_score_band_na_when_omitted() -> None:
    expected = ExpectedOutcome(expected_decision="apply", expected_score_band=None)
    m = score_case(expected=expected, actual=_actual(score=50),
                   retrieved_chunk_ids=["c1"], cited_chunk_texts={"c1": "x"})
    assert m.score_in_band is None
    assert m.score_abs_error is None


def test_citation_evidence_ok_substring_match() -> None:
    expected = ExpectedOutcome(
        expected_decision="apply",
        key_evidence_chunk_substrings=["RAG", "LangGraph"],
    )
    m_pass = score_case(expected=expected, actual=_actual(cited=("c1",)),
                        retrieved_chunk_ids=["c1"],
                        cited_chunk_texts={"c1": "Built a RAG pipeline using LangGraph"})
    assert m_pass.citation_evidence_ok is True

    m_fail = score_case(expected=expected, actual=_actual(cited=("c1",)),
                        retrieved_chunk_ids=["c1"],
                        cited_chunk_texts={"c1": "Kubernetes cluster operations"})
    assert m_fail.citation_evidence_ok is False


def test_citation_evidence_na_when_empty() -> None:
    expected = ExpectedOutcome(expected_decision="apply", key_evidence_chunk_substrings=[])
    m = score_case(expected=expected, actual=_actual(cited=("c1",)),
                   retrieved_chunk_ids=["c1"], cited_chunk_texts={"c1": "anything"})
    assert m.citation_evidence_ok is None


def test_required_risk_flags() -> None:
    expected = ExpectedOutcome(
        expected_decision="apply", required_risk_flags=["management experience"]
    )
    ok = score_case(expected=expected,
                    actual=_actual(risk_flags=("needs 10+ years management experience",)),
                    retrieved_chunk_ids=["c1"], cited_chunk_texts={"c1": "x"})
    fail = score_case(expected=expected,
                      actual=_actual(risk_flags=()),
                      retrieved_chunk_ids=["c1"], cited_chunk_texts={"c1": "x"})
    assert ok.required_risk_flags_present is True
    assert fail.required_risk_flags_present is False


def test_disallowed_risk_flags() -> None:
    expected = ExpectedOutcome(
        expected_decision="apply", disallowed_risk_flags=["AWS experience"]
    )
    bad = score_case(expected=expected,
                     actual=_actual(risk_flags=("no AWS experience",)),
                     retrieved_chunk_ids=["c1"], cited_chunk_texts={"c1": "x"})
    good = score_case(expected=expected,
                      actual=_actual(risk_flags=()),
                      retrieved_chunk_ids=["c1"], cited_chunk_texts={"c1": "x"})
    assert bad.disallowed_risk_flags_absent is False
    assert good.disallowed_risk_flags_absent is True


def test_cited_chunks_subset_of_retrieved() -> None:
    expected = ExpectedOutcome(expected_decision="apply")
    ok = score_case(expected=expected, actual=_actual(cited=("c1",)),
                    retrieved_chunk_ids=["c1", "c2"], cited_chunk_texts={"c1": "x"})
    bad = score_case(expected=expected, actual=_actual(cited=("ghost",)),
                     retrieved_chunk_ids=["c1"], cited_chunk_texts={})
    assert ok.cited_chunks_retrieved is True
    assert bad.cited_chunks_retrieved is False


def _record(stem: str, *, decision="apply", expected_decision="apply",
            score=80, score_band=(70, 90), error=False) -> EvalRecord:
    from jobpilot.evals.metrics import CaseError
    expected = ExpectedOutcome(expected_decision=expected_decision, expected_score_band=score_band)  # type: ignore[arg-type]
    if error:
        return EvalRecord(
            stem=stem, ts="2026-05-22T00:00:00Z", prompt_version="v1", model="m",
            expected=expected, actual=None, retrieved_chunk_ids=[],
            metrics=None,
            telemetry=CaseTelemetry(latency_ms=10.0, calls=[], input_tokens=0,
                                    output_tokens=0, estimated_usd=None),
            error=CaseError(type="X", message="boom"),
        )
    actual = ActualOutcome(decision=decision, score=score, reasoning="r",
                           cited_chunk_ids=["c1"], risk_flags=[])
    metrics = PerCaseMetrics(
        decision_correct=(decision == expected_decision),
        score_in_band=(score_band[0] <= score <= score_band[1]),
        score_abs_error=abs(score - (score_band[0] + score_band[1]) / 2),
        citation_evidence_ok=None,
        required_risk_flags_present=None,
        disallowed_risk_flags_absent=None,
        cited_chunks_retrieved=True,
    )
    return EvalRecord(
        stem=stem, ts="2026-05-22T00:00:00Z", prompt_version="v1", model="m",
        expected=expected, actual=actual, retrieved_chunk_ids=["c1"],
        metrics=metrics,
        telemetry=CaseTelemetry(
            latency_ms=1000.0,
            calls=[CallTelemetryDTO(model="m", input_tokens=100, output_tokens=20, latency_ms=900.0)],
            input_tokens=100, output_tokens=20, estimated_usd=0.001,
        ),
        error=None,
    )


def test_aggregate_basic_counts() -> None:
    records = [
        _record("a", decision="apply", expected_decision="apply", score=80),
        _record("b", decision="maybe", expected_decision="apply", score=60),
        _record("c", error=True),
    ]
    agg = aggregate(records)
    assert isinstance(agg, BatchAggregates)
    assert agg.n_cases == 3
    assert agg.n_errors == 1
    # decision accuracy excludes errored cases from numerator and denominator
    assert agg.decision_correct_count == 1
    assert agg.decision_total == 2
    assert agg.decision_accuracy == pytest.approx(0.5)


def test_aggregate_score_mae_only_over_banded_cases() -> None:
    r1 = _record("a", score=80)  # in band [70,90], midpoint 80, abs_err 0
    r2 = _record("b", score=60)  # out of band, abs_err 20
    agg = aggregate([r1, r2])
    assert agg.score_mae_n == 2
    assert agg.score_mae == pytest.approx(10.0)


def test_aggregate_p50_p95_latency() -> None:
    records = [_record(f"r{i}", score=80) for i in range(10)]
    # All latencies are 1000.0 in the fixture
    agg = aggregate(records)
    assert agg.p50_latency_ms == pytest.approx(1000.0)
    assert agg.p95_latency_ms == pytest.approx(1000.0)


def test_aggregate_total_usd_sums_with_unknown_model_safe() -> None:
    r1 = _record("a", score=80)
    # mutate one record to have unknown-model cost
    r2 = _record("b", score=80).model_copy(
        update={"telemetry": _record("b", score=80).telemetry.model_copy(update={"estimated_usd": None})}
    )
    agg = aggregate([r1, r2])
    # None entries are skipped; sum is the one known cost
    assert agg.total_usd == pytest.approx(0.001)
    # mean_usd_per_known_case divides by KNOWN cases (1), not total (2)
    assert agg.mean_usd_per_known_case == pytest.approx(0.001)


def test_aggregate_total_usd_none_when_all_unknown() -> None:
    r1 = _record("a", score=80).model_copy(
        update={"telemetry": _record("a", score=80).telemetry.model_copy(update={"estimated_usd": None})}
    )
    agg = aggregate([r1])
    assert agg.total_usd is None
    assert agg.mean_usd_per_known_case is None
