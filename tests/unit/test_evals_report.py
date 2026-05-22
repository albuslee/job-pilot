from __future__ import annotations

import json
from pathlib import Path

from jobpilot.evals.fixtures import ExpectedOutcome
from jobpilot.evals.metrics import (
    ActualOutcome,
    BatchConfig,
    CallTelemetryDTO,
    CaseTelemetry,
    EvalRecord,
    PerCaseMetrics,
    ScoredBatch,
)
from jobpilot.evals.report import (
    write_meta_yaml,
    write_report_md,
    write_results_jsonl,
)


def _record(stem: str, *, decision="apply", expected_decision="apply", score=80,
            error=False) -> EvalRecord:
    expected = ExpectedOutcome(expected_decision=expected_decision, expected_score_band=(70, 90))  # type: ignore[arg-type]
    base = dict(
        stem=stem, ts="2026-05-22T00:00:00Z", prompt_version="v1", model="m",
        expected=expected, retrieved_chunk_ids=["c1"],
        telemetry=CaseTelemetry(
            latency_ms=1000.0,
            calls=[CallTelemetryDTO(model="m", input_tokens=100, output_tokens=20, latency_ms=900.0)],
            input_tokens=100, output_tokens=20, estimated_usd=0.001,
        ),
    )
    if error:
        from jobpilot.evals.metrics import CaseError
        return EvalRecord(**base, actual=None, metrics=None,
                          error=CaseError(type="RuntimeError", message="boom"))
    actual = ActualOutcome(decision=decision, score=score, reasoning="r",
                           cited_chunk_ids=["c1"], risk_flags=[])
    metrics = PerCaseMetrics(
        decision_correct=(decision == expected_decision),
        score_in_band=(70 <= score <= 90),
        score_abs_error=abs(score - 80),
        citation_evidence_ok=None,
        required_risk_flags_present=None,
        disallowed_risk_flags_absent=None,
        cited_chunks_retrieved=True,
    )
    return EvalRecord(**base, actual=actual, metrics=metrics, error=None)


def _scored(records: list[EvalRecord]) -> ScoredBatch:
    from jobpilot.evals.metrics import aggregate
    return ScoredBatch(
        config=BatchConfig(run_id="2026-05-22T00-00-00", ts="2026-05-22T00:00:00Z",
                           model="claude-sonnet-4-6", prompt_version="v1",
                           settings={"retrieval_k": 8}, git_sha="deadbeef"),
        records=records,
        aggregates=aggregate(records),
    )


def test_write_results_jsonl_one_row_per_record(tmp_path: Path) -> None:
    scored = _scored([_record("a", score=80), _record("b", score=60)])
    out = tmp_path / "results.jsonl"
    write_results_jsonl(scored, out)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rows = [json.loads(line) for line in lines]
    assert {r["stem"] for r in rows} == {"a", "b"}
    assert rows[0]["model"] == "m"  # from per-record EvalRecord.model, not BatchConfig.model
    assert rows[0]["telemetry"]["estimated_usd"] == 0.001


def test_write_meta_yaml(tmp_path: Path) -> None:
    scored = _scored([_record("a", score=80)])
    out = tmp_path / "meta.yaml"
    write_meta_yaml(scored, out)
    body = out.read_text(encoding="utf-8")
    assert "run_id: 2026-05-22T00-00-00" in body
    assert "model: claude-sonnet-4-6" in body
    assert "n_cases: 1" in body


def test_report_md_contains_summary(tmp_path: Path) -> None:
    scored = _scored([_record("a", score=80), _record("b", decision="maybe", score=60)])
    out = tmp_path / "report.md"
    write_report_md(scored, out, baseline=None)
    body = out.read_text(encoding="utf-8")
    assert "JobPilot Eval" in body
    assert "decision_accuracy" in body
    assert "score_mae" in body
    assert "1/2" in body  # one of two correct


def test_report_md_lists_failures_sorted_by_stem(tmp_path: Path) -> None:
    # Two failing cases: 'z' (score out of band) + 'a' (decision wrong)
    records = [
        _record("z", score=60),                              # out-of-band
        _record("a", decision="maybe"),                      # wrong decision
        _record("b", score=80),                              # passes
    ]
    scored = _scored(records)
    out = tmp_path / "report.md"
    write_report_md(scored, out, baseline=None)
    body = out.read_text(encoding="utf-8")
    # Sorted by stem: 'a' appears before 'z'
    a_idx = body.find("### a ")
    z_idx = body.find("### z ")
    # If a / z headings include trailing content, allow no trailing space:
    if a_idx == -1:
        a_idx = body.find("### a\n")
    if z_idx == -1:
        z_idx = body.find("### z\n")
    assert 0 < a_idx < z_idx
    assert "### b " not in body and "### b\n" not in body  # passing cases not in failures


def test_report_md_lists_errors(tmp_path: Path) -> None:
    scored = _scored([_record("err", error=True), _record("ok", score=80)])
    out = tmp_path / "report.md"
    write_report_md(scored, out, baseline=None)
    body = out.read_text(encoding="utf-8")
    assert "Errors (1)" in body
    assert "err" in body
    assert "RuntimeError" in body
    # Errored records must not appear in Failures
    assert "Failures (0)" in body


def test_report_md_no_baseline_section(tmp_path: Path) -> None:
    scored = _scored([_record("a", score=80)])
    out = tmp_path / "report.md"
    write_report_md(scored, out, baseline=None)
    body = out.read_text(encoding="utf-8")
    assert "Delta vs baseline" not in body
    assert "Regressions" not in body
