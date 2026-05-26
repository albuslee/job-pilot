from __future__ import annotations

from pathlib import Path

import yaml

from jobpilot.evals.compare import diff_runs, load_scored_batch
from jobpilot.evals.fixtures import ExpectedOutcome
from jobpilot.evals.metrics import (
    ActualOutcome,
    BatchConfig,
    CaseError,
    CaseTelemetry,
    EvalRecord,
    PerCaseMetrics,
    ScoredBatch,
    aggregate,
)


def _rec(stem: str, *, decision="apply", score=80, expected_decision="apply") -> EvalRecord:
    expected = ExpectedOutcome(expected_decision=expected_decision, expected_score_band=(70, 90))  # type: ignore[arg-type]
    actual = ActualOutcome(
        decision=decision, score=score, reasoning="r", cited_chunk_ids=["c1"], risk_flags=[]
    )
    metrics = PerCaseMetrics(
        decision_correct=(decision == expected_decision),
        score_in_band=(70 <= score <= 90),
        score_abs_error=abs(score - 80),
        citation_evidence_ok=None,
        required_risk_flags_present=None,
        disallowed_risk_flags_absent=None,
        cited_chunks_retrieved=True,
    )
    return EvalRecord(
        stem=stem,
        ts="2026-05-22T00:00:00Z",
        prompt_version="v1",
        model="m",
        expected=expected,
        actual=actual,
        retrieved_chunk_ids=["c1"],
        metrics=metrics,
        telemetry=CaseTelemetry(
            latency_ms=1.0, calls=[], input_tokens=10, output_tokens=2, estimated_usd=0.0001
        ),
        error=None,
    )


def _scored(records: list[EvalRecord], *, model="m", run_id="rid") -> ScoredBatch:
    return ScoredBatch(
        config=BatchConfig(
            run_id=run_id,
            ts="2026-05-22T00:00:00Z",
            model=model,
            prompt_version="v1",
            settings={},
            git_sha=None,
        ),
        records=records,
        aggregates=aggregate(records),
    )


def test_diff_runs_identical_returns_empty_regressions_and_improvements() -> None:
    a = _scored([_rec("x", score=80)])
    b = _scored([_rec("x", score=80)])
    cmp = diff_runs(a, b)
    assert cmp.regressions == []
    assert cmp.improvements == []
    assert cmp.added_cases == []
    assert cmp.removed_cases == []


def test_diff_runs_detects_regression() -> None:
    baseline = _scored([_rec("x", decision="apply", score=80)])  # pass
    candidate = _scored([_rec("x", decision="maybe", score=60)])  # fail
    cmp = diff_runs(baseline, candidate)
    assert len(cmp.regressions) == 1
    reg = cmp.regressions[0]
    assert reg.stem == "x"
    assert reg.baseline_decision == "apply" and reg.candidate_decision == "maybe"
    assert reg.baseline_score == 80 and reg.candidate_score == 60


def test_diff_runs_detects_improvement() -> None:
    baseline = _scored([_rec("x", decision="maybe", score=60)])  # fail
    candidate = _scored([_rec("x", decision="apply", score=80)])  # pass
    cmp = diff_runs(baseline, candidate)
    assert len(cmp.improvements) == 1


def test_diff_runs_added_and_removed_cases() -> None:
    baseline = _scored([_rec("a", score=80), _rec("b", score=80)])
    candidate = _scored([_rec("b", score=80), _rec("c", score=80)])
    cmp = diff_runs(baseline, candidate)
    assert cmp.added_cases == ["c"]
    assert cmp.removed_cases == ["a"]


def test_load_scored_batch_reads_jsonl_and_meta(tmp_path: Path) -> None:
    run_dir = tmp_path / "rid"
    run_dir.mkdir()
    rec = _rec("x", score=80)
    (run_dir / "results.jsonl").write_text(rec.model_dump_json() + "\n", encoding="utf-8")
    (run_dir / "meta.yaml").write_text(
        yaml.safe_dump(
            {
                "run_id": "rid",
                "ts": "2026-05-22T00:00:00Z",
                "model": "m",
                "prompt_version": "v1",
                "n_cases": 1,
                "settings": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    scored = load_scored_batch(run_dir / "results.jsonl")
    assert scored.config.run_id == "rid"
    assert len(scored.records) == 1
    assert scored.aggregates.n_cases == 1


def test_diff_runs_skips_cases_where_one_side_errored() -> None:
    expected = ExpectedOutcome(expected_decision="apply", expected_score_band=(70, 90))  # type: ignore[arg-type]
    errored = EvalRecord(
        stem="x",
        ts="t",
        prompt_version="v1",
        model="m",
        expected=expected,
        actual=None,
        retrieved_chunk_ids=[],
        metrics=None,
        telemetry=CaseTelemetry(
            latency_ms=1.0, calls=[], input_tokens=0, output_tokens=0, estimated_usd=None
        ),
        error=CaseError(type="X", message="boom"),
    )
    baseline = _scored([errored])
    candidate = _scored([_rec("x", score=80)])  # passes
    cmp = diff_runs(baseline, candidate)
    # error→pass transition is dropped from improvements
    assert cmp.improvements == []
    assert cmp.regressions == []
