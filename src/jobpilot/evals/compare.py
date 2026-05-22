"""Diff two ScoredBatch runs to surface regressions + improvements."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from jobpilot.evals.metrics import (
    BatchConfig,
    EvalRecord,
    ScoredBatch,
    aggregate,
)


class DeltaRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    metric: str
    baseline: str
    candidate: str
    delta: str


class RegressionRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    stem: str
    baseline_decision: str
    candidate_decision: str
    baseline_score: int
    candidate_score: int


class ImprovementRow(RegressionRow):
    """A case that failed in baseline and passes in candidate."""


class ComparisonReport(BaseModel):
    model_config = ConfigDict(frozen=True)
    delta_table: list[DeltaRow] = Field(default_factory=list)
    regressions: list[RegressionRow] = Field(default_factory=list)
    improvements: list[ImprovementRow] = Field(default_factory=list)
    added_cases: list[str] = Field(default_factory=list)
    removed_cases: list[str] = Field(default_factory=list)


def _passes(rec: EvalRecord) -> bool:
    if rec.error is not None or rec.metrics is None:
        return False
    m = rec.metrics
    return not any(
        v is False for v in (
            m.decision_correct, m.score_in_band, m.citation_evidence_ok,
            m.required_risk_flags_present, m.disallowed_risk_flags_absent,
            m.cited_chunks_retrieved,
        )
    )


def _fmt_pp_delta(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return "n/a"
    return f"{(b - a) * 100:+.1f}pp"


def _fmt_usd(x: float | None) -> str:
    return f"${x:.3f}" if x is not None else "n/a"


def _fmt_usd_delta(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return "n/a"
    return f"{b - a:+.3f}"


def diff_runs(baseline: ScoredBatch, candidate: ScoredBatch) -> ComparisonReport:
    base_by_stem = {r.stem: r for r in baseline.records}
    cand_by_stem = {r.stem: r for r in candidate.records}
    common = set(base_by_stem) & set(cand_by_stem)

    regressions: list[RegressionRow] = []
    improvements: list[ImprovementRow] = []
    for stem in sorted(common):
        b = base_by_stem[stem]
        c = cand_by_stem[stem]
        # Skip cases where one side errored: error transitions are surfaced via
        # the Errors section of each run's report, not as regressions/improvements.
        # Tracking pass<>error transitions explicitly is a future enhancement.
        if b.actual is None or c.actual is None:
            continue
        b_pass = _passes(b)
        c_pass = _passes(c)
        row_kwargs = dict(
            stem=stem,
            baseline_decision=b.actual.decision,
            candidate_decision=c.actual.decision,
            baseline_score=b.actual.score,
            candidate_score=c.actual.score,
        )
        if b_pass and not c_pass:
            regressions.append(RegressionRow(**row_kwargs))
        elif not b_pass and c_pass:
            improvements.append(ImprovementRow(**row_kwargs))

    ba = baseline.aggregates
    ca = candidate.aggregates
    delta_table = [
        DeltaRow(metric="decision_accuracy",
                 baseline=f"{ba.decision_correct_count}/{ba.decision_total}",
                 candidate=f"{ca.decision_correct_count}/{ca.decision_total}",
                 delta=_fmt_pp_delta(ba.decision_accuracy, ca.decision_accuracy)),
        DeltaRow(metric="score_mae",
                 baseline=f"{ba.score_mae:.1f}" if ba.score_mae is not None else "n/a",
                 candidate=f"{ca.score_mae:.1f}" if ca.score_mae is not None else "n/a",
                 delta=(f"{ca.score_mae - ba.score_mae:+.1f}"
                        if ba.score_mae is not None and ca.score_mae is not None else "n/a")),
        DeltaRow(metric="total_usd",
                 baseline=_fmt_usd(ba.total_usd),
                 candidate=_fmt_usd(ca.total_usd),
                 delta=_fmt_usd_delta(ba.total_usd, ca.total_usd)),
    ]

    added = sorted(set(cand_by_stem) - set(base_by_stem))
    removed = sorted(set(base_by_stem) - set(cand_by_stem))

    return ComparisonReport(
        delta_table=delta_table,
        regressions=regressions,
        improvements=improvements,
        added_cases=added,
        removed_cases=removed,
    )


def load_scored_batch(jsonl_path: Path) -> ScoredBatch:
    """Reconstruct a ScoredBatch from `results.jsonl` + sibling `meta.yaml`."""
    jsonl_path = Path(jsonl_path)
    meta_path = jsonl_path.parent / "meta.yaml"
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    config = BatchConfig(
        run_id=meta.get("run_id", jsonl_path.parent.name),
        ts=meta.get("ts", ""),
        model=meta.get("model", "unknown"),
        prompt_version=meta.get("prompt_version", "unknown"),
        settings=meta.get("settings", {}) or {},
        git_sha=meta.get("git_sha"),
    )
    records: list[EvalRecord] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(EvalRecord.model_validate(json.loads(line)))
    return ScoredBatch(config=config, records=records, aggregates=aggregate(records))


def render_comparison_section(
    baseline: ScoredBatch, candidate: ScoredBatch, cmp: ComparisonReport
) -> str:
    """Markdown block injected into report.md when --baseline is set."""
    bcfg = baseline.config
    lines = [
        f"## Delta vs baseline ({bcfg.ts}, model=`{bcfg.model}`, prompt=`{bcfg.prompt_version}`)",
        "",
        "| metric | baseline | candidate | Δ |",
        "| --- | --- | --- | --- |",
    ]
    for row in cmp.delta_table:
        lines.append(f"| {row.metric} | {row.baseline} | {row.candidate} | {row.delta} |")
    lines.append("")
    lines.append(f"## Regressions ({len(cmp.regressions)})  *(passed in baseline, fail in candidate)*")
    for r in cmp.regressions:
        lines.append(f"### {r.stem}")
        lines.append(f"- baseline: decision={r.baseline_decision}, score={r.baseline_score} ✓")
        lines.append(f"- candidate: decision={r.candidate_decision}, score={r.candidate_score} ✗")
        lines.append(
            f"- diff: decision {r.baseline_decision}→{r.candidate_decision}, "
            f"score {r.baseline_score}→{r.candidate_score}"
        )
        lines.append("")
    if cmp.improvements:
        lines.append(f"## Improvements ({len(cmp.improvements)})")
        for r in cmp.improvements:
            lines.append(f"### {r.stem}")
            lines.append(
                f"- baseline ✗ → candidate ✓ "
                f"({r.baseline_decision}/{r.baseline_score} → {r.candidate_decision}/{r.candidate_score})"
            )
            lines.append("")
    if cmp.added_cases:
        lines.append("## Added cases")
        lines.extend(f"- {s}" for s in cmp.added_cases)
        lines.append("")
    if cmp.removed_cases:
        lines.append("## Removed cases")
        lines.extend(f"- {s}" for s in cmp.removed_cases)
        lines.append("")
    return "\n".join(lines)
