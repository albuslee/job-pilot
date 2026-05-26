"""Write run artifacts: results.jsonl, report.md, meta.yaml.

Baseline-aware comparison sections are added in Task 9 (see compare.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from jobpilot.evals.metrics import EvalRecord, ScoredBatch

if TYPE_CHECKING:
    from jobpilot.evals.compare import ComparisonReport


def write_results_jsonl(scored: ScoredBatch, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [r.model_dump_json() for r in scored.records]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_meta_yaml(scored: ScoredBatch, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "run_id": scored.config.run_id,
        "ts": scored.config.ts,
        "model": scored.config.model,
        "prompt_version": scored.config.prompt_version,
        "n_cases": scored.aggregates.n_cases,
        "settings": dict(scored.config.settings),
    }
    if scored.config.git_sha:
        payload["git_sha"] = scored.config.git_sha
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


# ---- Markdown ---------------------------------------------------------------


def _fmt_pct(rate: float | None) -> str:
    return f"{rate * 100:.1f}%" if rate is not None else "n/a"


def _fmt_usd(amount: float | None) -> str:
    return f"${amount:.3f}" if amount is not None else "n/a"


def _is_failure(rec: EvalRecord) -> bool:
    if rec.error is not None or rec.metrics is None:
        return False
    m = rec.metrics
    checks = [
        m.decision_correct,
        m.score_in_band,
        m.citation_evidence_ok,
        m.required_risk_flags_present,
        m.disallowed_risk_flags_absent,
        m.cited_chunks_retrieved,
    ]
    return any(c is False for c in checks)


def _failure_lines(rec: EvalRecord) -> list[str]:
    assert rec.actual is not None and rec.metrics is not None
    out = [f"### {rec.stem}"]
    m = rec.metrics
    out.append(
        f"- expected_decision={rec.expected.expected_decision}, "
        f"got={rec.actual.decision} {'✓' if m.decision_correct else '✗'}"
    )
    if rec.expected.expected_score_band is not None:
        band = rec.expected.expected_score_band
        out.append(
            f"- expected_score_band=[{band[0]},{band[1]}], "
            f"got={rec.actual.score} {'✓' if m.score_in_band else '✗'}"
        )
    if m.citation_evidence_ok is False:
        out.append(f"- citation_evidence_ok=✗ (cited: {', '.join(rec.actual.cited_chunk_ids)})")
    if m.required_risk_flags_present is False:
        out.append(f"- required_risk_flags missing (got: {rec.actual.risk_flags})")
    if m.disallowed_risk_flags_absent is False:
        out.append(f"- disallowed_risk_flags present (got: {rec.actual.risk_flags})")
    if m.cited_chunks_retrieved is False:
        out.append(
            f"- cited_chunks_retrieved=✗ (cited {rec.actual.cited_chunk_ids} not in retrieved)"
        )
    excerpt = (rec.actual.reasoning or "").strip().split("\n", 1)[0][:160]
    out.append(f'- reasoning excerpt: "{excerpt}"')
    return out


def write_report_md(
    scored: ScoredBatch,
    out: Path,
    *,
    baseline: ScoredBatch | None = None,
    comparison: ComparisonReport | None = None,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    agg = scored.aggregates
    cfg = scored.config

    lines: list[str] = []
    lines.append(f"# JobPilot Eval — {cfg.ts}\n")
    lines.append(
        f"**Config:** model=`{cfg.model}`, prompt_version=`{cfg.prompt_version}`, "
        f"n_cases={agg.n_cases}\n"
    )

    if (baseline is None) != (comparison is None):
        raise ValueError("baseline and comparison must both be provided or both be None")

    if baseline is not None and comparison is not None:
        from jobpilot.evals.compare import render_comparison_section

        lines.append(render_comparison_section(baseline, scored, comparison))

    # Summary table
    lines.append("## Summary")
    rows = [
        (
            "decision_accuracy",
            f"{agg.decision_correct_count}/{agg.decision_total} ({_fmt_pct(agg.decision_accuracy)})",
        ),
        (
            "score_mae",
            f"{agg.score_mae:.1f} (n={agg.score_mae_n})" if agg.score_mae is not None else "n/a",
        ),
        (
            "score_in_band_rate",
            f"{agg.score_in_band_count}/{agg.score_band_total} ({_fmt_pct(agg.score_in_band_rate)})"
            if agg.score_band_total
            else "n/a",
        ),
        (
            "citation_evidence_pass_rate",
            f"{agg.citation_evidence_count}/{agg.citation_evidence_total} "
            f"({_fmt_pct(agg.citation_evidence_pass_rate)})"
            if agg.citation_evidence_total
            else "n/a",
        ),
        (
            "required_risk_flags_pass_rate",
            f"{agg.required_risk_flags_count}/{agg.required_risk_flags_total} "
            f"({_fmt_pct(agg.required_risk_flags_pass_rate)})"
            if agg.required_risk_flags_total
            else "n/a",
        ),
        (
            "disallowed_risk_flags_pass_rate",
            f"{agg.disallowed_risk_flags_count}/{agg.disallowed_risk_flags_total} "
            f"({_fmt_pct(agg.disallowed_risk_flags_pass_rate)})"
            if agg.disallowed_risk_flags_total
            else "n/a",
        ),
        (
            "p50 latency / p95",
            f"{agg.p50_latency_ms / 1000.0:.1f}s / {agg.p95_latency_ms / 1000.0:.1f}s",
        ),
        ("total tokens (in / out)", f"{agg.total_input_tokens:,} / {agg.total_output_tokens:,}"),
        ("total cost", _fmt_usd(agg.total_usd)),
        (
            "mean cost / known case",
            _fmt_usd(agg.mean_usd_per_known_case)
            if agg.mean_usd_per_known_case is not None
            else "n/a",
        ),
        ("errors", str(agg.n_errors)),
    ]
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    for k, v in rows:
        lines.append(f"| {k} | {v} |")
    lines.append("")

    # Failures, sorted by stem
    failures = sorted([r for r in scored.records if _is_failure(r)], key=lambda r: r.stem)
    lines.append(f"## Failures ({len(failures)})")
    if not failures:
        lines.append("*(none)*")
    for rec in failures:
        lines.extend(_failure_lines(rec))
        lines.append("")

    # Errors
    errors = sorted([r for r in scored.records if r.error is not None], key=lambda r: r.stem)
    lines.append(f"## Errors ({len(errors)})")
    if not errors:
        lines.append("*(none)*")
    else:
        for rec in errors:
            assert rec.error is not None
            lines.append(f"### {rec.stem}")
            lines.append(f"- {rec.error.type}: {rec.error.message}")
            lines.append("")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
