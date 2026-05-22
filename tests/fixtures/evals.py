"""Builders for eval-harness Pydantic objects, kept small + composable."""

from __future__ import annotations

from jobpilot.evals.fixtures import EvalCase, ExpectedOutcome
from jobpilot.models.schemas import Decision, JobDescription


def make_expected(
    *,
    decision: Decision = "apply",
    score_band: tuple[int, int] | None = None,
    key_evidence: list[str] | None = None,
    required_flags: list[str] | None = None,
    disallowed_flags: list[str] | None = None,
    notes: str = "",
) -> ExpectedOutcome:
    return ExpectedOutcome(
        expected_decision=decision,
        expected_score_band=score_band,
        key_evidence_chunk_substrings=key_evidence or [],
        required_risk_flags=required_flags or [],
        disallowed_risk_flags=disallowed_flags or [],
        notes=notes,
    )


def make_eval_case(
    stem: str = "case_1",
    *,
    jd_body: str = "Some JD",
    decision: Decision = "apply",
    score_band: tuple[int, int] | None = None,
    **expected_kwargs,
) -> EvalCase:
    return EvalCase(
        stem=stem,
        jd=JobDescription(source=f"{stem}.txt", body=jd_body),
        expected=make_expected(decision=decision, score_band=score_band, **expected_kwargs),
    )
