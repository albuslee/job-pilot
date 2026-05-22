"""Load JD + label pairs into typed EvalCase objects for the harness."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from jobpilot.logging_setup import get_logger
from jobpilot.models.schemas import Decision, JobDescription

log = get_logger(__name__)


class ExpectedOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    expected_decision: Decision
    expected_score_band: tuple[int, int] | None = None
    key_evidence_chunk_substrings: list[str] = Field(default_factory=list)
    required_risk_flags: list[str] = Field(default_factory=list)
    disallowed_risk_flags: list[str] = Field(default_factory=list)
    notes: str = ""

    @field_validator("expected_score_band", mode="before")
    @classmethod
    def _coerce_band(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return (int(v[0]), int(v[1]))
        raise ValueError("expected_score_band must be a 2-element [min, max] list of ints")


class EvalCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    stem: str
    jd: JobDescription
    expected: ExpectedOutcome


def load_eval_set(*, jobs_glob: str, labels_dir: Path) -> list[EvalCase]:
    """Load every JD in `jobs_glob` paired with `labels_dir/<stem>.yaml`.

    JDs without a matching label file are skipped with a warning. Bad YAML or
    invalid schema raises ValueError mentioning the offending file.
    """
    cases: list[EvalCase] = []
    for jd_path_str in sorted(glob.glob(jobs_glob)):
        jd_path = Path(jd_path_str)
        stem = jd_path.stem
        label_path = labels_dir / f"{stem}.yaml"
        if not label_path.exists():
            log.warning("eval.label_missing", stem=stem, label_path=str(label_path))
            continue
        try:
            data = yaml.safe_load(label_path.read_text(encoding="utf-8")) or {}
            expected = ExpectedOutcome.model_validate(data)
        except Exception as exc:
            raise ValueError(f"invalid label file {label_path.name}: {exc}") from exc
        jd = JobDescription(source=jd_path.name, body=jd_path.read_text(encoding="utf-8"))
        cases.append(EvalCase(stem=stem, jd=jd, expected=expected))
    return cases
