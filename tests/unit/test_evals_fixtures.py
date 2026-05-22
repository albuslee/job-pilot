from __future__ import annotations

from pathlib import Path

import pytest

from jobpilot.evals.fixtures import ExpectedOutcome, load_eval_set


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_load_eval_set_returns_cases_with_jds_and_labels(tmp_path: Path) -> None:
    _write(tmp_path / "jobs/a.txt", "JD A body")
    _write(tmp_path / "labels/a.yaml", "expected_decision: apply\nexpected_score_band: [70, 90]\n")

    cases = load_eval_set(jobs_glob=str(tmp_path / "jobs/*.txt"), labels_dir=tmp_path / "labels")

    assert len(cases) == 1
    case = cases[0]
    assert case.stem == "a"
    assert case.jd.body == "JD A body"
    assert case.jd.source == "a.txt"
    assert case.expected.expected_decision == "apply"
    assert case.expected.expected_score_band == (70, 90)


def test_load_eval_set_defaults_optional_fields(tmp_path: Path) -> None:
    _write(tmp_path / "jobs/a.txt", "JD")
    _write(tmp_path / "labels/a.yaml", "expected_decision: skip\n")

    cases = load_eval_set(jobs_glob=str(tmp_path / "jobs/*.txt"), labels_dir=tmp_path / "labels")

    e = cases[0].expected
    assert e.expected_decision == "skip"
    assert e.expected_score_band is None
    assert e.key_evidence_chunk_substrings == []
    assert e.required_risk_flags == []
    assert e.disallowed_risk_flags == []
    assert e.notes == ""


def test_load_eval_set_skips_jds_without_labels(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _write(tmp_path / "jobs/a.txt", "JD A")
    _write(tmp_path / "jobs/b.txt", "JD B")
    _write(tmp_path / "labels/a.yaml", "expected_decision: apply\n")
    # No labels/b.yaml — should warn and skip, not raise.

    cases = load_eval_set(jobs_glob=str(tmp_path / "jobs/*.txt"), labels_dir=tmp_path / "labels")

    assert [c.stem for c in cases] == ["a"]


def test_load_eval_set_rejects_invalid_yaml(tmp_path: Path) -> None:
    _write(tmp_path / "jobs/a.txt", "JD")
    _write(tmp_path / "labels/a.yaml", "expected_decision: not_a_decision\n")
    with pytest.raises(ValueError, match=r"a\.yaml"):
        load_eval_set(jobs_glob=str(tmp_path / "jobs/*.txt"), labels_dir=tmp_path / "labels")


def test_score_band_must_be_two_ints(tmp_path: Path) -> None:
    _write(tmp_path / "jobs/a.txt", "JD")
    _write(tmp_path / "labels/a.yaml", "expected_decision: apply\nexpected_score_band: [70]\n")
    with pytest.raises(ValueError, match="score_band"):
        load_eval_set(jobs_glob=str(tmp_path / "jobs/*.txt"), labels_dir=tmp_path / "labels")


def test_expected_outcome_model_directly() -> None:
    e = ExpectedOutcome(expected_decision="apply", expected_score_band=[80, 95])  # type: ignore[arg-type]
    assert e.expected_score_band == (80, 95)
    assert isinstance(e, ExpectedOutcome)


def test_load_eval_set_sorts_by_stem(tmp_path: Path) -> None:
    for stem in ("c", "a", "b"):
        _write(tmp_path / f"jobs/{stem}.txt", stem)
        _write(tmp_path / f"labels/{stem}.yaml", "expected_decision: apply\n")
    cases = load_eval_set(jobs_glob=str(tmp_path / "jobs/*.txt"), labels_dir=tmp_path / "labels")
    assert [c.stem for c in cases] == ["a", "b", "c"]
