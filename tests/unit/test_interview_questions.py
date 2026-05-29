from __future__ import annotations

from pathlib import Path

import pytest

from jobpilot.interview.questions import load_question_bank

_YAML = """
questions:
  - id: why_leave
    category: motivational
    text: "Why do you want to leave your current role?"
    guidance: "Forward-looking; tie to growth."
  - id: next_role
    category: motivational
    text: "What are you looking for next?"
  - id: strength
    category: behavioral
    text: "Tell me about a strength."
"""


def _write(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_loads_override_when_present(tmp_path: Path) -> None:
    override = _write(tmp_path / "q.yaml", _YAML)
    example = tmp_path / "q.example.yaml"  # absent
    bank = load_question_bank(override, example_path=example)
    assert len(bank.questions) == 3


def test_falls_back_to_example(tmp_path: Path) -> None:
    override = tmp_path / "q.yaml"  # absent
    example = _write(tmp_path / "q.example.yaml", _YAML)
    bank = load_question_bank(override, example_path=example)
    assert bank.questions[0].id == "why_leave"


def test_category_filter_and_pick(tmp_path: Path) -> None:
    bank = load_question_bank(_write(tmp_path / "q.yaml", _YAML), example_path=tmp_path / "x")
    assert {q.id for q in bank.by_category("motivational")} == {"why_leave", "next_role"}
    picked = bank.pick(2, category="motivational")
    assert len(picked) == 2
    assert all(q.category == "motivational" for q in picked)
    assert len(bank.pick(99)) == 3


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    dupe = _YAML + '  - id: why_leave\n    category: x\n    text: "dup"\n'
    with pytest.raises(ValueError, match="duplicate"):
        load_question_bank(_write(tmp_path / "q.yaml", dupe), example_path=tmp_path / "x")


def test_missing_questions_key_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="questions"):
        load_question_bank(
            _write(tmp_path / "q.yaml", "foo: bar"), example_path=tmp_path / "x"
        )
