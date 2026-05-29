"""Interview question bank loader. Mirrors tools/bullet_pool.py: a committed
`.example.yaml` template plus a gitignored override; unknown shape or duplicate
IDs are a hard error."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from jobpilot.models.schemas import InterviewQuestion


@dataclass(frozen=True)
class QuestionBank:
    questions: list[InterviewQuestion]

    def by_category(self, category: str) -> list[InterviewQuestion]:
        return [q for q in self.questions if q.category == category]

    def pick(self, n: int, *, category: str | None = None) -> list[InterviewQuestion]:
        pool = self.by_category(category) if category else self.questions
        return pool[: max(n, 0)]


def load_question_bank(path: Path, *, example_path: Path) -> QuestionBank:
    src = path if path.is_file() else example_path
    if not src.is_file():
        raise FileNotFoundError(
            f"No question bank at {path} and no example at {example_path}."
        )
    raw = yaml.safe_load(src.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "questions" not in raw:
        raise ValueError(f"Question bank {src} must have a top-level `questions` list.")
    questions = [
        InterviewQuestion(
            id=item["id"],
            category=item["category"],
            text=item["text"],
            guidance=item.get("guidance"),
        )
        for item in raw["questions"]
    ]
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Question bank {src} contains duplicate IDs.")
    return QuestionBank(questions=questions)
