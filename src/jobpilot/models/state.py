"""LangGraph-compatible AgentState (TypedDict). Day 2 reuses this verbatim."""

from __future__ import annotations

from typing import TypedDict

from jobpilot.models.schemas import (
    EvaluationResult,
    JobDescription,
    ProfileChunk,
    TailoredCV,
)


class AgentState(TypedDict, total=False):
    job: JobDescription
    retrieved: list[ProfileChunk]
    evaluation: EvaluationResult | None
    tailored: TailoredCV | None
    output_paths: dict[str, str]
