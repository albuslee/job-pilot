"""Pydantic schemas for JobPilot: JD, profile chunks, evaluator output, and the
Day 2 TailoredCV (pool-based bullet IDs; no free-form bullet text by design)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Decision = Literal["apply", "maybe", "skip"]


class JobDescription(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str = Field(..., description="Origin file or URL of the JD.")
    body: str = Field(..., min_length=1)
    company: str | None = None
    title: str | None = None
    location: str | None = None


class ProfileChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    source: str
    heading_path: list[str] = Field(default_factory=list)
    text: str = Field(..., min_length=1)


class EvaluationResult(BaseModel):
    score: int = Field(..., ge=0, le=100)
    decision: Decision
    reasoning: str = Field(..., min_length=1)
    cited_chunk_ids: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)


class ExperienceBlock(BaseModel):
    company: str
    title: str
    dates: str
    bullets: list[str] = Field(default_factory=list)


class TailoredCV(BaseModel):
    """Day 2 tailor output. The schema deliberately exposes NO free-form bullet
    field — current-role bullets are selected by ID from a fixed pool, making
    bullet hallucination structurally impossible."""

    target_company: str = Field(..., min_length=1)
    target_role: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    skills_lines: list[str] = Field(..., min_length=1)
    current_role_bullet_ids: list[str] = Field(..., min_length=1)
