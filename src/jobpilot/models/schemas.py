"""Pydantic schemas for JobPilot. Day 1 covers JD, profile chunks, evaluator output;
TailoredCV is included as a stub so Day 2's tailor can drop in without churn."""

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
    """Day 2 will populate this. Defined now so AgentState's type is stable."""

    header: str
    summary: str
    experience: list[ExperienceBlock] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
