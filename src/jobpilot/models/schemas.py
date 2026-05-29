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


class DocxSectionAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    block_index: int = Field(..., ge=0)
    heading_path: list[str] = Field(default_factory=list)
    is_heading: bool = False
    level: int = Field(default=0, ge=0)


class DocxSectionNormalization(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assignments: list[DocxSectionAssignment] = Field(default_factory=list)


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


# ---- Interview practice (Coach mode) ----------------------------------------


class InterviewQuestion(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    category: str
    text: str = Field(..., min_length=1)
    guidance: str | None = None


class Word(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    start: float
    end: float


class Transcript(BaseModel):
    text: str
    words: list[Word] = Field(default_factory=list)
    duration_s: float | None = None


class CapturedAnswer(BaseModel):
    transcript: str
    words: list[Word] = Field(default_factory=list)
    duration_s: float | None = None
    pitch_hz: list[float] = Field(default_factory=list)
    source: Literal["audio", "text"]


class FillerStat(BaseModel):
    word: str
    count: int


class DeliveryMetrics(BaseModel):
    word_count: int
    duration_s: float | None
    wpm: float | None
    filler_count: int
    top_fillers: list[FillerStat] = Field(default_factory=list)
    long_pause_count: int
    longest_pause_s: float | None
    mean_pitch_hz: float | None
    pitch_range_hz: float | None
    pitch_std_semitones: float | None
    monotone: bool | None


class AnswerFeedback(BaseModel):
    answered_question: bool
    structure_notes: str = Field(..., min_length=1)
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    missed_experiences: list[str] = Field(default_factory=list)
    cited_chunk_ids: list[str] = Field(default_factory=list)


class FollowUp(BaseModel):
    should_continue: bool
    question: str | None = None


class InterviewTurn(BaseModel):
    question: InterviewQuestion
    transcript: str
    metrics: DeliveryMetrics
    feedback: AnswerFeedback | None = None


class SessionSummary(BaseModel):
    overall: str = Field(..., min_length=1)
    top_strengths: list[str] = Field(default_factory=list)
    top_improvements: list[str] = Field(default_factory=list)
