from __future__ import annotations

import pytest
from pydantic import ValidationError

from jobpilot.models.schemas import (
    EvaluationResult,
    ExperienceBlock,
    JobDescription,
    ProfileChunk,
    TailoredCV,
)


def test_job_description_minimal() -> None:
    jd = JobDescription(source="canva.txt", body="Senior fullstack at Canva.")
    assert jd.company is None
    assert jd.title is None
    assert jd.body.startswith("Senior")


def test_profile_chunk_requires_text_and_source() -> None:
    chunk = ProfileChunk(
        id="cv-001",
        source="Albus_Li_CV.docx",
        heading_path=["Experience", "Trend Micro"],
        text="Built a multi-agent eval pipeline.",
    )
    assert chunk.id == "cv-001"
    assert chunk.heading_path == ["Experience", "Trend Micro"]


def test_evaluation_result_decision_enum() -> None:
    eval_ = EvaluationResult(
        score=82,
        decision="apply",
        reasoning="Strong match on backend + AI agents.",
        cited_chunk_ids=["cv-001", "cv-004"],
        risk_flags=[],
    )
    assert eval_.decision == "apply"
    assert eval_.score == 82


def test_evaluation_result_score_bounds() -> None:
    with pytest.raises(ValidationError):
        EvaluationResult(
            score=150,
            decision="apply",
            reasoning="x",
            cited_chunk_ids=[],
            risk_flags=[],
        )


def test_evaluation_result_decision_rejects_unknown() -> None:
    with pytest.raises(ValidationError):
        EvaluationResult(
            score=50,
            decision="ponder",  # type: ignore[arg-type]
            reasoning="x",
            cited_chunk_ids=[],
            risk_flags=[],
        )


def test_experience_block_round_trip() -> None:
    block = ExperienceBlock(
        company="Trend Micro",
        title="Senior SWE",
        dates="2022 — present",
        bullets=["Led the JobPilot prototype."],
    )
    assert block.bullets[0].startswith("Led")


def test_tailored_cv_stub_constructs() -> None:
    cv = TailoredCV(
        header="Albus Li — Senior Fullstack Engineer",
        summary="Six years building distributed systems.",
        experience=[],
        skills=["Python", "TypeScript"],
        education=["BSc CS"],
    )
    assert cv.skills == ["Python", "TypeScript"]
