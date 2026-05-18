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
        source="cv.docx",
        heading_path=["Experience", "ACME Corp"],
        text="Built a multi-agent eval pipeline.",
    )
    assert chunk.id == "cv-001"
    assert chunk.heading_path == ["Experience", "ACME Corp"]


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
        company="ACME Corp",
        title="Senior SWE",
        dates="2022 — present",
        bullets=["Led the prototype."],
    )
    assert block.bullets[0].startswith("Led")


def test_tailored_cv_minimal() -> None:
    cv = TailoredCV(
        target_company="Canva",
        target_role="Senior Fullstack Engineer",
        summary="Senior engineer with 7+ years building AWS serverless platforms.",
        skills_lines=[
            "Languages: TypeScript, Python",
            "AWS: Lambda, Step Functions, EventBridge",
        ],
        current_role_bullet_ids=["entry-impact", "entry-leadership"],
    )
    assert cv.target_company == "Canva"
    assert cv.skills_lines[0].startswith("Languages")
    assert cv.current_role_bullet_ids == ["entry-impact", "entry-leadership"]


def test_tailored_cv_requires_company_and_role() -> None:
    with pytest.raises(ValidationError):
        TailoredCV(
            target_company="",
            target_role="x",
            summary="x",
            skills_lines=["x"],
            current_role_bullet_ids=["a"],
        )


def test_tailored_cv_requires_skills_lines_nonempty() -> None:
    with pytest.raises(ValidationError):
        TailoredCV(
            target_company="x",
            target_role="x",
            summary="x",
            skills_lines=[],
            current_role_bullet_ids=["a"],
        )


def test_tailored_cv_requires_bullet_ids_nonempty() -> None:
    with pytest.raises(ValidationError):
        TailoredCV(
            target_company="x",
            target_role="x",
            summary="x",
            skills_lines=["x"],
            current_role_bullet_ids=[],
        )
