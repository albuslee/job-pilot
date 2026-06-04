from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from docx import Document
from typer.testing import CliRunner

from jobpilot.cli import app
from jobpilot.models.schemas import EvaluationResult, TailoredCV
from tests.fixtures.cv_template import ROLE_ANCHOR


def _fake_embedder() -> Any:
    class _Stub:
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[float(len(t)), 0.0] for t in texts]

    return _Stub()


@pytest.fixture
def pipeline_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cv_template: Path,
) -> dict[str, Path]:
    profile = tmp_path / "profile"
    profile.mkdir()
    output = tmp_path / "output"

    template = tmp_path / "cv_template.docx"
    shutil.copy(cv_template, template)
    # Also drop a copy into profile/ so `ingest` has something to chew on.
    shutil.copy(cv_template, profile / "cv.docx")

    pool = tmp_path / "bullet_pool.yaml"
    pool.write_text(
        "current_role:\n"
        "  - id: entry-a\n"
        "    text: Real bullet alpha.\n"
        "  - id: entry-b\n"
        "    text: Real bullet beta.\n"
    )

    jd = tmp_path / "canva.txt"
    jd.write_text("Senior fullstack at Canva. Python, TypeScript, RAG a plus.")

    monkeypatch.setenv("LITELLM_BASE_URL", "http://localhost:4000/v1")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-4-6")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setenv("VOYAGE_API_KEY", "vy-test")
    monkeypatch.setenv("PROFILE_DIR", str(profile))
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("OUTPUT_DIR", str(output))
    monkeypatch.setenv("CV_TEMPLATE_PATH", str(template))
    monkeypatch.setenv("BULLET_POOL_PATH", str(pool))
    monkeypatch.setenv("OWNER_NAME", "Test Owner")
    monkeypatch.setenv("CURRENT_ROLE_ANCHOR", ROLE_ANCHOR)
    return {"jd": jd, "output": output}


def test_run_writes_tailored_docx_when_score_above_threshold(
    pipeline_env: dict[str, Path],
) -> None:
    runner = CliRunner()

    eval_result = EvaluationResult(
        score=85,
        decision="apply",
        reasoning="Strong fullstack signals.",
        cited_chunk_ids=[],
        risk_flags=[],
    )
    tailor_result = TailoredCV(
        target_company="Canva",
        target_role="Senior Fullstack Engineer",
        summary="Seven years on AWS serverless fullstack.",
        skills_lines=[
            "Languages: TypeScript, Python",
            "AWS Serverless: Lambda, Step Functions",
        ],
        current_role_bullet_ids=["entry-b", "entry-a"],
    )

    mock_llm = MagicMock()
    # Each call to with_structured_output(schema) returns a chain whose .invoke() pops from queue.
    responses = [eval_result, tailor_result]
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = responses
    mock_llm.with_structured_output.return_value = mock_chain

    with (
        patch("jobpilot.cli.build_embedder", return_value=_fake_embedder()),
        patch("jobpilot.cli.build_llm", return_value=mock_llm),
    ):
        ingest = runner.invoke(app, ["ingest"])
        assert ingest.exit_code == 0, ingest.stdout

        result = runner.invoke(app, ["run", str(pipeline_env["jd"])])
        assert result.exit_code == 0, result.stdout
        assert "85" in result.stdout
        assert "Canva" in result.stdout

        out_path = pipeline_env["output"] / "Test_Owner_CV_Canva.docx"
        assert out_path.is_file()
        doc = Document(str(out_path))
        texts = [p.text.strip() for p in doc.paragraphs]
        overview = texts.index("CAREER OVERVIEW")
        assert "Seven years on AWS serverless" in texts[overview + 1]
        # Pool bullets are emitted in the requested order (entry-b, entry-a).
        role_idx = next(i for i, t in enumerate(texts) if ROLE_ANCHOR in t)
        tech_idx = next(i for i in range(role_idx + 1, len(texts)) if texts[i].startswith("Tech:"))
        between = "\n".join(texts[role_idx + 1 : tech_idx])
        assert "Real bullet beta." in between
        assert "Real bullet alpha." in between
        assert between.index("Real bullet beta.") < between.index("Real bullet alpha.")


def test_run_skips_tailoring_when_score_below_threshold(
    pipeline_env: dict[str, Path],
) -> None:
    runner = CliRunner()

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.return_value = EvaluationResult(
        score=35,
        decision="skip",
        reasoning="Mismatch on domain.",
        cited_chunk_ids=[],
        risk_flags=["no cardiology background"],
    )

    with (
        patch("jobpilot.cli.build_embedder", return_value=_fake_embedder()),
        patch("jobpilot.cli.build_llm", return_value=mock_llm),
    ):
        runner.invoke(app, ["ingest"])
        result = runner.invoke(app, ["run", str(pipeline_env["jd"])])
        assert result.exit_code == 0, result.stdout
        assert "skip" in result.stdout.lower()
        assert not any(pipeline_env["output"].glob("*.docx"))
