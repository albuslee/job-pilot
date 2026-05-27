"""Tests for the `jobpilot scrape-eval` CLI command."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from jobpilot.cli import app
from jobpilot.models.schemas import EvaluationResult
from jobpilot.tools.linkedin_jd import LinkedInJob

_FAKE_JOB = LinkedInJob(
    title="Senior AI Engineer",
    company="Acme Corp",
    about_job="Build LLM pipelines.",
    source_url="https://www.linkedin.com/jobs/view/9999999999",
)

_FAKE_EVAL = EvaluationResult(
    score=82,
    decision="apply",
    reasoning="Strong AI match.",
    cited_chunk_ids=["c1"],
    risk_flags=[],
)

_URL = "https://www.linkedin.com/jobs/view/9999999999"

runner = CliRunner()


@pytest.fixture()
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LITELLM_BASE_URL", "http://localhost:4000/v1")
    monkeypatch.setenv("LITELLM_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / ".chroma"))
    (tmp_path / "evals/jobs").mkdir(parents=True)
    (tmp_path / "evals/labels").mkdir(parents=True)
    return tmp_path


def _patch_deps(
    monkeypatch: pytest.MonkeyPatch, eval_result: EvaluationResult = _FAKE_EVAL
) -> None:
    monkeypatch.setattr(
        "jobpilot.cli.fetch_linkedin_job",
        lambda url, **kw: _FAKE_JOB,
    )

    async def _fake_run(self, state):
        state["evaluation"] = eval_result
        return state

    monkeypatch.setattr("jobpilot.agents.evaluator.EvaluatorAgent.run", _fake_run)

    class _StubStore:
        def query(self, text: str, k: int) -> list:
            return []

        def add(self, chunks: list) -> None:
            pass

    monkeypatch.setattr("jobpilot.cli._build_store", lambda: _StubStore())


def test_creates_jd_file(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_deps(monkeypatch)
    result = runner.invoke(app, ["scrape-eval", _URL])
    assert result.exit_code == 0, result.output
    jd_files = list((workspace / "evals/jobs").glob("*.txt"))
    assert len(jd_files) == 1
    assert "acme_corp" in jd_files[0].name.lower()


def test_creates_stub_label(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_deps(monkeypatch)
    runner.invoke(app, ["scrape-eval", _URL])
    label_files = list((workspace / "evals/labels").glob("*.yaml"))
    assert len(label_files) == 1
    data = yaml.safe_load(label_files[0].read_text(encoding="utf-8"))
    assert data["expected_decision"] == "apply"
    assert "notes" in data


def test_prints_score_and_decision(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_deps(monkeypatch)
    result = runner.invoke(app, ["scrape-eval", _URL])
    assert "82" in result.output
    assert "apply" in result.output.lower()
    assert "Strong AI match" in result.output


def test_duplicate_warns_and_exits(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_deps(monkeypatch)
    runner.invoke(app, ["scrape-eval", _URL])
    result = runner.invoke(app, ["scrape-eval", _URL])
    assert result.exit_code == 1
    assert "already exists" in result.output.lower()


def test_force_overwrites(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_deps(monkeypatch)
    runner.invoke(app, ["scrape-eval", _URL])
    result = runner.invoke(app, ["scrape-eval", "--force", _URL])
    assert result.exit_code == 0
    assert "already exists" not in result.output.lower()


def test_suggests_label_update_when_model_disagrees(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skip_eval = EvaluationResult(
        score=30,
        decision="skip",
        reasoning="Not a fit.",
        cited_chunk_ids=[],
        risk_flags=["no Python experience"],
    )
    _patch_deps(monkeypatch, eval_result=skip_eval)
    result = runner.invoke(app, ["scrape-eval", _URL])
    assert result.exit_code == 0
    assert "expected_decision: skip" in result.output


def test_no_suggestion_when_model_agrees(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_deps(monkeypatch)
    result = runner.invoke(app, ["scrape-eval", _URL])
    assert result.exit_code == 0
    assert "expected_decision" not in result.output
