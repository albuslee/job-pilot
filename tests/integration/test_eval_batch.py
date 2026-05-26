from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from jobpilot.cli import app
from jobpilot.models.schemas import EvaluationResult


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def eval_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Lay out a self-contained evals/ directory + minimal Chroma/profile state."""
    monkeypatch.chdir(tmp_path)
    _write(tmp_path / "evals/jobs/a.txt", "JD A — RAG and LangGraph")
    _write(tmp_path / "evals/jobs/b.txt", "JD B — Java backend")
    _write(
        tmp_path / "evals/labels/a.yaml",
        "expected_decision: apply\nexpected_score_band: [70, 90]\n",
    )
    _write(
        tmp_path / "evals/labels/b.yaml",
        "expected_decision: apply\nexpected_score_band: [70, 90]\n",
    )
    _write(tmp_path / "evals/model_prices.yaml", '"test-model":\n  input: 0.001\n  output: 0.005\n')
    return tmp_path


def _patch_pipeline(
    monkeypatch: pytest.MonkeyPatch, responses: dict[str, EvaluationResult]
) -> None:
    """Replace build_graph + LLMClient + RagStore with stubs."""
    from jobpilot import cli
    from jobpilot.models.schemas import ProfileChunk
    from jobpilot.models.state import AgentState

    chunk = ProfileChunk(
        id="c1", source="cv.docx", heading_path=["Work"], text="Built RAG pipeline using LangGraph"
    )

    class _StubStore:
        def query(self, text: str, *, k: int) -> list[ProfileChunk]:
            return [chunk]

    class _StubGraph:
        async def ainvoke(self, state: AgentState) -> AgentState:
            stem = state["job"].source.replace(".txt", "")
            return AgentState(
                job=state["job"],
                retrieved=[chunk],
                evaluation=responses[stem],
                tailored=None,
                output_paths={},
            )

    def _build_graph_stub(*args: Any, **kwargs: Any) -> _StubGraph:
        return _StubGraph()

    monkeypatch.setattr(cli, "build_graph", _build_graph_stub)
    monkeypatch.setattr(cli, "_build_store", lambda: _StubStore())

    # LLMClient needs record() to work as a context manager.
    real_record = MagicMock()
    real_record.__enter__ = MagicMock(return_value=[])
    real_record.__exit__ = MagicMock(return_value=None)
    llm_stub = MagicMock()
    llm_stub.record.return_value = real_record
    monkeypatch.setattr(cli, "LLMClient", lambda *a, **kw: llm_stub)


def test_eval_batch_writes_jsonl_and_report(
    eval_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_pipeline(
        monkeypatch,
        {
            "a": EvaluationResult(
                score=80, decision="apply", reasoning="r", cited_chunk_ids=["c1"], risk_flags=[]
            ),
            "b": EvaluationResult(
                score=60, decision="maybe", reasoning="r", cited_chunk_ids=["c1"], risk_flags=[]
            ),
        },
    )
    monkeypatch.setenv("LITELLM_BASE_URL", "http://x")
    monkeypatch.setenv("LITELLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "test-model")

    runner = CliRunner()
    result = runner.invoke(app, ["eval-batch"])
    assert result.exit_code == 0, result.output

    run_dirs = list((eval_workspace / "evals/runs").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    rows = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {r["stem"] for r in rows} == {"a", "b"}
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "decision_accuracy" in report
    assert "1/2" in report  # one correct
    meta = (run_dir / "meta.yaml").read_text(encoding="utf-8")
    assert "model: test-model" in meta


def test_eval_batch_fail_under_returns_nonzero(
    eval_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_pipeline(
        monkeypatch,
        {
            "a": EvaluationResult(
                score=80, decision="apply", reasoning="r", cited_chunk_ids=["c1"], risk_flags=[]
            ),
            "b": EvaluationResult(
                score=60, decision="maybe", reasoning="r", cited_chunk_ids=["c1"], risk_flags=[]
            ),
        },
    )
    monkeypatch.setenv("LITELLM_BASE_URL", "http://x")
    monkeypatch.setenv("LITELLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "test-model")

    runner = CliRunner()
    bad = runner.invoke(app, ["eval-batch", "--fail-under", "1.0"])
    good = runner.invoke(app, ["eval-batch", "--fail-under", "0.0"])
    assert bad.exit_code == 1
    assert good.exit_code == 0


def test_eval_batch_with_baseline(eval_workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_pipeline(
        monkeypatch,
        {
            "a": EvaluationResult(
                score=80, decision="apply", reasoning="r", cited_chunk_ids=["c1"], risk_flags=[]
            ),
            "b": EvaluationResult(
                score=80, decision="apply", reasoning="r", cited_chunk_ids=["c1"], risk_flags=[]
            ),
        },
    )
    monkeypatch.setenv("LITELLM_BASE_URL", "http://x")
    monkeypatch.setenv("LITELLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "test-model")

    runner = CliRunner()
    first = runner.invoke(app, ["eval-batch"])
    assert first.exit_code == 0
    baseline_jsonl = next((eval_workspace / "evals/runs").iterdir()) / "results.jsonl"

    # Second run: candidate where 'b' regresses.
    _patch_pipeline(
        monkeypatch,
        {
            "a": EvaluationResult(
                score=80, decision="apply", reasoning="r", cited_chunk_ids=["c1"], risk_flags=[]
            ),
            "b": EvaluationResult(
                score=60, decision="maybe", reasoning="r", cited_chunk_ids=["c1"], risk_flags=[]
            ),
        },
    )
    second = runner.invoke(app, ["eval-batch", "--baseline", str(baseline_jsonl)])
    assert second.exit_code == 0
    # Find the *new* run dir
    new_dirs = sorted((eval_workspace / "evals/runs").iterdir())
    report = (new_dirs[-1] / "report.md").read_text(encoding="utf-8")
    assert "Delta vs baseline" in report
    assert "Regressions" in report
