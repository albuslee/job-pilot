"""JobPilot CLI. Day 2 commands: version, ingest, eval, run."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import typer

from jobpilot import __version__
from jobpilot.agents.evaluator import EvaluatorAgent
from jobpilot.agents.orchestrator import build_graph
from jobpilot.agents.tailor import TailorAgent
from jobpilot.config import get_settings
from jobpilot.llm.client import LLMClient
from jobpilot.logging_setup import configure_logging, get_logger
from jobpilot.models.schemas import JobDescription
from jobpilot.models.state import AgentState
from jobpilot.rag.embeddings import build_embedder
from jobpilot.rag.ingest import ingest_profile_dir
from jobpilot.rag.store import RagStore
from jobpilot.tools.bullet_pool import load_bullet_pool
from jobpilot.tools.docx_writer import render_cv

# Backwards-compat alias kept so existing tests that patch `jobpilot.cli.AnthropicClient` still work.
AnthropicClient = LLMClient

app = typer.Typer(
    help="JobPilot — RAG-grounded job evaluation + CV tailoring.", no_args_is_help=True
)
log = get_logger(__name__)

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _build_store() -> RagStore:
    settings = get_settings()
    embedder = build_embedder(settings=settings)
    return RagStore(persist_dir=settings.chroma_dir, embedder=embedder)


def _slug(value: str, *, default: str = "Unknown") -> str:
    """Slugify a string to a filesystem-safe component."""
    return _SAFE.sub("_", value).strip("_") or default


def _output_filename(owner_name: str, company: str) -> str:
    """Compose `<owner_slug>_CV_<company_slug>.docx` from settings + tailored target."""
    return f"{_slug(owner_name, default='Owner')}_CV_{_slug(company)}.docx"


@app.command()
def version() -> None:
    """Print the JobPilot version."""
    typer.echo(f"jobpilot {__version__}")


@app.command()
def ingest(
    profile: Path | None = typer.Option(
        None, "--profile", help="Profile directory containing .docx files."
    ),
) -> None:
    """Ingest profile .docx files into the RAG store."""
    settings = get_settings()
    configure_logging(settings.log_format)
    profile_dir = profile or settings.profile_dir
    store = _build_store()
    n = ingest_profile_dir(profile_dir, store=store)
    typer.echo(f"Ingested {n} chunks from {profile_dir} into {settings.chroma_dir}.")


@app.command(name="eval")
def evaluate_cmd(
    jd_path: Path = typer.Argument(..., exists=True, readable=True),
) -> None:
    """Evaluate a single JD against the ingested profile (no tailoring)."""
    settings = get_settings()
    configure_logging(settings.log_format)
    jd = JobDescription(source=jd_path.name, body=jd_path.read_text(encoding="utf-8"))
    store = _build_store()
    llm = AnthropicClient(settings=settings)
    agent = EvaluatorAgent(settings=settings, llm=llm, rag=store)
    state: AgentState = {"job": jd}

    out = asyncio.run(agent.run(state))
    result = out["evaluation"]
    assert result is not None
    typer.echo(f"score={result.score} decision={result.decision}")
    typer.echo("reasoning:")
    typer.echo(result.reasoning)
    if result.cited_chunk_ids:
        typer.echo(f"cited_chunks: {', '.join(result.cited_chunk_ids)}")
    if result.risk_flags:
        typer.echo(f"risk_flags: {', '.join(result.risk_flags)}")


@app.command(name="run")
def run_cmd(
    jd_path: Path = typer.Argument(..., exists=True, readable=True),
) -> None:
    """Evaluate then (if score ≥ threshold) tailor a CV to .docx in output_dir."""
    settings = get_settings()
    configure_logging(settings.log_format)
    jd = JobDescription(source=jd_path.name, body=jd_path.read_text(encoding="utf-8"))

    pool = load_bullet_pool(settings.bullet_pool_path)
    store = _build_store()
    llm = LLMClient(settings=settings)
    evaluator = EvaluatorAgent(settings=settings, llm=llm, rag=store)
    tailor = TailorAgent(settings=settings, llm=llm, rag=store, pool=pool)
    graph = build_graph(settings=settings, evaluator=evaluator, tailor=tailor)

    state: AgentState = {"job": jd}
    out = asyncio.run(graph.ainvoke(state))

    evaluation = out["evaluation"]
    typer.echo(f"score={evaluation.score} decision={evaluation.decision}")
    typer.echo("reasoning:")
    typer.echo(evaluation.reasoning)

    # `tailored` is None whenever the orchestrator's conditional edge skipped the tailor node
    # (score < settings.score_threshold). See agents/orchestrator.py::decide.
    tailored = out.get("tailored")
    if tailored is None:
        typer.echo("(score below threshold; tailor skipped)")
        return

    resolved_bullets = pool.resolve_current_role(tailored.current_role_bullet_ids)
    output_path = settings.output_dir / _output_filename(
        settings.owner_name, tailored.target_company
    )
    render_cv(
        template_path=settings.cv_template_path,
        tailored=tailored,
        resolved_current_role_bullets=resolved_bullets,
        output_path=output_path,
        current_role_anchor=settings.current_role_anchor or None,
    )
    typer.echo(f"tailored CV → {output_path}")
    typer.echo(f"target: {tailored.target_company} / {tailored.target_role}")
    typer.echo(f"selected_bullets: {', '.join(tailored.current_role_bullet_ids)}")


# ---- eval-batch ----------------------------------------------------------------


@app.command(name="eval-batch")
def eval_batch_cmd(
    jobs_glob: str = typer.Argument("evals/jobs/*.txt"),
    labels_dir: Path = typer.Option(Path("evals/labels"), "--labels-dir"),
    prompt_version: str = typer.Option("v1", "--prompt-version"),
    model: str | None = typer.Option(None, "--model"),
    baseline: Path | None = typer.Option(None, "--baseline", exists=True, readable=True),
    run_dir: Path | None = typer.Option(None, "--run-dir"),
    with_tailor: bool = typer.Option(False, "--tailor/--no-tailor"),
    fail_under: float | None = typer.Option(None, "--fail-under"),
    prices_path: Path = typer.Option(Path("evals/model_prices.yaml"), "--prices"),
    concurrency: int = typer.Option(1, "--concurrency", min=1, max=8),
) -> None:
    """Run the evaluator (and optionally tailor) over a labelled fixture set.

    Writes evals/runs/<ts>/{results.jsonl, report.md, meta.yaml}. With --baseline,
    the report also includes a Delta + Regressions section.
    """
    _ = concurrency  # reserved for future use

    import subprocess
    from datetime import UTC, datetime

    from jobpilot.evals.compare import diff_runs, load_scored_batch
    from jobpilot.evals.fixtures import load_eval_set
    from jobpilot.evals.metrics import BatchConfig, ScoredBatch, aggregate
    from jobpilot.evals.pricing import load_prices
    from jobpilot.evals.report import (
        write_meta_yaml,
        write_report_md,
        write_results_jsonl,
    )
    from jobpilot.evals.runner import run_batch

    settings = get_settings()
    configure_logging(settings.log_format)
    if model:
        settings = settings.model_copy(update={"llm_model": model})

    cases = load_eval_set(jobs_glob=jobs_glob, labels_dir=labels_dir)
    if not cases:
        typer.echo("No labelled cases found.")
        raise typer.Exit(code=1)

    prices = load_prices(prices_path) if prices_path.exists() else None

    store = _build_store()
    llm = LLMClient(settings=settings)
    evaluator = EvaluatorAgent(settings=settings, llm=llm, rag=store, prompt_version=prompt_version)
    if with_tailor:
        pool = load_bullet_pool(settings.bullet_pool_path)
        tailor = TailorAgent(settings=settings, llm=llm, rag=store, pool=pool)
        graph = build_graph(settings=settings, evaluator=evaluator, tailor=tailor)
    else:
        graph = build_graph(settings=settings, evaluator=evaluator, tailor=None)

    records = asyncio.run(
        run_batch(
            cases=cases,
            graph=graph,  # type: ignore[arg-type]
            llm=llm,
            prompt_version=prompt_version,
            model=settings.llm_model,
            prices=prices,
        )
    )

    ts = datetime.now(UTC)
    run_id = ts.strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = run_dir or (Path("evals/runs") / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        git_sha: str | None = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        ).stdout.strip()
    except Exception:
        git_sha = None

    scored = ScoredBatch(
        config=BatchConfig(
            run_id=run_id,
            ts=ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            model=settings.llm_model,
            prompt_version=prompt_version,
            settings={
                "retrieval_k": settings.retrieval_k,
                "score_threshold": settings.score_threshold,
            },
            git_sha=git_sha,
        ),
        records=records,
        aggregates=aggregate(records),
    )

    comparison = None
    base_scored = None
    if baseline is not None:
        base_scored = load_scored_batch(baseline)
        comparison = diff_runs(base_scored, scored)

    write_results_jsonl(scored, out_dir / "results.jsonl")
    write_meta_yaml(scored, out_dir / "meta.yaml")
    write_report_md(scored, out_dir / "report.md", baseline=base_scored, comparison=comparison)

    # Terse stdout
    a = scored.aggregates
    score_mae_str = "None" if a.score_mae is None else f"{a.score_mae:.1f}"
    total_usd_str = f"${a.total_usd:.4f}" if a.total_usd is not None else "n/a"
    typer.echo(
        f"n_cases={a.n_cases}  errors={a.n_errors}  "
        f"decision_accuracy={a.decision_accuracy:.2f}  "
        f"score_mae={score_mae_str}  "
        f"total_usd={total_usd_str}"
    )
    typer.echo(f"report: {out_dir / 'report.md'}")

    if fail_under is not None and scored.aggregates.decision_accuracy < fail_under:
        raise typer.Exit(code=1)
