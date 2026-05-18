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
