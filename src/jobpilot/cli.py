"""JobPilot CLI. Day 1 commands: version, ingest, eval."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from jobpilot import __version__
from jobpilot.agents.evaluator import EvaluatorAgent
from jobpilot.config import get_settings
from jobpilot.llm.client import AnthropicClient
from jobpilot.logging_setup import configure_logging, get_logger
from jobpilot.models.schemas import JobDescription
from jobpilot.models.state import AgentState
from jobpilot.rag.embeddings import build_embedder
from jobpilot.rag.ingest import ingest_profile_dir
from jobpilot.rag.store import RagStore

app = typer.Typer(help="JobPilot — RAG-grounded job evaluation.", no_args_is_help=True)
log = get_logger(__name__)


def _build_store() -> RagStore:
    settings = get_settings()
    embedder = build_embedder(settings=settings)
    return RagStore(persist_dir=settings.chroma_dir, embedder=embedder)


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
    """Evaluate a single job description against the ingested profile."""
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
