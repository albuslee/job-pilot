"""Application settings, sourced from .env / environment."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EmbeddingProvider = Literal["voyage", "openai", "ollama"]
LogFormat = Literal["console", "json"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # LLM — LiteLLM gateway (OpenAI-compatible)
    litellm_base_url: str = "http://localhost:4000/v1"
    litellm_api_key: str = Field(default="no-key")
    llm_model: str = "claude-sonnet-4-6"

    # Embeddings
    embedding_provider: EmbeddingProvider = "ollama"
    voyage_api_key: str | None = None
    voyage_model: str = "voyage-3"
    openai_api_key: str | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    ollama_base_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "nomic-embed-text"

    profile_dir: Path = Path("data/profile")
    chroma_dir: Path = Path(".chroma")
    output_dir: Path = Path("output")
    cv_template_path: Path = Path("data/cv_template.docx")
    bullet_pool_path: Path = Path("data/bullet_pool.yaml")

    # Personal identity — used only to compose output filenames and to locate
    # the candidate's current-role section in their CV template. Defaults are
    # placeholders; supply real values via .env.
    owner_name: str = "Owner"
    # Exact text in the CV template's current-role line. When empty, the tailor
    # skips role-bullet replacement (only summary + skills are rewritten).
    current_role_anchor: str = ""

    score_threshold: int = 70
    retrieval_k: int = 8
    smart_docx_ingest: bool = False
    log_format: LogFormat = "console"


def get_settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue]
