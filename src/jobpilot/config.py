"""Application settings, sourced from .env / environment."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EmbeddingProvider = Literal["voyage", "openai"]
LogFormat = Literal["console", "json"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    anthropic_api_key: str = Field(..., min_length=1)
    anthropic_model: str = "claude-sonnet-4-6"

    embedding_provider: EmbeddingProvider = "voyage"
    voyage_api_key: str | None = None
    voyage_model: str = "voyage-3"
    openai_api_key: str | None = None
    openai_embedding_model: str = "text-embedding-3-small"

    profile_dir: Path = Path("data/profile")
    chroma_dir: Path = Path(".chroma")
    output_dir: Path = Path("output")

    score_threshold: int = 70
    retrieval_k: int = 8
    log_format: LogFormat = "console"


def get_settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue]
