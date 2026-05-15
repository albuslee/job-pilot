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

    score_threshold: int = 70
    retrieval_k: int = 8
    log_format: LogFormat = "console"


def get_settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue]
