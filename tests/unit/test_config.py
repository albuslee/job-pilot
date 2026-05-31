from __future__ import annotations

import pytest

from jobpilot.config import Settings


def test_settings_load_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_BASE_URL", "http://localhost:4000/v1")
    monkeypatch.setenv("LITELLM_API_KEY", "my-token")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "oa-test")
    monkeypatch.setenv("SCORE_THRESHOLD", "65")
    s = Settings()
    assert s.litellm_api_key == "my-token"
    assert s.embedding_provider == "openai"
    assert s.score_threshold == 65


def test_settings_ollama_default_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    s = Settings()
    assert s.embedding_provider == "ollama"
    assert s.ollama_base_url == "http://localhost:11434"
    assert s.ollama_embedding_model == "nomic-embed-text"


def test_settings_voyage_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setenv("VOYAGE_API_KEY", "vy-test")
    s = Settings()
    assert s.embedding_provider == "voyage"
    assert s.voyage_api_key == "vy-test"


def test_settings_litellm_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.litellm_base_url == "http://localhost:4000/v1"
    assert s.litellm_api_key == "no-key"


def test_settings_smart_docx_ingest_default_and_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SMART_DOCX_INGEST", raising=False)
    assert Settings(_env_file=None).smart_docx_ingest is False  # type: ignore[call-arg]

    monkeypatch.setenv("SMART_DOCX_INGEST", "true")
    assert Settings(_env_file=None).smart_docx_ingest is True  # type: ignore[call-arg]
