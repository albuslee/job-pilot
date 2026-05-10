from __future__ import annotations

import pytest

from jobpilot.config import Settings


def test_settings_load_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "oa-test")
    monkeypatch.setenv("SCORE_THRESHOLD", "65")
    s = Settings()
    assert s.anthropic_api_key == "sk-test"
    assert s.embedding_provider == "openai"
    assert s.score_threshold == 65


def test_settings_voyage_default_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("VOYAGE_API_KEY", "vy-test")
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    s = Settings()
    assert s.embedding_provider == "voyage"
    assert s.voyage_api_key == "vy-test"


def test_settings_missing_anthropic_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError):
        Settings(_env_file=None)  # type: ignore[call-arg]
