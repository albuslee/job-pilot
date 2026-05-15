from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from jobpilot.config import Settings
from jobpilot.rag.embeddings import build_embedder


def test_ollama_embedder_calls_sdk(settings: Settings) -> None:
    sdk = MagicMock()

    def _fake_create(model: str, input: list[str]) -> Any:
        return MagicMock(data=[MagicMock(embedding=[0.1, 0.2]) for _ in input])

    sdk.embeddings.create.side_effect = _fake_create
    embedder = build_embedder(settings=settings, ollama_sdk=sdk)
    out = embedder.embed(["a", "b"])
    assert out == [[0.1, 0.2], [0.1, 0.2]]
    sdk.embeddings.create.assert_called_once_with(
        model=settings.ollama_embedding_model, input=["a", "b"]
    )


def test_voyage_embedder_calls_sdk(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setenv("VOYAGE_API_KEY", "vy-test")
    s = Settings()  # type: ignore[call-arg]
    sdk = MagicMock()
    sdk.embed.return_value = MagicMock(embeddings=[[0.1, 0.2], [0.3, 0.4]])
    embedder = build_embedder(settings=s, voyage_sdk=sdk)
    out = embedder.embed(["a", "b"])
    assert out == [[0.1, 0.2], [0.3, 0.4]]
    sdk.embed.assert_called_once_with(["a", "b"], model=s.voyage_model, input_type="document")


def test_openai_embedder_calls_sdk(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "oa-test")
    s = Settings()  # type: ignore[call-arg]
    sdk = MagicMock()

    def _fake_create(model: str, input: list[str]) -> Any:
        return MagicMock(data=[MagicMock(embedding=[0.5, 0.6]) for _ in input])

    sdk.embeddings.create.side_effect = _fake_create
    embedder = build_embedder(settings=s, openai_sdk=sdk)
    out = embedder.embed(["x"])
    assert out == [[0.5, 0.6]]
