"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from jobpilot.config import Settings


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("VOYAGE_API_KEY", "vy-test")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setenv("PROFILE_DIR", str(tmp_path / "profile"))
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    yield Settings()  # type: ignore[call-arg]
