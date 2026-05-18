"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from jobpilot.config import Settings
from tests.fixtures.cv_template import build_minimal_template


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    monkeypatch.setenv("LITELLM_BASE_URL", "http://localhost:4000/v1")
    monkeypatch.setenv("LITELLM_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("PROFILE_DIR", str(tmp_path / "profile"))
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    yield Settings()  # type: ignore[call-arg]


@pytest.fixture(scope="session")
def cv_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthetic CV template that mirrors the real template's structure.

    The real template (data/cv_template.docx) is gitignored — tests build their
    own so they don't depend on the user's CV being present.
    """
    path = tmp_path_factory.mktemp("templates") / "cv_template.docx"
    return build_minimal_template(path)
