from __future__ import annotations

from pathlib import Path

from jobpilot.tools.audio import extract_pitch


def test_extract_pitch_returns_empty_on_missing_file(tmp_path: Path) -> None:
    # parselmouth will fail to load a non-existent wav; adapter must swallow it.
    assert extract_pitch(tmp_path / "does_not_exist.wav") == []


def test_extract_pitch_returns_empty_when_parselmouth_unavailable(monkeypatch, tmp_path) -> None:
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "parselmouth":
            raise ImportError("parselmouth not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    assert extract_pitch(tmp_path / "any.wav") == []
