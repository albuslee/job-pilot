from __future__ import annotations

from pathlib import Path

from jobpilot.config import Settings


def test_interview_defaults(settings: Settings) -> None:
    assert settings.interview_questions_path == Path("data/interview_questions.yaml")
    assert settings.whisper_backend == "local"
    assert settings.whisper_model == "base.en"
    assert settings.interview_followups == 2
    assert settings.pause_threshold_s == 1.5
    assert settings.monotone_std_threshold_semitones == 1.5
