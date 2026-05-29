"""Whisper STT adapter. Returns a Transcript{text, words, duration_s}.

Heavy deps (faster-whisper / openai) are imported lazily inside the backend
functions so importing this module — and unit-testing `_segments_to_transcript` —
needs neither installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jobpilot.config import Settings
from jobpilot.models.schemas import Transcript, Word


def _segments_to_transcript(segments: Any, info: Any) -> Transcript:
    """Map faster-whisper (segments, info) into a Transcript. Pure / testable."""
    words: list[Word] = []
    text_parts: list[str] = []
    for seg in segments:
        seg_words = getattr(seg, "words", None)
        if seg_words:
            for w in seg_words:
                token = w.word.strip()
                words.append(Word(text=token, start=float(w.start), end=float(w.end)))
                text_parts.append(token)
        else:
            text_parts.append(getattr(seg, "text", "").strip())
    duration = getattr(info, "duration", None)
    return Transcript(
        text=" ".join(p for p in text_parts if p),
        words=words,
        duration_s=float(duration) if duration is not None else None,
    )


def _transcribe_local(wav_path: Path, model_name: str) -> Transcript:
    from faster_whisper import WhisperModel  # lazy

    model = WhisperModel(model_name)
    segments, info = model.transcribe(
        str(wav_path),
        word_timestamps=True,
        condition_on_previous_text=False,
        vad_filter=False,
    )
    return _segments_to_transcript(list(segments), info)


def _transcribe_openai(wav_path: Path, settings: Settings) -> Transcript:
    from openai import OpenAI  # lazy

    client = OpenAI(base_url=settings.litellm_base_url, api_key=settings.litellm_api_key)
    with wav_path.open("rb") as fh:
        resp = client.audio.transcriptions.create(model=settings.whisper_model, file=fh)
    return Transcript(text=resp.text.strip(), words=[], duration_s=None)


def transcribe(wav_path: Path, *, settings: Settings) -> Transcript:
    if settings.whisper_backend == "openai":
        return _transcribe_openai(wav_path, settings)
    return _transcribe_local(wav_path, settings.whisper_model)
