"""Mic capture + speech pitch extraction. All heavy deps imported lazily.

record_audio: push-to-talk mic capture -> temp WAV + duration.
extract_pitch: parselmouth f0 over a WAV -> voiced f0 samples (Hz). Never raises;
returns [] on any failure so the caller degrades gracefully.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from jobpilot.logging_setup import get_logger

log = get_logger(__name__)

_SAMPLE_RATE = 16_000  # 16 kHz mono — what Whisper expects
_PITCH_FLOOR_HZ = 75.0
_PITCH_CEILING_HZ = 500.0


def record_audio() -> tuple[Path, float]:
    """Record from the default mic until the user presses Enter. Returns (wav_path, seconds)."""
    import threading

    import numpy as np  # lazy (pulled in by sounddevice)
    import sounddevice as sd  # lazy
    import soundfile as sf  # lazy

    print("Recording... press Enter to stop.")
    frames: list = []
    stream = sd.InputStream(samplerate=_SAMPLE_RATE, channels=1, dtype="float32")
    with stream:
        stop = threading.Event()
        threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()
        while not stop.is_set():
            block, _ = stream.read(_SAMPLE_RATE // 10)
            frames.append(block)

    audio = np.concatenate(frames, axis=0) if frames else np.zeros((0, 1), dtype="float32")
    duration_s = len(audio) / _SAMPLE_RATE
    wav_path = Path(tempfile.mkdtemp(prefix="jobpilot_interview_")) / "answer.wav"
    sf.write(str(wav_path), audio, _SAMPLE_RATE)
    return wav_path, duration_s


def extract_pitch(
    wav_path: Path,
    *,
    floor_hz: float = _PITCH_FLOOR_HZ,
    ceiling_hz: float = _PITCH_CEILING_HZ,
) -> list[float]:
    """Voiced f0 samples (Hz) via parselmouth/Praat. Returns [] on any failure."""
    try:
        import parselmouth  # lazy

        snd = parselmouth.Sound(str(wav_path))
        pitch = snd.to_pitch(pitch_floor=floor_hz, pitch_ceiling=ceiling_hz)
        values = pitch.selected_array["frequency"]
        return [float(f) for f in values if f > 0.0]
    except Exception as exc:  # missing dep, bad/empty wav, etc.
        log.warning("pitch.extract_failed", error=str(exc))
        return []
