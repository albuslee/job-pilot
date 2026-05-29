"""Pure, deterministic delivery metrics over a transcript + word timestamps + pitch.

No audio/IO here — pitch samples are extracted by tools/audio.py and passed in,
so this whole module is unit-testable with synthetic data.
"""

from __future__ import annotations

import math
import re
import statistics

from jobpilot.models.schemas import DeliveryMetrics, FillerStat, Word

DEFAULT_FILLERS: tuple[str, ...] = (
    "um",
    "uh",
    "like",
    "you know",
    "basically",
    "actually",
    "sort of",
    "kind of",
    "i mean",
    "right",
)

_WORD_RE = re.compile(r"[\w']+")
_MAX_TOP_FILLERS = 5


def _count_words(transcript: str) -> int:
    return len(_WORD_RE.findall(transcript.lower()))


def _count_fillers(
    transcript: str, fillers: tuple[str, ...]
) -> tuple[int, list[FillerStat]]:
    text = transcript.lower()
    stats: list[FillerStat] = []
    total = 0
    for phrase in fillers:
        n = len(re.findall(r"\b" + re.escape(phrase) + r"\b", text))
        if n:
            stats.append(FillerStat(word=phrase, count=n))
            total += n
    stats.sort(key=lambda s: (-s.count, s.word))
    return total, stats[:_MAX_TOP_FILLERS]


def _pause_metrics(words: list[Word], threshold_s: float) -> tuple[int, float | None]:
    if len(words) < 2:
        return 0, None
    gaps = [words[i + 1].start - words[i].end for i in range(len(words) - 1)]
    gaps = [g for g in gaps if g > 0]
    if not gaps:
        return 0, None
    long_count = sum(1 for g in gaps if g > threshold_s)
    return long_count, round(max(gaps), 2)


def _pitch_metrics(
    pitch_hz: list[float], monotone_threshold: float
) -> tuple[float | None, float | None, float | None, bool | None]:
    voiced = [f for f in pitch_hz if f > 0.0]
    if not voiced:
        return None, None, None, None
    mean_hz = statistics.mean(voiced)
    rng = max(voiced) - min(voiced)
    if mean_hz <= 0.0:  # defensive; voiced samples are >0 so unreachable, but guards log2 (intentionally uncovered)
        return round(mean_hz, 1), round(rng, 1), None, None
    semitones = [12.0 * math.log2(f / mean_hz) for f in voiced]
    std_st = statistics.pstdev(semitones)  # 0.0 for single/identical samples
    return round(mean_hz, 1), round(rng, 1), round(std_st, 2), std_st < monotone_threshold


def compute_delivery_metrics(
    transcript: str,
    words: list[Word],
    duration_s: float | None,
    pitch_hz: list[float],
    *,
    fillers: tuple[str, ...] = DEFAULT_FILLERS,
    pause_threshold_s: float = 1.5,
    monotone_std_threshold_semitones: float = 1.5,
) -> DeliveryMetrics:
    word_count = _count_words(transcript)
    wpm = (
        round(word_count / duration_s * 60.0, 1)
        if duration_s and duration_s > 0
        else None
    )
    filler_count, top_fillers = _count_fillers(transcript, fillers)
    long_pause_count, longest_pause_s = _pause_metrics(words, pause_threshold_s)
    mean_hz, range_hz, std_st, monotone = _pitch_metrics(
        pitch_hz, monotone_std_threshold_semitones
    )
    return DeliveryMetrics(
        word_count=word_count,
        duration_s=duration_s,
        wpm=wpm,
        filler_count=filler_count,
        top_fillers=top_fillers,
        long_pause_count=long_pause_count,
        longest_pause_s=longest_pause_s,
        mean_pitch_hz=mean_hz,
        pitch_range_hz=range_hz,
        pitch_std_semitones=std_st,
        monotone=monotone,
    )
