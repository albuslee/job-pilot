from __future__ import annotations

from jobpilot.interview.delivery import DEFAULT_FILLERS, compute_delivery_metrics
from jobpilot.models.schemas import Word


def _w(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end)


def test_wpm_from_duration() -> None:
    m = compute_delivery_metrics("one two three four", [], 2.0, [])
    assert m.word_count == 4
    assert m.wpm == 120.0


def test_wpm_none_without_duration() -> None:
    m = compute_delivery_metrics("one two", [], None, [])
    assert m.wpm is None


def test_filler_counting_and_top() -> None:
    text = "um so like I basically um you know finished it like right"
    m = compute_delivery_metrics(text, [], 10.0, [])
    assert m.filler_count == 7
    top = {f.word: f.count for f in m.top_fillers}
    assert top["um"] == 2
    assert top["like"] == 2
    assert "you know" in top


def test_long_pauses_from_word_timestamps() -> None:
    words = [_w("a", 0.0, 0.5), _w("b", 2.5, 3.0), _w("c", 3.1, 3.4)]
    m = compute_delivery_metrics("a b c", words, 3.4, [], pause_threshold_s=1.5)
    assert m.long_pause_count == 1  # 2.5 - 0.5 = 2.0s gap
    assert m.longest_pause_s == 2.0


def test_pauses_none_with_fewer_than_two_words() -> None:
    m = compute_delivery_metrics("a", [_w("a", 0.0, 0.5)], 0.5, [])
    assert m.long_pause_count == 0
    assert m.longest_pause_s is None


def test_pitch_present_not_monotone() -> None:
    pitch = [100.0, 150.0, 200.0, 120.0, 180.0]
    m = compute_delivery_metrics("hi there", [], 1.0, pitch)
    assert m.mean_pitch_hz is not None
    assert m.pitch_range_hz == 100.0
    assert m.pitch_std_semitones is not None and m.pitch_std_semitones > 1.5
    assert m.monotone is False


def test_pitch_flat_is_monotone() -> None:
    pitch = [120.0, 121.0, 119.5, 120.5, 120.0]
    m = compute_delivery_metrics("hi", [], 1.0, pitch)
    assert m.pitch_std_semitones is not None and m.pitch_std_semitones < 1.5
    assert m.monotone is True


def test_pitch_empty_yields_none() -> None:
    m = compute_delivery_metrics("hi", [], 1.0, [])
    assert m.mean_pitch_hz is None
    assert m.pitch_range_hz is None
    assert m.pitch_std_semitones is None
    assert m.monotone is None


def test_pitch_drops_non_positive_samples() -> None:
    m = compute_delivery_metrics("hi", [], 1.0, [0.0, -1.0, 120.0])
    assert m.mean_pitch_hz == 120.0
    assert m.pitch_std_semitones == 0.0  # single voiced sample
    assert m.monotone is True


def test_pitch_all_unvoiced_yields_none() -> None:
    m = compute_delivery_metrics("hi", [], 1.0, [0.0, 0.0])
    assert m.mean_pitch_hz is None
    assert m.monotone is None


def test_default_fillers_is_tuple() -> None:
    assert "you know" in DEFAULT_FILLERS


def test_custom_fillers_override() -> None:
    m = compute_delivery_metrics("yep yep yep", [], 3.0, [], fillers=("yep",))
    assert m.filler_count == 3
    assert m.top_fillers[0].word == "yep"


def test_overlapping_word_timestamps_yield_no_pauses() -> None:
    words = [_w("a", 0.0, 1.0), _w("b", 0.5, 1.5)]  # gap = 0.5 - 1.0 = -0.5 (overlap)
    m = compute_delivery_metrics("a b", words, 1.5, [])
    assert m.long_pause_count == 0
    assert m.longest_pause_s is None


def test_word_count_includes_digits() -> None:
    # Whisper often writes spoken numbers as digits; they should count toward WPM.
    m = compute_delivery_metrics("managed 3 teams", [], 2.0, [])
    assert m.word_count == 3


def test_word_count_keeps_contractions_intact() -> None:
    # Apostrophes must not split contractions ("don't" is one word, not two).
    m = compute_delivery_metrics("I don't know", [], 2.0, [])
    assert m.word_count == 3
