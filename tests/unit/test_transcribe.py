from __future__ import annotations

from dataclasses import dataclass

from jobpilot.tools.transcribe import _segments_to_transcript


@dataclass
class _FakeWord:
    word: str
    start: float
    end: float


@dataclass
class _FakeSegment:
    words: list[_FakeWord]


@dataclass
class _FakeInfo:
    duration: float


def test_segments_map_to_transcript_with_words() -> None:
    segments = [
        _FakeSegment(words=[_FakeWord(" Hello", 0.0, 0.4), _FakeWord(" world", 0.5, 0.9)]),
        _FakeSegment(words=[_FakeWord(" again", 1.5, 1.9)]),
    ]
    tr = _segments_to_transcript(segments, _FakeInfo(duration=2.0))
    assert tr.text == "Hello world again"
    assert tr.duration_s == 2.0
    assert len(tr.words) == 3
    assert tr.words[0].text == "Hello"
    assert tr.words[2].start == 1.5


def test_segments_without_word_timestamps() -> None:
    @dataclass
    class _SegNoWords:
        words: None
        text: str

    segments = [_SegNoWords(words=None, text=" Hi there ")]
    tr = _segments_to_transcript(segments, _FakeInfo(duration=1.0))
    assert tr.text == "Hi there"
    assert tr.words == []
    assert tr.duration_s == 1.0
