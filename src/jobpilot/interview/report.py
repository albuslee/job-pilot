"""Markdown session report writer. Mirrors evals/report.py (build a list of lines,
join, write)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from jobpilot.models.schemas import DeliveryMetrics, InterviewTurn, SessionSummary


def default_report_path(output_dir: Path) -> Path:
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
    return output_dir / f"interview_{ts}.md"


def _metrics_lines(m: DeliveryMetrics) -> list[str]:
    wpm = "n/a" if m.wpm is None else f"{m.wpm}"
    pitch = (
        "n/a (no voiced audio)"
        if m.pitch_std_semitones is None
        else (
            f"mean {m.mean_pitch_hz} Hz, range {m.pitch_range_hz} Hz, "
            f"{m.pitch_std_semitones} semitones "
            f"({'monotone' if m.monotone else 'varied'})"
        )
    )
    longest = "" if m.longest_pause_s is None else f" (longest {m.longest_pause_s}s)"
    return [
        f"- **Pace:** {wpm} wpm ({m.word_count} words)",
        f"- **Fillers:** {m.filler_count}",
        f"- **Long pauses:** {m.long_pause_count}{longest}",
        f"- **Pitch:** {pitch}",
    ]


def _bullets(label: str, items: list[str]) -> list[str]:
    if not items:
        return []
    return [f"**{label}:**", *[f"- {it}" for it in items], ""]


def _turn_lines(turn: InterviewTurn, idx: int) -> list[str]:
    out = [f"## Q{idx}: {turn.question.text}", ""]
    out.append(f"_Category: {turn.question.category}_")
    out.append("")
    out.append("**Your answer (transcribed):**")
    out.append("")
    out.append(f"> {turn.transcript}")
    out.append("")
    out.append("**Delivery:**")
    out.extend(_metrics_lines(turn.metrics))
    out.append("")
    fb = turn.feedback
    if fb is not None:
        out.append(f"**Answered the question:** {'yes' if fb.answered_question else 'no'}")
        out.append("")
        out.append(f"**Structure:** {fb.structure_notes}")
        out.append("")
        out.extend(_bullets("Strengths", fb.strengths))
        out.extend(_bullets("Improvements", fb.improvements))
        out.extend(_bullets("Unsupported claims (not in your CV)", fb.unsupported_claims))
        out.extend(_bullets("Missed real experiences", fb.missed_experiences))
        if fb.cited_chunk_ids:
            out.append(f"_Grounded in CV chunks: {', '.join(fb.cited_chunk_ids)}_")
            out.append("")
    return out


def write_interview_report(
    turns: list[InterviewTurn],
    out: Path,
    *,
    mode: str,
    summary: SessionSummary | None = None,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines: list[str] = [f"# Interview Practice — {ts}", "", f"**Mode:** {mode}", ""]
    for i, turn in enumerate(turns, 1):
        lines.extend(_turn_lines(turn, i))
    if summary is not None:
        lines.append("## Session Summary")
        lines.append("")
        lines.append(summary.overall)
        lines.append("")
        lines.extend(_bullets("Top strengths", summary.top_strengths))
        lines.extend(_bullets("Top improvements", summary.top_improvements))
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
