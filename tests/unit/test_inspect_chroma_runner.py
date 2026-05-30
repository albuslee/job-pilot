from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_runner():
    path = Path(__file__).resolve().parents[2] / "runners" / "inspect_chroma.py"
    spec = importlib.util.spec_from_file_location("inspect_chroma_runner", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_format_chunk_prints_metadata_and_preview() -> None:
    runner = _load_runner()

    output = runner._format_chunk(
        1,
        "cv-abc123",
        "Built a RAG prototype with ChromaDB.",
        {"source": "cv.docx", "heading_path": "WORK EXPERIENCE"},
        preview_chars=12,
    )

    assert "[1] cv-abc123" in output
    assert "source: cv.docx" in output
    assert "heading: WORK EXPERIENCE" in output
    assert "Built a RAG\n..." in output
