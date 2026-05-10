from __future__ import annotations

from pathlib import Path

from docx import Document

from jobpilot.rag.ingest import chunk_blocks, ingest_profile_dir
from jobpilot.rag.store import RagStore
from jobpilot.tools.docx_reader import DocxBlock


class _FakeEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 0.0] for t in texts]


def _write_cv(path: Path) -> None:
    doc = Document()
    doc.add_heading("Experience", level=1)
    doc.add_heading("Trend Micro", level=2)
    doc.add_paragraph("Built JobPilot prototype with RAG.")
    doc.add_paragraph("Led the GCP networking migration.")
    doc.add_heading("Skills", level=1)
    doc.add_paragraph("Python, TypeScript, AWS.")
    doc.save(path)


def test_chunk_blocks_groups_under_heading() -> None:
    blocks = [
        DocxBlock("Experience", ["Experience"], True, 1),
        DocxBlock("Trend Micro", ["Experience", "Trend Micro"], True, 2),
        DocxBlock("Built JobPilot prototype.", ["Experience", "Trend Micro"], False, 0),
        DocxBlock("Led migration.", ["Experience", "Trend Micro"], False, 0),
        DocxBlock("Skills", ["Skills"], True, 1),
        DocxBlock("Python.", ["Skills"], False, 0),
    ]
    chunks = chunk_blocks(blocks, source="cv.docx")
    headings = [c.heading_path for c in chunks]
    assert ["Experience", "Trend Micro"] in headings
    assert ["Skills"] in headings
    # Two body paragraphs under "Trend Micro" must be merged into the same chunk's text.
    tm = next(c for c in chunks if c.heading_path == ["Experience", "Trend Micro"])
    assert "Built JobPilot prototype." in tm.text
    assert "Led migration." in tm.text


def test_ingest_profile_dir_persists_chunks(tmp_path: Path) -> None:
    profile = tmp_path / "profile"
    profile.mkdir()
    _write_cv(profile / "cv.docx")

    store = RagStore(persist_dir=tmp_path / "chroma", embedder=_FakeEmbedder())
    n = ingest_profile_dir(profile, store=store)

    assert n >= 2  # at least Experience>Trend Micro and Skills
    hits = store.query("anything", k=10)
    assert any("JobPilot" in h.text for h in hits)
