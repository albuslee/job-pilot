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
    doc.add_heading("ACME Corp", level=2)
    doc.add_paragraph("Built a RAG prototype.")
    doc.add_paragraph("Led the cloud networking migration.")
    doc.add_heading("Skills", level=1)
    doc.add_paragraph("Python, TypeScript, AWS.")
    doc.save(path)


def test_chunk_blocks_groups_under_heading() -> None:
    blocks = [
        DocxBlock("Experience", ["Experience"], True, 1),
        DocxBlock("ACME Corp", ["Experience", "ACME Corp"], True, 2),
        DocxBlock("Built RAG prototype.", ["Experience", "ACME Corp"], False, 0),
        DocxBlock("Led migration.", ["Experience", "ACME Corp"], False, 0),
        DocxBlock("Skills", ["Skills"], True, 1),
        DocxBlock("Python.", ["Skills"], False, 0),
    ]
    chunks = chunk_blocks(blocks, source="cv.docx")
    headings = [c.heading_path for c in chunks]
    assert ["Experience", "ACME Corp"] in headings
    assert ["Skills"] in headings
    # Two body paragraphs under "ACME Corp" must be merged into the same chunk's text.
    tm = next(c for c in chunks if c.heading_path == ["Experience", "ACME Corp"])
    assert "Built RAG prototype." in tm.text
    assert "Led migration." in tm.text


def test_ingest_profile_dir_persists_chunks(tmp_path: Path) -> None:
    profile = tmp_path / "profile"
    profile.mkdir()
    _write_cv(profile / "cv.docx")

    store = RagStore(persist_dir=tmp_path / "chroma", embedder=_FakeEmbedder())
    n = ingest_profile_dir(profile, store=store)

    assert n >= 2  # at least Experience>ACME Corp and Skills
    hits = store.query("anything", k=10)
    assert any("RAG prototype" in h.text for h in hits)
