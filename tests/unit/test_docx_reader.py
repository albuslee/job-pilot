from __future__ import annotations

from pathlib import Path

from docx import Document

from jobpilot.tools.docx_reader import DocxBlock, read_docx


def _build_fixture(path: Path) -> None:
    doc = Document()
    doc.add_heading("Experience", level=1)
    doc.add_heading("ACME Corp", level=2)
    doc.add_paragraph("Built RAG prototype.")
    doc.add_paragraph("Led cloud networking migration.")
    doc.add_heading("Education", level=1)
    doc.add_paragraph("BSc Computer Science.")
    doc.save(path)


def test_read_docx_yields_blocks_with_heading_path(tmp_path: Path) -> None:
    fixture = tmp_path / "cv.docx"
    _build_fixture(fixture)

    blocks = read_docx(fixture)

    assert all(isinstance(b, DocxBlock) for b in blocks)
    paragraphs = [b for b in blocks if not b.is_heading]
    assert any(b.text == "Built RAG prototype." for b in paragraphs)
    bullet = next(b for b in paragraphs if b.text == "Built RAG prototype.")
    assert bullet.heading_path == ["Experience", "ACME Corp"]
    edu = next(b for b in paragraphs if b.text == "BSc Computer Science.")
    assert edu.heading_path == ["Education"]
