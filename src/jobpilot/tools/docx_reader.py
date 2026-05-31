"""Read a .docx into ordered blocks tagged with their heading path."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from docx import Document
from docx.document import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

_HEADING_RE = re.compile(r"^Heading\s+(\d+)$", re.IGNORECASE)
# Heuristic: short all-caps lines with no sentence punctuation are section headers
_HEURISTIC_HEADING_RE = re.compile(r"^[A-Z][A-Z\s&/\-]{2,39}$")
_COMMON_SECTION_NAMES = {
    "career overview",
    "core capabilities",
    "certifications",
    "education",
    "employment history",
    "experience",
    "key achievements",
    "professional experience",
    "professional summary",
    "projects",
    "qualifications",
    "selected projects",
    "skills",
    "skills and expertise",
    "summary",
    "technical skills",
    "work experience",
}

BlockSource = Literal["paragraph", "table_cell"]


@dataclass(frozen=True)
class DocxBlock:
    text: str
    heading_path: list[str]
    is_heading: bool
    level: int  # 0 for body paragraphs


@dataclass(frozen=True)
class RawDocxBlock:
    index: int
    text: str
    style_name: str
    source: BlockSource
    table_index: int | None = None
    row_index: int | None = None
    col_index: int | None = None


@dataclass(frozen=True)
class DocxParseDiagnostics:
    heading_count: int
    styled_heading_count: int
    inferred_heading_count: int
    body_count: int
    assigned_body_count: int
    assigned_body_ratio: float
    chunk_count: int
    largest_chunk_body_ratio: float
    fallback_recommended: bool


@dataclass(frozen=True)
class DocxParseResult:
    raw_blocks: list[RawDocxBlock]
    blocks: list[DocxBlock]
    diagnostics: DocxParseDiagnostics


def read_docx(path: Path) -> list[DocxBlock]:
    """Backward-compatible reader: return deterministic parsed blocks only."""
    return read_docx_detailed(path).blocks


def read_docx_detailed(path: Path) -> DocxParseResult:
    doc = Document(str(path))
    raw_blocks = _extract_raw_blocks(doc)
    blocks = _parse_raw_blocks(raw_blocks)
    return DocxParseResult(
        raw_blocks=raw_blocks,
        blocks=blocks,
        diagnostics=_diagnose_blocks(raw_blocks, blocks),
    )


def _extract_raw_blocks(doc: DocxDocument) -> list[RawDocxBlock]:
    raw_blocks: list[RawDocxBlock] = []
    table_index = 0

    for child in doc.element.body.iterchildren():
        if isinstance(child, CT_P):
            para = Paragraph(child, doc)
            _append_paragraph(raw_blocks, para, source="paragraph")
        elif isinstance(child, CT_Tbl):
            table = Table(child, doc)
            for row_index, row in enumerate(table.rows):
                for col_index, cell in enumerate(row.cells):
                    for para in cell.paragraphs:
                        _append_paragraph(
                            raw_blocks,
                            para,
                            source="table_cell",
                            table_index=table_index,
                            row_index=row_index,
                            col_index=col_index,
                        )
            table_index += 1

    return raw_blocks


def _append_paragraph(
    raw_blocks: list[RawDocxBlock],
    para: Paragraph,
    *,
    source: BlockSource,
    table_index: int | None = None,
    row_index: int | None = None,
    col_index: int | None = None,
) -> None:
    text = para.text.strip()
    if not text:
        return
    style_name = para.style.name if para.style is not None else ""
    raw_blocks.append(
        RawDocxBlock(
            index=len(raw_blocks),
            text=text,
            style_name=style_name,
            source=source,
            table_index=table_index,
            row_index=row_index,
            col_index=col_index,
        )
    )


def _parse_raw_blocks(raw_blocks: list[RawDocxBlock]) -> list[DocxBlock]:
    blocks: list[DocxBlock] = []
    stack: list[tuple[int, str]] = []  # (level, text)

    for raw in raw_blocks:
        heading_level = _heading_level(raw)
        if heading_level is not None:
            level = heading_level
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, raw.text))
            blocks.append(
                DocxBlock(
                    text=raw.text,
                    heading_path=[h for _, h in stack],
                    is_heading=True,
                    level=level,
                )
            )
        else:
            blocks.append(
                DocxBlock(
                    text=raw.text,
                    heading_path=[h for _, h in stack],
                    is_heading=False,
                    level=0,
                )
            )
    return blocks


def _heading_level(raw: RawDocxBlock) -> int | None:
    match = _HEADING_RE.match(raw.style_name or "")
    if match:
        return int(match.group(1))
    if _is_common_section_name(raw.text):
        return 1
    if raw.source == "paragraph" and _HEURISTIC_HEADING_RE.match(raw.text):
        return 1
    return None


def _is_common_section_name(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) > 80 or stripped.endswith((".", "!", "?")):
        return False
    normalized = stripped.lower().replace("&", "and")
    normalized = re.sub(r"[^a-z0-9+/# ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized in _COMMON_SECTION_NAMES


def _diagnose_blocks(
    raw_blocks: list[RawDocxBlock],
    blocks: list[DocxBlock],
) -> DocxParseDiagnostics:
    heading_count = sum(1 for block in blocks if block.is_heading)
    styled_heading_count = sum(
        1
        for raw, block in zip(raw_blocks, blocks, strict=False)
        if block.is_heading and _HEADING_RE.match(raw.style_name or "")
    )
    inferred_heading_count = heading_count - styled_heading_count
    body_blocks = [block for block in blocks if not block.is_heading]
    body_count = len(body_blocks)
    assigned_body_count = sum(1 for block in body_blocks if block.heading_path)
    assigned_body_ratio = assigned_body_count / body_count if body_count else 1.0
    chunk_sizes = _body_chunk_sizes(body_blocks)
    largest_chunk_body_ratio = max(chunk_sizes, default=0) / body_count if body_count else 0.0
    complex_unstyled_parse = (
        styled_heading_count == 0 and inferred_heading_count >= 3 and body_count >= 8
    )
    fallback_recommended = (
        heading_count < 2
        or assigned_body_ratio < 0.5
        or (body_count >= 4 and largest_chunk_body_ratio >= 0.9)
        or complex_unstyled_parse
    )
    return DocxParseDiagnostics(
        heading_count=heading_count,
        styled_heading_count=styled_heading_count,
        inferred_heading_count=inferred_heading_count,
        body_count=body_count,
        assigned_body_count=assigned_body_count,
        assigned_body_ratio=assigned_body_ratio,
        chunk_count=len(chunk_sizes),
        largest_chunk_body_ratio=largest_chunk_body_ratio,
        fallback_recommended=fallback_recommended,
    )


def _body_chunk_sizes(body_blocks: list[DocxBlock]) -> list[int]:
    sizes: list[int] = []
    current_path: list[str] | None = None
    current_size = 0
    for block in body_blocks:
        if current_path is None:
            current_path = list(block.heading_path)
            current_size = 1
        elif block.heading_path == current_path:
            current_size += 1
        else:
            sizes.append(current_size)
            current_path = list(block.heading_path)
            current_size = 1
    if current_path is not None:
        sizes.append(current_size)
    return sizes
