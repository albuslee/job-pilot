"""Read a .docx into ordered blocks tagged with their heading path."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from docx import Document

_HEADING_RE = re.compile(r"^Heading\s+(\d+)$")


@dataclass(frozen=True)
class DocxBlock:
    text: str
    heading_path: list[str]
    is_heading: bool
    level: int  # 0 for body paragraphs


def read_docx(path: Path) -> list[DocxBlock]:
    doc = Document(str(path))
    blocks: list[DocxBlock] = []
    stack: list[tuple[int, str]] = []  # (level, text)

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style_name = para.style.name if para.style is not None else ""
        match = _HEADING_RE.match(style_name or "")
        if match:
            level = int(match.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
            blocks.append(
                DocxBlock(
                    text=text,
                    heading_path=[h for _, h in stack],
                    is_heading=True,
                    level=level,
                )
            )
        else:
            blocks.append(
                DocxBlock(
                    text=text,
                    heading_path=[h for _, h in stack],
                    is_heading=False,
                    level=0,
                )
            )
    return blocks
