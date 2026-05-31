"""Walk profile_dir → docx_reader → chunk_blocks → RagStore.add()."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from jobpilot.logging_setup import get_logger
from jobpilot.models.schemas import ProfileChunk
from jobpilot.rag.store import RagStore
from jobpilot.tools.docx_normalizer import normalize_docx_sections
from jobpilot.tools.docx_reader import DocxBlock, read_docx_detailed

log = get_logger(__name__)


def _chunk_id(source: str, heading_path: list[str], text: str) -> str:
    digest = hashlib.sha1(f"{source}|{'>'.join(heading_path)}|{text}".encode()).hexdigest()[:12]
    return f"{Path(source).stem}-{digest}"


def chunk_blocks(blocks: list[DocxBlock], *, source: str) -> list[ProfileChunk]:
    """Group consecutive body paragraphs that share the same heading_path into one chunk."""
    chunks: list[ProfileChunk] = []
    buffer_text: list[str] = []
    buffer_path: list[str] | None = None

    for block in blocks:
        if block.is_heading:
            if buffer_text and buffer_path is not None:
                text = "\n".join(buffer_text).strip()
                if text:
                    chunks.append(
                        ProfileChunk(
                            id=_chunk_id(source, buffer_path, text),
                            source=source,
                            heading_path=list(buffer_path),
                            text=text,
                        )
                    )
            buffer_text = []
            buffer_path = list(block.heading_path)
            continue
        if buffer_path is None:
            buffer_path = list(block.heading_path)
        if block.heading_path != buffer_path:
            if buffer_text and buffer_path is not None:
                text = "\n".join(buffer_text).strip()
                if text:
                    chunks.append(
                        ProfileChunk(
                            id=_chunk_id(source, buffer_path, text),
                            source=source,
                            heading_path=list(buffer_path),
                            text=text,
                        )
                    )
            buffer_text = []
            buffer_path = list(block.heading_path)
        buffer_text.append(block.text)

    # Final flush
    if buffer_text and buffer_path is not None:
        text = "\n".join(buffer_text).strip()
        if text:
            chunks.append(
                ProfileChunk(
                    id=_chunk_id(source, buffer_path, text),
                    source=source,
                    heading_path=list(buffer_path),
                    text=text,
                )
            )

    return chunks


def ingest_profile_dir(
    profile_dir: Path,
    *,
    store: RagStore,
    smart_docx: bool = False,
    llm: Any | None = None,
) -> int:
    if not profile_dir.is_dir():
        raise FileNotFoundError(f"Profile directory not found: {profile_dir}")
    docx_files = sorted(
        p
        for p in profile_dir.iterdir()
        if p.suffix.lower() == ".docx" and not p.name.startswith("~$")
    )
    if not docx_files:
        log.warning("ingest.no_docx_found", profile_dir=str(profile_dir))
        return 0

    all_chunks: list[ProfileChunk] = []
    for path in docx_files:
        parse_result = read_docx_detailed(path)
        blocks = parse_result.blocks
        diagnostics = parse_result.diagnostics
        if smart_docx and diagnostics.fallback_recommended:
            if llm is None:
                raise ValueError("smart_docx=True requires an LLM client.")
            try:
                blocks = normalize_docx_sections(parse_result.raw_blocks, llm=llm)
                log.info("ingest.docx_normalized", file=path.name)
            except Exception as exc:
                log.warning("ingest.docx_normalize_failed", file=path.name, error=str(exc))
        chunks = chunk_blocks(blocks, source=path.name)
        log.info(
            "ingest.parsed",
            file=path.name,
            chunks=len(chunks),
            headings=diagnostics.heading_count,
            assigned_body_ratio=round(diagnostics.assigned_body_ratio, 3),
            smart_docx=smart_docx,
            fallback_recommended=diagnostics.fallback_recommended,
        )
        all_chunks.extend(chunks)

    store.add(all_chunks)
    log.info("ingest.persisted", total_chunks=len(all_chunks))
    return len(all_chunks)
