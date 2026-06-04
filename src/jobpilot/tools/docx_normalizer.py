"""LLM-assisted DOCX section normalization."""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from jobpilot.models.schemas import DocxSectionAssignment, DocxSectionNormalization
from jobpilot.prompts.loader import PromptLoader
from jobpilot.tools.docx_reader import DocxBlock, RawDocxBlock


def normalize_docx_sections(
    raw_blocks: list[RawDocxBlock],
    *,
    llm: ChatOpenAI,
    prompts: PromptLoader | None = None,
    prompt_version: str = "v1",
) -> list[DocxBlock]:
    prompt_loader = prompts or PromptLoader(Path(__file__).parent.parent / "prompts")
    prompt = prompt_loader.render(
        "docx_normalizer",
        prompt_version,
        variables={"raw_blocks": _format_raw_blocks(raw_blocks)},
    )
    messages = [SystemMessage(prompt.system), HumanMessage(prompt.user)]
    normalization: DocxSectionNormalization = llm.with_structured_output(
        DocxSectionNormalization, method="function_calling"
    ).invoke(messages)
    return apply_docx_section_normalization(raw_blocks, normalization)


def apply_docx_section_normalization(
    raw_blocks: list[RawDocxBlock],
    normalization: DocxSectionNormalization,
) -> list[DocxBlock]:
    assignments = {
        assignment.block_index: assignment
        for assignment in normalization.assignments
        if 0 <= assignment.block_index < len(raw_blocks)
    }
    return [_block_from_assignment(raw, assignments.get(raw.index)) for raw in raw_blocks]


def _block_from_assignment(
    raw: RawDocxBlock,
    assignment: DocxSectionAssignment | None,
) -> DocxBlock:
    if assignment is None:
        return DocxBlock(text=raw.text, heading_path=[], is_heading=False, level=0)

    heading_path = [part.strip() for part in assignment.heading_path if part.strip()]
    is_heading = assignment.is_heading
    if is_heading and not heading_path:
        heading_path = [raw.text]
    level = assignment.level if is_heading and assignment.level > 0 else 0
    if is_heading and level == 0:
        level = max(1, len(heading_path))
    return DocxBlock(
        text=raw.text,
        heading_path=heading_path,
        is_heading=is_heading,
        level=level,
    )


def _format_raw_blocks(raw_blocks: list[RawDocxBlock]) -> str:
    lines: list[str] = []
    for block in raw_blocks:
        location: str = block.source
        if block.source == "table_cell":
            location = (
                f"table_cell table={block.table_index} row={block.row_index} col={block.col_index}"
            )
        lines.append(f"[{block.index}] source={location} style={block.style_name}")
        lines.append(block.text)
        lines.append("")
    return "\n".join(lines).strip()
