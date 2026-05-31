from __future__ import annotations

from unittest.mock import MagicMock

from jobpilot.models.schemas import DocxSectionAssignment, DocxSectionNormalization
from jobpilot.tools.docx_normalizer import normalize_docx_sections
from jobpilot.tools.docx_reader import RawDocxBlock


def test_normalize_docx_sections_preserves_text_and_uses_block_indices() -> None:
    raw_blocks = [
        RawDocxBlock(
            index=0, text="Cloud systems builder", style_name="Normal", source="paragraph"
        ),
        RawDocxBlock(index=1, text="Python, TypeScript", style_name="Normal", source="paragraph"),
    ]
    llm = MagicMock()
    llm.complete_structured.return_value = DocxSectionNormalization(
        assignments=[
            DocxSectionAssignment(
                block_index=0,
                heading_path=["Summary"],
                is_heading=False,
                level=0,
            ),
            DocxSectionAssignment(
                block_index=1,
                heading_path=["Skills"],
                is_heading=False,
                level=0,
            ),
        ]
    )

    blocks = normalize_docx_sections(raw_blocks, llm=llm)

    assert [b.text for b in blocks] == ["Cloud systems builder", "Python, TypeScript"]
    assert blocks[0].heading_path == ["Summary"]
    assert blocks[1].heading_path == ["Skills"]
    kwargs = llm.complete_structured.call_args.kwargs
    assert kwargs["schema"] is DocxSectionNormalization
    assert kwargs["tool_name"] == "submit_docx_section_normalization"
