from __future__ import annotations

from pathlib import Path

import pytest

from jobpilot.tools.bullet_pool import (
    BulletPool,
    InvalidBulletIdError,
    extract_current_role_bullets,
    load_bullet_pool,
)
from tests.fixtures.cv_template import ROLE_ANCHOR


def _write_pool(path: Path) -> None:
    path.write_text(
        "current_role:\n  - id: a\n    text: First bullet.\n  - id: b\n    text: Second bullet.\n"
    )


def test_load_pool_returns_ordered_entries(tmp_path: Path) -> None:
    pool_path = tmp_path / "pool.yaml"
    _write_pool(pool_path)

    pool = load_bullet_pool(pool_path)
    assert isinstance(pool, BulletPool)
    assert [b.id for b in pool.current_role] == ["a", "b"]
    assert pool.current_role[0].text == "First bullet."


def test_lookup_resolves_ids_to_texts(tmp_path: Path) -> None:
    pool_path = tmp_path / "pool.yaml"
    _write_pool(pool_path)

    pool = load_bullet_pool(pool_path)
    texts = pool.resolve_current_role(["b", "a"])
    assert texts == ["Second bullet.", "First bullet."]


def test_lookup_rejects_unknown_id(tmp_path: Path) -> None:
    pool_path = tmp_path / "pool.yaml"
    _write_pool(pool_path)
    pool = load_bullet_pool(pool_path)

    with pytest.raises(InvalidBulletIdError) as exc:
        pool.resolve_current_role(["a", "ghost"])
    assert "ghost" in str(exc.value)


def test_lookup_rejects_duplicate_ids(tmp_path: Path) -> None:
    pool_path = tmp_path / "pool.yaml"
    _write_pool(pool_path)
    pool = load_bullet_pool(pool_path)

    with pytest.raises(InvalidBulletIdError) as exc:
        pool.resolve_current_role(["a", "a"])
    assert "duplicate" in str(exc.value).lower()


def test_render_for_prompt_includes_ids_and_text(tmp_path: Path) -> None:
    pool_path = tmp_path / "pool.yaml"
    _write_pool(pool_path)
    pool = load_bullet_pool(pool_path)

    rendered = pool.render_for_prompt()
    assert "[a]" in rendered
    assert "First bullet." in rendered
    assert "[b]" in rendered


def test_load_rejects_missing_current_role_key(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("other_key: []\n")
    with pytest.raises(ValueError, match="current_role"):
        load_bullet_pool(bad)


def test_load_rejects_duplicate_ids_in_pool_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("current_role:\n  - id: x\n    text: A\n  - id: x\n    text: B\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_bullet_pool(bad)


def test_extract_yields_auto_iderd_entries_from_role_block(cv_template: Path) -> None:
    entries = extract_current_role_bullets(cv_template, ROLE_ANCHOR)
    # Synthetic template has 7 List-Paragraph rows under the role.
    assert len(entries) == 7
    assert [e.id for e in entries] == [f"entry-{i:03d}" for i in range(1, 8)]
    assert all(e.text.strip() for e in entries)
    # Verify bullets came from the role block, not from PREVIOUS EXPERIENCE.
    assert not any("BetaCo" in e.text or "GammaWorks" in e.text for e in entries)


def test_extract_raises_when_anchor_missing(cv_template: Path) -> None:
    with pytest.raises(ValueError, match="Role anchor not found"):
        extract_current_role_bullets(cv_template, "NonExistent Corp")


def test_extract_raises_when_tech_line_missing(tmp_path: Path) -> None:
    from docx import Document as Doc

    bad = tmp_path / "bad.docx"
    doc = Doc()
    doc.add_paragraph("ACME Corp, Sydney — Senior Engineer")
    doc.add_paragraph("Did stuff.", style="List Paragraph")
    # No "Tech:" line — extraction should fail clearly.
    doc.save(str(bad))

    with pytest.raises(ValueError, match="Tech:"):
        extract_current_role_bullets(bad, "ACME Corp, Sydney")
