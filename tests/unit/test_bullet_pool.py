from __future__ import annotations

from pathlib import Path

import pytest

from jobpilot.tools.bullet_pool import (
    BulletPool,
    InvalidBulletIdError,
    load_bullet_pool,
)


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
