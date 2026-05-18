"""Bullet pool loader.

The pool is the source of truth for which work-experience bullets the tailor
may emit for the candidate's current role. It maps stable IDs → bullet text.
The tailor agent receives this pool in its prompt and returns a list of IDs;
resolving those IDs back to text is the writer's job. Unknown IDs are a hard
error — the tailor MUST NOT invent bullets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


class InvalidBulletIdError(ValueError):
    """Raised when the tailor returns a bullet id not present in the pool, or duplicates one."""


@dataclass(frozen=True)
class BulletEntry:
    id: str
    text: str


@dataclass(frozen=True)
class BulletPool:
    current_role: list[BulletEntry]

    def resolve_current_role(self, ids: list[str]) -> list[str]:
        """Map bullet IDs to text in input order. Raises InvalidBulletIdError on unknown or duplicate IDs."""
        by_id = {b.id: b.text for b in self.current_role}
        seen: set[str] = set()
        out: list[str] = []
        for bid in ids:
            if bid in seen:
                raise InvalidBulletIdError(f"Duplicate bullet id from tailor: {bid!r}")
            if bid not in by_id:
                known = ", ".join(sorted(by_id))
                raise InvalidBulletIdError(
                    f"Unknown bullet id {bid!r} — not in pool. Known ids: {known}"
                )
            seen.add(bid)
            out.append(by_id[bid])
        return out

    def render_for_prompt(self) -> str:
        """Render the pool as `[id]\\ntext` blocks for inclusion in an LLM prompt."""
        return "\n\n".join(f"[{b.id}]\n{b.text}" for b in self.current_role)


def load_bullet_pool(path: Path) -> BulletPool:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "current_role" not in raw:
        raise ValueError(f"Bullet pool {path} must have a top-level `current_role` list.")
    entries = [BulletEntry(id=item["id"], text=item["text"]) for item in raw["current_role"]]
    ids = [e.id for e in entries]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Bullet pool {path} contains duplicate IDs.")
    return BulletPool(current_role=entries)
