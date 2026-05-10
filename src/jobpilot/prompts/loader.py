"""Versioned YAML prompt loader. Each prompt file has top-level `system` and `user` strings
with Python str-format placeholders, e.g. `{jd}`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PromptTemplate:
    system: str
    user: str


class PromptLoader:
    def __init__(self, root: Path) -> None:
        self._root = root

    def render(
        self,
        agent: str,
        version: str,
        *,
        variables: dict[str, Any],
    ) -> PromptTemplate:
        path = self._root / agent / f"{version}.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"Prompt not found: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or "system" not in raw or "user" not in raw:
            raise ValueError(f"Prompt {path} must define top-level `system` and `user`.")
        system = str(raw["system"]).format_map(_StrictMap(variables))
        user = str(raw["user"]).format_map(_StrictMap(variables))
        return PromptTemplate(system=system.strip(), user=user.strip())


class _StrictMap(dict[str, Any]):
    def __missing__(self, key: str) -> Any:
        raise KeyError(f"Prompt variable not provided: {key!r}")
