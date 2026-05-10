from __future__ import annotations

from pathlib import Path

import pytest

from jobpilot.prompts.loader import PromptLoader, PromptTemplate


def test_loader_renders_yaml_with_vars(tmp_path: Path) -> None:
    prompts_root = tmp_path / "prompts"
    (prompts_root / "evaluator").mkdir(parents=True)
    (prompts_root / "evaluator" / "v1.yaml").write_text(
        "system: |\n  You score {role} JDs.\nuser: |\n  JD:\n  {jd}\n  Profile:\n  {profile}\n"
    )
    loader = PromptLoader(prompts_root)
    rendered = loader.render(
        "evaluator",
        "v1",
        variables={"role": "fullstack", "jd": "JD body", "profile": "Profile excerpt"},
    )
    assert isinstance(rendered, PromptTemplate)
    assert "You score fullstack JDs." in rendered.system
    assert "JD body" in rendered.user
    assert "Profile excerpt" in rendered.user


def test_loader_missing_var_raises(tmp_path: Path) -> None:
    prompts_root = tmp_path / "prompts"
    (prompts_root / "evaluator").mkdir(parents=True)
    (prompts_root / "evaluator" / "v1.yaml").write_text("system: hi {missing}\nuser: x\n")
    loader = PromptLoader(prompts_root)
    with pytest.raises(KeyError):
        loader.render("evaluator", "v1", variables={})


def test_loader_missing_file_raises(tmp_path: Path) -> None:
    loader = PromptLoader(tmp_path)
    with pytest.raises(FileNotFoundError):
        loader.render("evaluator", "v9", variables={})
