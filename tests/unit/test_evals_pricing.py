from __future__ import annotations

from pathlib import Path

import pytest

from jobpilot.evals.pricing import PriceTable, cost, load_prices


def _write_prices(tmp_path: Path) -> Path:
    p = tmp_path / "model_prices.yaml"
    p.write_text(
        """
"claude-sonnet-4-6":
  input: 0.003
  output: 0.015
"claude-haiku-4-5-20251001":
  input: 0.001
  output: 0.005
""",
        encoding="utf-8",
    )
    return p


def test_load_prices_returns_typed_table(tmp_path: Path) -> None:
    prices = load_prices(_write_prices(tmp_path))
    assert isinstance(prices, PriceTable)
    assert prices.usd_per_1k("claude-sonnet-4-6") == (0.003, 0.015)


def test_cost_for_known_model(tmp_path: Path) -> None:
    prices = load_prices(_write_prices(tmp_path))
    # 1000 input @ $0.003/1k = $0.003; 1000 output @ $0.015/1k = $0.015; total $0.018
    assert cost("claude-sonnet-4-6", 1000, 1000, prices=prices) == pytest.approx(0.018)


def test_cost_handles_zero_tokens(tmp_path: Path) -> None:
    prices = load_prices(_write_prices(tmp_path))
    assert cost("claude-sonnet-4-6", 0, 0, prices=prices) == 0.0


def test_cost_unknown_model_returns_none(tmp_path: Path) -> None:
    prices = load_prices(_write_prices(tmp_path))
    assert cost("unknown-model-xyz", 1000, 1000, prices=prices) is None


def test_load_prices_rejects_negative(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text('"m":\n  input: -0.001\n  output: 0.01\n', encoding="utf-8")
    with pytest.raises(ValueError, match="must be >= 0"):
        load_prices(p)


def test_load_prices_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_prices(tmp_path / "nope.yaml")


def test_load_prices_rejects_nan(tmp_path: Path) -> None:
    p = tmp_path / "nan.yaml"
    p.write_text('"m":\n  input: .nan\n  output: 0.01\n', encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        load_prices(p)


def test_load_prices_rejects_inf(tmp_path: Path) -> None:
    p = tmp_path / "inf.yaml"
    p.write_text('"m":\n  input: .inf\n  output: 0.01\n', encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        load_prices(p)


def test_load_prices_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    p = tmp_path / "list.yaml"
    p.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(TypeError, match="YAML mapping"):
        load_prices(p)
