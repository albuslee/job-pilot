"""Model pricing config: tokens → USD."""

from __future__ import annotations

import math
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelPrice(BaseModel):
    model_config = ConfigDict(frozen=True)

    input: float = Field()
    output: float = Field()

    @field_validator("input", "output")
    @classmethod
    def _check_nonneg(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("must be a finite number >= 0")
        if v < 0:
            raise ValueError("must be >= 0")
        return v


class PriceTable(BaseModel):
    model_config = ConfigDict(frozen=True)

    prices: dict[str, ModelPrice]

    def usd_per_1k(self, model: str) -> tuple[float, float] | None:
        mp = self.prices.get(model)
        return (mp.input, mp.output) if mp else None


def load_prices(path: Path) -> PriceTable:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"expected a YAML mapping in {path}, got {type(raw).__name__}")
    parsed: dict[str, ModelPrice] = {}
    for model, entry in raw.items():
        parsed[model] = ModelPrice.model_validate(entry)
    return PriceTable(prices=parsed)


def cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    prices: PriceTable,
) -> float | None:
    per_1k = prices.usd_per_1k(model)
    if per_1k is None:
        return None
    in_price, out_price = per_1k
    return (input_tokens / 1000.0) * in_price + (output_tokens / 1000.0) * out_price
