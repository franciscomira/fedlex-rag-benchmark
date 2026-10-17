"""Dated price table (eval/prices.json) and cost arithmetic."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from fedlex_bench.llm.config import repo_root


def default_prices_path() -> Path:
    return repo_root() / "eval" / "prices.json"


@dataclass(frozen=True)
class PriceTable:
    date: str  # day the prices were retrieved; reports must state it
    _models: dict

    @classmethod
    def load(cls, path: Path | None = None) -> PriceTable:
        data = json.loads((path or default_prices_path()).read_text(encoding="utf-8"))
        return cls(date=data["retrieved"], _models=data["models"])

    def models(self) -> dict:
        return self._models

    def cost(
        self, model_id: str, input_tokens: int, cached_input_tokens: int, output_tokens: int
    ) -> float:
        """USD for one call. `input_tokens` excludes cached ones; output includes reasoning."""
        if model_id not in self._models:
            raise KeyError(f"no price for model {model_id!r} in the price table")
        p = self._models[model_id]
        lc = p.get("long_context")
        if lc and input_tokens + cached_input_tokens > lc["above_input_tokens"]:
            p = lc
        return (
            input_tokens * p["input"]
            + cached_input_tokens * p["cached_input"]
            + output_tokens * p["output"]
        ) / 1_000_000
