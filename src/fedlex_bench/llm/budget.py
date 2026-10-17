"""Monthly budget cap, persistent spend ledger and run-cost projection."""

from __future__ import annotations

import datetime as dt
import json
import threading
from dataclasses import dataclass
from pathlib import Path

from fedlex_bench.llm.cache import default_cache_dir
from fedlex_bench.llm.config import load_budget_config
from fedlex_bench.llm.prices import PriceTable


class BudgetExceeded(RuntimeError):
    """Raised before a call that would push this month's spend over the cap."""


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _month(ts: dt.datetime) -> str:
    return ts.astimezone(dt.UTC).strftime("%Y-%m")


class Ledger:
    """Append-only JSONL of live (non-cached) spend. Month = UTC calendar month."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else default_cache_dir() / "ledger.jsonl"
        self._lock = threading.Lock()

    def record(
        self,
        provider: str,
        model_id: str,
        cost_usd: float,
        *,
        now: dt.datetime | None = None,
        **extra,
    ) -> None:
        ts = now or _now()
        row = {
            "ts": ts.isoformat(),
            "month": _month(ts),
            "provider": provider,
            "model_id": model_id,
            "cost_usd": cost_usd,
            **extra,
        }
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def month_spend(self, now: dt.datetime | None = None) -> float:
        month = _month(now or _now())
        total = 0.0
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return 0.0
        for line in lines:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("month") == month:
                total += float(row["cost_usd"])
        return total


class Budget:
    def __init__(self, monthly_usd: float, confirm_usd: float, ledger: Ledger) -> None:
        self.monthly_usd = monthly_usd
        self.confirm_usd = confirm_usd
        self.ledger = ledger

    @classmethod
    def from_config(cls, config_path: Path | None = None, ledger: Ledger | None = None) -> Budget:
        cfg = load_budget_config(config_path)
        return cls(cfg.monthly_usd, cfg.confirm_usd, ledger or Ledger())

    def remaining(self, now: dt.datetime | None = None) -> float:
        return self.monthly_usd - self.ledger.month_spend(now)

    def check(self, planned_cost_usd: float, now: dt.datetime | None = None) -> None:
        """Raise BudgetExceeded if spending `planned_cost_usd` more would pass the cap."""
        spent = self.ledger.month_spend(now)
        if spent + planned_cost_usd > self.monthly_usd + 1e-12:
            raise BudgetExceeded(
                f"call would cost up to {planned_cost_usd:.4f} USD; month spend {spent:.4f} of "
                f"{self.monthly_usd:.2f} USD cap"
            )


@dataclass(frozen=True)
class PlannedCalls:
    """A planned batch of identical-shape calls to one model."""

    model_id: str
    n_calls: int
    avg_input_tokens: int
    avg_output_tokens: int  # include reasoning tokens
    n_cached: int = 0  # calls known to be served from the disk cache (cost 0)


@dataclass(frozen=True)
class Projection:
    cost_usd: float
    remaining_usd: float
    needs_confirmation: bool  # projected cost above confirm_usd
    exceeds_remaining: bool  # projected cost above what is left this month: do not start


def project_cost(
    plan: list[PlannedCalls],
    prices: PriceTable,
    budget: Budget,
    now: dt.datetime | None = None,
) -> Projection:
    cost = 0.0
    for p in plan:
        live = max(p.n_calls - p.n_cached, 0)
        cost += live * prices.cost(p.model_id, p.avg_input_tokens, 0, p.avg_output_tokens)
    remaining = budget.remaining(now)
    return Projection(
        cost_usd=cost,
        remaining_usd=remaining,
        needs_confirmation=cost > budget.confirm_usd,
        exceeds_remaining=cost > remaining,
    )
