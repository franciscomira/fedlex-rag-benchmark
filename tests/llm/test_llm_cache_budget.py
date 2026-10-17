import datetime as dt

import pytest

from fedlex_bench.llm.budget import (
    Budget,
    BudgetExceeded,
    Ledger,
    PlannedCalls,
    project_cost,
)
from fedlex_bench.llm.cache import DiskCache, request_key
from fedlex_bench.llm.prices import PriceTable
from fedlex_bench.types import LLMRequest, Usage

REQ = LLMRequest(system="sys", user="user", max_output_tokens=100, temperature=0.0)


def test_key_depends_on_provider_model_prompt_params():
    base = request_key("anthropic", "m1", REQ, {"effort": "low"})
    assert base == request_key("anthropic", "m1", REQ, {"effort": "low"})
    assert base != request_key("openai", "m1", REQ, {"effort": "low"})
    assert base != request_key("anthropic", "m2", REQ, {"effort": "low"})
    assert base != request_key("anthropic", "m1", REQ, {"effort": "high"})
    for changed in (
        LLMRequest("sys2", "user", 100, 0.0),
        LLMRequest("sys", "user2", 100, 0.0),
        LLMRequest("sys", "user", 101, 0.0),
        LLMRequest("sys", "user", 100, 0.5),
        LLMRequest("sys", "user", 100, 0.0, json_output=True),
    ):
        assert base != request_key("anthropic", "m1", changed, {"effort": "low"})


def test_system_and_user_cannot_collide():
    a = request_key("p", "m", LLMRequest("ab", "c"), {})
    b = request_key("p", "m", LLMRequest("a", "bc"), {})
    assert a != b


def test_disk_cache_roundtrip_and_miss(tmp_path):
    cache = DiskCache(tmp_path / "cache")
    key = request_key("p", "m", REQ, {})
    assert cache.get(key) is None
    assert not cache.has(key)
    usage = Usage(10, 2, 5, 0.001, 1.5, 1)
    cache.put(key, text="hello", model_id="m", usage=usage)
    hit = cache.get(key)
    assert hit is not None and cache.has(key)
    assert hit.text == "hello"
    assert hit.usage == usage
    # survives a new instance (on disk)
    assert DiskCache(tmp_path / "cache").get(key).text == "hello"


def test_corrupt_cache_entry_is_a_miss(tmp_path):
    cache = DiskCache(tmp_path)
    key = request_key("p", "m", REQ, {})
    cache.put(key, text="x", model_id="m", usage=Usage())
    next(tmp_path.rglob("*.json")).write_text("{not json")
    assert cache.get(key) is None


NOW = dt.datetime(2026, 10, 17, 12, tzinfo=dt.UTC)


def test_ledger_sums_only_current_month(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.record("anthropic", "m", 1.5, now=dt.datetime(2026, 9, 30, tzinfo=dt.UTC))
    ledger.record("anthropic", "m", 2.0, now=NOW)
    ledger.record("openai", "m2", 0.25, now=NOW)
    assert ledger.month_spend(NOW) == pytest.approx(2.25)
    assert Ledger(tmp_path / "ledger.jsonl").month_spend(NOW) == pytest.approx(2.25)
    assert Ledger(tmp_path / "none.jsonl").month_spend(NOW) == 0.0


def test_budget_refuses_call_that_would_exceed_cap(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    budget = Budget(monthly_usd=10.0, confirm_usd=2.0, ledger=ledger)
    ledger.record("anthropic", "m", 9.0, now=NOW)
    budget.check(0.99, now=NOW)  # fits
    with pytest.raises(BudgetExceeded):
        budget.check(1.01, now=NOW)
    assert budget.remaining(NOW) == pytest.approx(1.0)


def test_budget_exact_cap_is_allowed(tmp_path):
    budget = Budget(10.0, 2.0, Ledger(tmp_path / "l.jsonl"))
    budget.check(10.0, now=NOW)


def _prices(tmp_path):
    import json

    p = tmp_path / "prices.json"
    p.write_text(
        json.dumps(
            {
                "retrieved": "2026-10-17",
                "models": {"m": {"input": 1.0, "cached_input": 0.1, "output": 10.0}},
            }
        )
    )
    return PriceTable.load(p)


def test_projection_flags_confirm_and_over_budget(tmp_path):
    prices = _prices(tmp_path)
    ledger = Ledger(tmp_path / "l.jsonl")
    budget = Budget(25.0, 5.0, ledger)
    # 1000 calls * (2000 in * $1/M + 500 out * $10/M) = 1000 * 0.007 = 7.0
    plan = [PlannedCalls("m", n_calls=1000, avg_input_tokens=2000, avg_output_tokens=500)]
    p = project_cost(plan, prices, budget, now=NOW)
    assert p.cost_usd == pytest.approx(7.0)
    assert p.needs_confirmation and not p.exceeds_remaining
    ledger.record("openai", "x", 20.0, now=NOW)
    p = project_cost(plan, prices, budget, now=NOW)
    assert p.exceeds_remaining and p.remaining_usd == pytest.approx(5.0)
    small = project_cost(
        [PlannedCalls("m", 10, 1000, 100)],
        prices,
        Budget(25.0, 5.0, Ledger(tmp_path / "e.jsonl")),
        now=NOW,
    )
    assert not small.needs_confirmation and not small.exceeds_remaining


def test_projection_excludes_cached_calls(tmp_path):
    prices = _prices(tmp_path)
    budget = Budget(25.0, 5.0, Ledger(tmp_path / "l.jsonl"))
    plan = [
        PlannedCalls("m", n_calls=1000, avg_input_tokens=2000, avg_output_tokens=500, n_cached=1000)
    ]
    assert project_cost(plan, prices, budget, now=NOW).cost_usd == 0.0
