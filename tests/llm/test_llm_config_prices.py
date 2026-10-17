import json

import pytest

from fedlex_bench.llm.config import (
    load_budget_config,
    load_models,
    load_provider_env_vars,
    repo_root,
    resolve_model,
)
from fedlex_bench.llm.prices import PriceTable


def test_shipped_registry_has_small_and_mid_per_provider():
    models = load_models()
    for provider in ("anthropic", "openai", "google"):
        tiers = {m.tier for m in models.values() if m.provider == provider}
        assert tiers == {"small", "mid"}, provider


def test_shipped_registry_ids_are_pinned_not_floating():
    for model_id in load_models():
        assert not model_id.endswith(("-latest", "-preview")), model_id
        assert model_id in PriceTable.load().models(), f"no price for {model_id}"


def test_resolve_model_by_id_and_by_role():
    models = load_models()
    assert resolve_model("claude-haiku-5-5", models).provider == "anthropic"
    assert resolve_model("openai/mid", models).model_id == "gpt-6.1-sol"
    with pytest.raises(KeyError):
        resolve_model("gpt-4", models)
    with pytest.raises(KeyError):
        resolve_model("openai/huge", models)


def test_budget_and_provider_config_read_from_config_toml():
    cfg = load_budget_config()
    assert cfg.monthly_usd == 25.0
    assert cfg.confirm_usd == 5.0
    env = load_provider_env_vars()
    assert env == {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GEMINI_API_KEY",
    }
    assert (repo_root() / "config.toml").exists()


def _table(tmp_path, models):
    path = tmp_path / "prices.json"
    path.write_text(json.dumps({"retrieved": "2026-10-17", "models": models}))
    return PriceTable.load(path)


def test_cost_uses_unit_prices_per_million(tmp_path):
    t = _table(tmp_path, {"m": {"input": 2.0, "cached_input": 0.5, "output": 10.0}})
    # 1M uncached in, 1M cached in, 1M out
    assert t.cost("m", 1_000_000, 1_000_000, 1_000_000) == pytest.approx(12.5)
    assert t.cost("m", 500, 0, 100) == pytest.approx(500 * 2e-6 + 100 * 10e-6)
    assert t.date == "2026-10-17"


def test_long_context_tier_applies_to_whole_call(tmp_path):
    t = _table(
        tmp_path,
        {
            "m": {
                "input": 0.1,
                "cached_input": 0.01,
                "output": 0.5,
                "long_context": {
                    "above_input_tokens": 100,
                    "input": 1.0,
                    "cached_input": 0.1,
                    "output": 5.0,
                },
            }
        },
    )
    assert t.cost("m", 100, 0, 0) == pytest.approx(100 * 0.1e-6)  # not above
    assert t.cost("m", 90, 20, 10) == pytest.approx(90 * 1e-6 + 20 * 0.1e-6 + 10 * 5e-6)


def test_unknown_model_price_raises(tmp_path):
    t = _table(tmp_path, {"m": {"input": 1, "cached_input": 1, "output": 1}})
    with pytest.raises(KeyError):
        t.cost("other", 1, 0, 1)
