import json

import pytest

from fedlex_bench.llm.budget import Budget, BudgetExceeded, Ledger
from fedlex_bench.llm.cache import DiskCache
from fedlex_bench.llm.client import ManagedClient
from fedlex_bench.llm.config import ModelSpec
from fedlex_bench.llm.fake import FakeTransport
from fedlex_bench.llm.prices import PriceTable
from fedlex_bench.llm.transports import ProviderError, RawResult
from fedlex_bench.types import LLMClient, LLMRequest

SPEC = ModelSpec("m-1", "anthropic", "small", "low", False, "url", "2026-10-17")
REQ = LLMRequest(system="s", user="question", max_output_tokens=100)


def make(tmp_path, transport=None, cap=25.0):
    p = tmp_path / "prices.json"
    p.write_text(
        json.dumps(
            {
                "retrieved": "2026-10-17",
                "models": {"m-1": {"input": 1.0, "cached_input": 0.1, "output": 10.0}},
            }
        )
    )
    ledger = Ledger(tmp_path / "ledger.jsonl")
    transport = transport or FakeTransport([RawResult("answer", 1000, 0, 200)])
    client = ManagedClient(
        SPEC,
        transport,
        DiskCache(tmp_path / "cache"),
        Budget(cap, cap * 0.2, ledger),
        PriceTable.load(p),
    )
    return client, transport, ledger


def test_satisfies_llmclient_protocol(tmp_path):
    client, _, _ = make(tmp_path)
    assert isinstance(client, LLMClient)
    assert client.model_id == "m-1"


def test_live_call_logs_usage_cost_and_ledger(tmp_path):
    client, transport, ledger = make(tmp_path)
    r = client.complete(REQ)
    assert r.text == "answer" and not r.cached and r.model_id == "m-1"
    u = r.usage
    assert (u.input_tokens, u.cached_input_tokens, u.output_tokens, u.calls) == (1000, 0, 200, 1)
    assert u.cost_usd == pytest.approx(1000e-6 + 200 * 10e-6)
    assert u.latency_s >= 0
    assert ledger.month_spend() == pytest.approx(u.cost_usd)
    assert len(transport.calls) == 1


def test_second_identical_call_hits_cache_replays_usage_and_costs_nothing(tmp_path):
    client, transport, ledger = make(tmp_path)
    first = client.complete(REQ)
    second = client.complete(REQ)
    assert second.cached and second.text == first.text
    assert second.usage == first.usage  # replays stored tokens, cost, latency
    assert len(transport.calls) == 1
    assert ledger.month_spend() == pytest.approx(first.usage.cost_usd)  # no second charge


def test_cache_survives_new_client_instance(tmp_path):
    client, _, _ = make(tmp_path)
    client.complete(REQ)
    client2, transport2, ledger2 = make(tmp_path, transport=FakeTransport([]))
    assert client2.complete(REQ).cached
    assert transport2.calls == []


def test_different_prompt_is_a_miss(tmp_path):
    client, transport, _ = make(
        tmp_path, FakeTransport([RawResult("a", 1, 0, 1), RawResult("b", 1, 0, 1)])
    )
    assert client.complete(REQ).text == "a"
    assert client.complete(LLMRequest("s", "other", 100)).text == "b"
    assert len(transport.calls) == 2


def test_budget_blocks_call_before_provider_is_hit(tmp_path):
    client, transport, ledger = make(tmp_path, cap=0.001)
    big = LLMRequest("s", "x" * 30_000, 100)
    with pytest.raises(BudgetExceeded):
        client.complete(big)
    assert transport.calls == []
    assert ledger.month_spend() == 0.0


def test_cache_hit_allowed_even_when_budget_is_exhausted(tmp_path):
    client, _, ledger = make(tmp_path)
    client.complete(REQ)
    ledger.record("anthropic", "m-1", 1000.0)  # blow the cap
    assert client.complete(REQ).cached
    with pytest.raises(BudgetExceeded):
        client.complete(LLMRequest("s", "new", 100))


def test_provider_error_is_not_cached_or_billed(tmp_path):
    class Boom:
        def call(self, spec, request):
            raise ProviderError("down")

    client, _, ledger = make(tmp_path, transport=Boom())
    with pytest.raises(ProviderError):
        client.complete(REQ)
    assert ledger.month_spend() == 0.0
    assert client.cache.get(client.key(REQ)) is None


def test_project_requests_skips_cached(tmp_path):
    client, _, _ = make(tmp_path)
    client.complete(REQ)
    other = LLMRequest("s", "new question", 100)
    proj = client.project([REQ, other], expected_output_tokens=200)
    assert proj.cost_usd == pytest.approx(
        client.prices.cost("m-1", client.estimate_input_tokens(other), 0, 200)
    )
    assert not proj.needs_confirmation
