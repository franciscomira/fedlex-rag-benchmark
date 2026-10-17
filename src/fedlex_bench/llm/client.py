"""The LLMClient implementation: cache -> budget check -> provider call -> ledger + cache."""

from __future__ import annotations

import math
import time

from fedlex_bench.llm.budget import Budget, PlannedCalls, Projection, project_cost
from fedlex_bench.llm.cache import DiskCache, request_key
from fedlex_bench.llm.config import ModelSpec
from fedlex_bench.llm.prices import PriceTable
from fedlex_bench.llm.transports import Transport
from fedlex_bench.types import LLMRequest, LLMResponse, Usage

CHARS_PER_TOKEN = 3.0  # deliberately low (DE/FR/IT legal text is ~3.5-4.5): overestimates cost


class ManagedClient:
    """One pinned model behind the `LLMClient` protocol."""

    def __init__(
        self,
        spec: ModelSpec,
        transport: Transport,
        cache: DiskCache,
        budget: Budget,
        prices: PriceTable,
    ) -> None:
        self.spec = spec
        self.model_id = spec.model_id
        self.transport = transport
        self.cache = cache
        self.budget = budget
        self.prices = prices
        self.prices.cost(spec.model_id, 0, 0, 0)  # fail early if the model has no price

    def key(self, request: LLMRequest) -> str:
        # effort / temperature policy change the response, so they are part of the key
        extra = {"effort": self.spec.effort, "send_temperature": self.spec.send_temperature}
        return request_key(self.spec.provider, self.spec.model_id, request, extra)

    @staticmethod
    def estimate_input_tokens(request: LLMRequest) -> int:
        return math.ceil((len(request.system) + len(request.user)) / CHARS_PER_TOKEN)

    def complete(self, request: LLMRequest) -> LLMResponse:
        key = self.key(request)
        hit = self.cache.get(key)
        if hit is not None:
            return LLMResponse(text=hit.text, model_id=hit.model_id, usage=hit.usage, cached=True)

        # Worst case: every output token is generated (reasoning tokens count against the cap).
        worst = self.prices.cost(
            self.spec.model_id, self.estimate_input_tokens(request), 0, request.max_output_tokens
        )
        self.budget.check(worst)

        start = time.perf_counter()
        raw = self.transport.call(self.spec, request)
        latency = time.perf_counter() - start
        cost = self.prices.cost(
            self.spec.model_id, raw.input_tokens, raw.cached_input_tokens, raw.output_tokens
        )
        usage = Usage(
            input_tokens=raw.input_tokens,
            cached_input_tokens=raw.cached_input_tokens,
            output_tokens=raw.output_tokens,
            cost_usd=cost,
            latency_s=latency,
            calls=1,
        )
        self.budget.ledger.record(
            self.spec.provider,
            self.spec.model_id,
            cost,
            input_tokens=raw.input_tokens,
            cached_input_tokens=raw.cached_input_tokens,
            output_tokens=raw.output_tokens,
            latency_s=round(latency, 4),
            key=key,
        )
        self.cache.put(key, text=raw.text, model_id=self.spec.model_id, usage=usage)
        return LLMResponse(text=raw.text, model_id=self.spec.model_id, usage=usage, cached=False)

    def project(self, requests: list[LLMRequest], expected_output_tokens: int) -> Projection:
        """Projected live cost of `requests` (cache hits are free) against the budget."""
        misses = [r for r in requests if not self.cache.has(self.key(r))]
        if not misses:
            return project_cost([], self.prices, self.budget)
        avg_in = math.ceil(sum(self.estimate_input_tokens(r) for r in misses) / len(misses))
        plan = [PlannedCalls(self.spec.model_id, len(misses), avg_in, expected_output_tokens)]
        return project_cost(plan, self.prices, self.budget)
