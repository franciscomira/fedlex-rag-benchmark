"""Offline test doubles. Tests (and other modules' tests) must use these, never a live API."""

from __future__ import annotations

from collections.abc import Callable

from fedlex_bench.llm.config import ModelSpec
from fedlex_bench.llm.transports import RawResult
from fedlex_bench.types import LLMRequest, LLMResponse, Usage


class FakeTransport:
    """Transport returning scripted RawResults in order (the last one repeats)."""

    def __init__(self, results: list[RawResult]) -> None:
        self._results = list(results)
        self.calls: list[LLMRequest] = []

    def call(self, spec: ModelSpec, request: LLMRequest) -> RawResult:
        self.calls.append(request)
        i = len(self.calls) - 1
        if not self._results:
            raise AssertionError("FakeTransport has no scripted result: unexpected live call")
        return self._results[min(i, len(self._results) - 1)]


class FakeClient:
    """LLMClient returning scripted texts (or computed from the request). No cache, no budget.

    `responses` is a list of texts used in order (the last repeats) or a callable
    `request -> text`. Every call is recorded in `requests`.
    """

    def __init__(
        self,
        responses: list[str] | Callable[[LLMRequest], str],
        model_id: str = "fake-model",
        cost_usd: float = 0.001,
    ) -> None:
        self.model_id = model_id
        self._responses = responses
        self._cost = cost_usd
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if callable(self._responses):
            text = self._responses(request)
        else:
            text = self._responses[min(len(self.requests) - 1, len(self._responses) - 1)]
        usage = Usage(
            input_tokens=100, output_tokens=20, cost_usd=self._cost, latency_s=0.01, calls=1
        )
        return LLMResponse(text=text, model_id=self.model_id, usage=usage, cached=False)
