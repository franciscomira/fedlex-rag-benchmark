"""Fake retriever, LLM and answerer for agent tests. No network, no API keys."""

from __future__ import annotations

import json
from collections.abc import Callable

from fedlex_bench.types import (
    AnswerResult,
    Candidate,
    LLMRequest,
    LLMResponse,
    RetrievalResult,
    Usage,
)

CALL_USAGE = Usage(input_tokens=10, output_tokens=5, cost_usd=0.001, latency_s=0.1, calls=1)


def cand(article_id: str, lang: str = "de", text: str | None = None, score: float = 1.0):
    return Candidate(article_id, lang, text if text is not None else f"text of {article_id}", score)


class FakeRetriever:
    """Returns candidates from `fn(query, query_lang)`; records every call."""

    def __init__(self, fn: Callable[[str, str], list[Candidate]], name: str = "fake/ret") -> None:
        self.fn, self.name, self.calls = fn, name, []

    def retrieve(self, query: str, query_lang: str, k: int) -> RetrievalResult:
        self.calls.append((query, query_lang, k))
        return RetrievalResult(self.fn(query, query_lang)[:k], Usage(latency_s=0.01, calls=1))


class ScriptedLLM:
    """Planner replies come from `plans` (last one repeats); checker from `check(user_text)`."""

    model_id = "fake-llm-2026-01-01"

    def __init__(self, plans: list[str], check: Callable[[str], str] | None = None) -> None:
        self.plans, self.check = plans, check or self.accept_all
        self.requests: list[LLMRequest] = []
        self.n_plan = 0

    @staticmethod
    def accept_all(user: str) -> str:
        ids = [line[4:-4] for line in user.splitlines() if line.startswith("=== ")]
        return json.dumps({"verdicts": [{"id": i, "supported": True} for i in ids], "missing": ""})

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if request.system.startswith("You plan"):
            text = self.plans[min(self.n_plan, len(self.plans) - 1)]
            self.n_plan += 1
        else:
            text = self.check(request.user)
        return LLMResponse(text, self.model_id, CALL_USAGE, cached=False)


class FakeAnswerer:
    """Cites `cite(candidates)`; ids are validated against the candidate list as in production."""

    name = "fake/answerer"

    def __init__(self, cite: Callable[[list[Candidate]], list[str]], refuse_if_empty=True) -> None:
        self.cite, self.refuse_if_empty = cite, refuse_if_empty
        self.calls: list[list[Candidate]] = []

    def answer(self, query: str, query_lang: str, candidates: list[Candidate]) -> AnswerResult:
        self.calls.append(candidates)
        ids = self.cite(candidates)
        valid = [i for i in dict.fromkeys(ids) if i in {c.article_id for c in candidates}]
        invalid = [i for i in dict.fromkeys(ids) if i not in valid]
        if not valid and self.refuse_if_empty:
            return AnswerResult("refused", "no basis", [], [], invalid, usage=CALL_USAGE)
        return AnswerResult("answered", "the answer", valid, [], invalid, usage=CALL_USAGE)


def plan(*actions: dict, done: bool = False) -> str:
    return json.dumps({"actions": list(actions), "done": done})


def search(lang: str, query: str, k: int = 8) -> dict:
    return {"tool": "search", "lang": lang, "query": query, "k": k}
