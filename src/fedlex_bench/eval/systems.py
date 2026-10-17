"""Pipeline systems under test: retriever (+ LLM reranker) (+ single-shot cited answerer)."""

from __future__ import annotations

import time
from typing import Any

from fedlex_bench.eval.metrics import dedupe_ids
from fedlex_bench.types import (
    Answerer,
    Candidate,
    QueryLang,
    Reranker,
    Retriever,
    SystemOutput,
    Usage,
)


class PipelineSystem:
    """retrieve `retrieve_k` chunks; optionally rerank the first `rerank_k`; optionally answer.

    Reranked chunks are placed first and the unreranked remainder follows in retriever order,
    so the reranker can only reorder what it saw. The answerer gets every chunk of the top
    `answer_articles` unique articles, at most one per (article, language).
    """

    def __init__(
        self,
        retriever: Retriever,
        *,
        reranker: Reranker | None = None,
        answerer: Answerer | None = None,
        retrieve_k: int = 60,
        rerank_k: int = 30,
        answer_articles: int = 5,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.answerer = answerer
        self.retrieve_k = retrieve_k
        self.rerank_k = rerank_k
        self.answer_articles = answer_articles
        parts = [retriever.name]
        if reranker:
            parts.append(reranker.name)
        if answerer:
            parts.append(answerer.name)
        self.name = " + ".join(parts)

    def _answer_context(self, candidates: list[Candidate]) -> list[Candidate]:
        keep = set(dedupe_ids(c.article_id for c in candidates)[: self.answer_articles])
        seen: set[tuple[str, str]] = set()
        out = []
        for c in candidates:
            key = (c.article_id, c.lang)
            if c.article_id in keep and key not in seen:
                seen.add(key)
                out.append(c)
        return out

    def run(self, query: str, query_lang: QueryLang) -> SystemOutput:
        start = time.perf_counter()
        res = self.retriever.retrieve(query, query_lang, self.retrieve_k)
        local_s = time.perf_counter() - start  # retrieval is local work, measured live
        candidates = list(res.candidates)
        usage = res.usage
        if self.reranker and candidates:
            head, tail = candidates[: self.rerank_k], candidates[self.rerank_k :]
            rr = self.reranker.rerank(query, query_lang, head, len(head))
            candidates = list(rr.candidates) + tail
            usage = usage + rr.usage
        answer = None
        if self.answerer:
            answer = self.answerer.answer(query, query_lang, self._answer_context(candidates))
            usage = usage + answer.usage
        # model calls report their own (replayed on cache hit) latency; add only local time
        usage = Usage(
            usage.input_tokens,
            usage.cached_input_tokens,
            usage.output_tokens,
            usage.cost_usd,
            usage.latency_s + local_s,
            usage.calls,
        )
        return SystemOutput(dedupe_ids(c.article_id for c in candidates), answer, usage)

    def config(self) -> dict[str, Any]:
        return {
            "kind": "pipeline",
            "retriever": self.retriever.name,
            "reranker": self.reranker.name if self.reranker else None,
            "answerer": self.answerer.name if self.answerer else None,
            "retrieve_k": self.retrieve_k,
            "rerank_k": self.rerank_k,
            "answer_articles": self.answer_articles,
        }
