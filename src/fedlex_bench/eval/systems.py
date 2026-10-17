"""Pipeline systems under test: retriever (+ LLM reranker) (+ single-shot cited answerer)."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any

from fedlex_bench.eval.metrics import dedupe_ids
from fedlex_bench.retrieval.fusion import rrf
from fedlex_bench.types import (
    Answerer,
    Candidate,
    LLMClient,
    LLMRequest,
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

    def _retrieve(self, query: str, query_lang: QueryLang) -> tuple[list[Candidate], Usage]:
        res = self.retriever.retrieve(query, query_lang, self.retrieve_k)
        return list(res.candidates), res.usage

    def run(self, query: str, query_lang: QueryLang) -> SystemOutput:
        start = time.perf_counter()
        candidates, usage = self._retrieve(query, query_lang)
        local_s = time.perf_counter() - start  # retrieval is local work, measured live
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


TRANSLATE_SYSTEM = (
    "Translate the user's question about Swiss federal law into German, French and Italian. "
    "Use the legal terminology of the federal acts. Reply with JSON only: "
    '{"de": "...", "fr": "...", "it": "..."}'
)
LANGS = ("de", "fr", "it")


class TranslateRetrieveSystem(PipelineSystem):
    """Control for the agent: one LLM translation call, then per-language retrieval and RRF.

    No planning loop, no checker. Isolates how much of the agent's gain is query translation.
    """

    def __init__(self, retrievers: Mapping[str, Retriever], llm: LLMClient, **kwargs: Any) -> None:
        first = retrievers[LANGS[0]]
        super().__init__(first, **kwargs)
        self.retrievers = dict(retrievers)
        self.llm = llm
        self.name = f"translate[{llm.model_id}] + " + self.name.replace(first.name, "per_lang")

    def _translate(self, query: str, query_lang: str) -> tuple[dict[str, str], Usage]:
        resp = self.llm.complete(
            LLMRequest(TRANSLATE_SYSTEM, query, max_output_tokens=1500, json_output=True)
        )
        try:
            raw = json.loads(resp.text)
        except json.JSONDecodeError:
            raw = {}
        out = {lang: raw.get(lang) if isinstance(raw.get(lang), str) else query for lang in LANGS}
        if query_lang in LANGS:
            out[query_lang] = query  # keep the original wording in its own language
        return out, resp.usage

    def _retrieve(self, query: str, query_lang: QueryLang) -> tuple[list[Candidate], Usage]:
        queries, usage = self._translate(query, query_lang)
        per_lang: list[list[Candidate]] = []
        for lang in LANGS:
            res = self.retrievers[lang].retrieve(queries[lang], lang, self.retrieve_k)
            per_lang.append(list(res.candidates))
            usage = usage + res.usage
        first: dict[str, Candidate] = {}
        for cands in per_lang:
            for c in cands:
                first.setdefault(c.article_id, c)
        fused = rrf([dedupe_ids(c.article_id for c in cands) for cands in per_lang])
        return [first[article_id] for article_id, _ in fused], usage

    def config(self) -> dict[str, Any]:
        return {**super().config(), "kind": "translate_retrieve", "translator": self.llm.model_id,
                "retriever": "per_lang/" + "+".join(r.name for r in self.retrievers.values())}
