"""Retrieval tools the per-language retriever agents call: search, get_article, follow_reference.

Retrievers are injected (any `Retriever`). The optional `ArticleStore` lets a corpus-backed
lookup replace the default behaviour (pool cache, then an exact-id search through the retriever).
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from fedlex_bench.agent.state import Action, RunState, extract_references
from fedlex_bench.types import Candidate, Lang, QueryLang, Retriever, Usage


@runtime_checkable
class ArticleStore(Protocol):
    """Optional exact lookup by article id (e.g. backed by data/processed)."""

    def get(self, article_id: str, lang: Lang) -> Candidate | None: ...


class Toolbox:
    """Tool implementations bound to one run. Every call is traced; failures never raise."""

    def __init__(
        self,
        retrievers: Mapping[Lang, Retriever],
        rt: RunState,
        *,
        store: ArticleStore | None = None,
        max_tool_calls: int = 24,
    ) -> None:
        self.retrievers = retrievers
        self.rt = rt
        self.store = store
        self.max_tool_calls = max_tool_calls
        self._last_usage = Usage()

    # -- public tools ---------------------------------------------------------------------
    def search(
        self, query: str, lang: Lang, k: int, query_lang: QueryLang | None = None
    ) -> list[str]:
        args = {"query": query, "lang": lang, "k": k, "query_lang": query_lang or lang}
        return self._call(
            "search", lang, args, lambda: self._search(query, query_lang or lang, lang, k)
        )

    def get_article(self, article_id: str, lang: Lang) -> list[str]:
        args = {"id": article_id, "lang": lang}
        return self._call(
            "get_article", lang, args, lambda: self._get(article_id, lang), boost=True
        )

    def follow_reference(self, article_id: str, lang: Lang) -> list[str]:
        args = {"id": article_id, "lang": lang}
        return self._call("follow_reference", lang, args, lambda: self._follow(article_id, lang))

    def run_action(self, act: Action) -> list[str]:
        if act.tool == "search":
            return self.search(act.query, act.lang, act.k, act.query_lang)
        if act.tool == "get_article":
            return self.get_article(act.article_id, act.lang)
        return self.follow_reference(act.article_id, act.lang)

    # -- internals ------------------------------------------------------------------------
    def _search(self, query: str, query_lang: QueryLang, lang: Lang, k: int) -> list[Candidate]:
        retriever = self.retrievers.get(lang) or next(iter(self.retrievers.values()))
        result = retriever.retrieve(query, query_lang, k)
        self.rt.add_usage(result.usage)
        self._last_usage = self._last_usage + result.usage
        return list(result.candidates)

    def _get(self, article_id: str, lang: Lang) -> list[Candidate]:
        cached = self.rt.pool.get(article_id, lang)
        if cached:
            return cached
        if self.store is not None:
            cand = self.store.get(article_id, lang)
            return [cand] if cand else []
        found = self._search(article_id, lang, lang, 10)  # exact-id fallback through the retriever
        return [c for c in found if c.article_id == article_id]

    def _follow(self, article_id: str, lang: Lang) -> list[Candidate]:
        source = self._get(article_id, lang)
        out: list[Candidate] = []
        for ref in dict.fromkeys(r for c in source for r in extract_references(c.text, article_id)):
            out.extend(self._get(ref, lang))
        return out

    def _call(self, tool, lang, args, fn, *, boost: bool = False) -> list[str]:
        agent = f"retriever_{lang}"
        rt = self.rt
        if rt.tool_calls >= self.max_tool_calls:
            rt.step_limit_hit = True
            rt.trace.add(agent, "limit", tool=tool, args=args, error="tool call limit reached")
            return []
        rt.tool_calls += 1
        self._last_usage = Usage()
        t0 = time.perf_counter()
        error = None
        cands: list[Candidate] = []
        try:
            cands = fn()
        except Exception as exc:  # tool failure is data for the planner and the classifier
            error = f"{type(exc).__name__}: {exc}"
        wall = time.perf_counter() - t0
        new_ids = rt.pool.add(cands, boost=boost)
        refs = sorted({r for c in cands for r in extract_references(c.text, c.article_id)})
        rt.refs_seen.update(refs)
        ids = list(dict.fromkeys(c.article_id for c in cands))
        rt.trace.add(
            agent,
            "tool",
            tool=tool,
            args=args,
            result_ids=ids,
            usage=self._last_usage,
            latency_s=wall,
            error=error,
            new_ids=new_ids,
            refs_seen=refs,
        )
        return ids
