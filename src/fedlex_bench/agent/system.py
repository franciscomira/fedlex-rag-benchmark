"""FedlexAgentSystem: the `System` implementation that runs the ADK agent tree per question."""

from __future__ import annotations

import asyncio
import hashlib
import threading
import time
import uuid
from collections.abc import Coroutine, Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, TypeVar

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from fedlex_bench.agent import prompts
from fedlex_bench.agent.agents import build_research_agent, refusal
from fedlex_bench.agent.state import LANGS, RunState
from fedlex_bench.agent.tools import ArticleStore, Toolbox
from fedlex_bench.agent.trace import TraceRecorder, TraceWriter
from fedlex_bench.types import (
    Answerer,
    Lang,
    LLMClient,
    QueryLang,
    Retriever,
    SystemOutput,
)

_T = TypeVar("_T")
APP_NAME = "fedlex_agent"


def run_coro(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run a coroutine to completion from sync code, also when a loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:  # re-raised in the caller thread
            box["error"] = exc

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


class FedlexAgentSystem:
    """Planner + per-language retrievers + citation checker, built on Google ADK.

    Args:
        retriever: one Retriever used for all languages, or a mapping lang -> Retriever.
        answerer: drafts the cited answer from the evidence pool (code validates ids).
        llm: LLMClient for the planner and the citation checker.
        max_rounds: hard cap on plan/retrieve/check rounds.
        max_tool_calls: hard cap on tool calls per question.
        trace_dir: when set, one JSON line per question is appended to `<trace_dir>/traces.jsonl`.
        store: optional exact article lookup for get_article / follow_reference.
    """

    def __init__(
        self,
        retriever: Retriever | Mapping[Lang, Retriever],
        answerer: Answerer,
        llm: LLMClient,
        *,
        max_rounds: int = 3,
        max_tool_calls: int = 24,
        k: int = 8,
        answer_k: int = 12,
        trace_dir: Path | str | None = None,
        store: ArticleStore | None = None,
        name: str | None = None,
    ) -> None:
        if isinstance(retriever, Mapping):
            missing = [lang for lang in LANGS if lang not in retriever]
            if missing:
                raise ValueError(f"retriever mapping lacks languages: {missing}")
            self.retrievers: dict[Lang, Retriever] = {lang: retriever[lang] for lang in LANGS}
        else:
            self.retrievers = {lang: retriever for lang in LANGS}
        self.answerer, self.llm, self.store = answerer, llm, store
        self.max_rounds, self.max_tool_calls = max_rounds, max_tool_calls
        self.k, self.answer_k = k, answer_k
        self.writer = TraceWriter(Path(trace_dir) / "traces.jsonl") if trace_dir else None
        self.name = name or f"adk_agent/{llm.model_id}/{answerer.name}"

    # -- System protocol ----------------------------------------------------------------------
    def run(self, query: str, query_lang: QueryLang) -> SystemOutput:
        rt = RunState(query=query, query_lang=query_lang, trace=TraceRecorder())
        t0 = time.perf_counter()
        error: str | None = None
        try:
            run_coro(self._run_adk(rt))
        except Exception as exc:  # never lose a trace; the harness counts this as a failure
            error = f"{type(exc).__name__}: {exc}"
        wall = time.perf_counter() - t0
        final = rt.final
        if final is None:
            final = replace(refusal(query_lang), parse_failed=True)
        ranked = list(dict.fromkeys(final.citations + rt.pool.article_ids()))
        rt.trace.add(
            "system", "final", latency_s=wall, error=error, status=final.status,
            citations=list(final.citations), pointers=list(final.pointers),
            invalid_citations=list(final.invalid_citations), parse_failed=final.parse_failed,
            rounds=rt.round, tool_calls=rt.tool_calls, stop_reason=rt.stop_reason or "error",
            step_limit_hit=rt.step_limit_hit, rejected=list(rt.rejected), ranked_ids=ranked[:50],
            max_rounds=self.max_rounds,
        )  # fmt: skip
        out = SystemOutput(ranked, final, rt.usage, rt.trace.steps)
        if self.writer is not None:
            self.writer.write({
                "trace_id": hashlib.sha1(f"{query_lang}|{query}".encode()).hexdigest()[:12],
                "system": self.name, "query": query, "query_lang": query_lang,
                "usage": asdict(rt.usage), "trace": rt.trace.steps,
            })  # fmt: skip
        return out

    def config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": "adk_multi_agent",
            "adk_version": _adk_version(),
            "llm_model_id": self.llm.model_id,
            "answerer": self.answerer.name,
            "retrievers": {lang: r.name for lang, r in self.retrievers.items()},
            "max_rounds": self.max_rounds,
            "max_tool_calls": self.max_tool_calls,
            "k": self.k,
            "answer_k": self.answer_k,
            "prompt_hash": prompts.prompt_hash(),
        }

    # -- ADK wiring ---------------------------------------------------------------------------
    async def _run_adk(self, rt: RunState) -> None:
        toolbox = Toolbox(self.retrievers, rt, store=self.store, max_tool_calls=self.max_tool_calls)
        root = build_research_agent(
            rt, llm=self.llm, answerer=self.answerer, toolbox=toolbox,
            max_rounds=self.max_rounds, default_k=self.k, answer_k=self.answer_k,
        )  # fmt: skip
        sessions = InMemorySessionService()
        runner = Runner(agent=root, app_name=APP_NAME, session_service=sessions)
        session_id = uuid.uuid4().hex
        await sessions.create_session(app_name=APP_NAME, user_id="bench", session_id=session_id)
        message = genai_types.Content(role="user", parts=[genai_types.Part(text=rt.query)])
        async for _event in runner.run_async(
            user_id="bench", session_id=session_id, new_message=message
        ):
            pass


def _adk_version() -> str:
    try:
        import google.adk

        return str(google.adk.__version__)
    except Exception:
        return "unknown"


__all__ = ["FedlexAgentSystem", "run_coro"]
