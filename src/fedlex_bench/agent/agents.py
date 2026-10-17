"""ADK agents of the Fedlex research system.

All agents are custom `BaseAgent` subclasses with deterministic control flow. Model calls go
through the injected `LLMClient` (planner, checker) and `Answerer` (drafting), so the same
tree runs on any provider and on fakes. The root `ResearchAgent` runs the round loop itself
(plan -> retrieve -> draft -> check) with a hard round cap; ADK's LoopAgent is deprecated in
favour of Workflow in 2.12 and cannot express our stop conditions.
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from dataclasses import replace
from typing import Any

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types as genai_types
from typing_extensions import override

from fedlex_bench.agent import prompts
from fedlex_bench.agent.state import (
    LANGS,
    Action,
    RunState,
    parse_actions,
    parse_json_object,
)
from fedlex_bench.agent.tools import Toolbox
from fedlex_bench.types import (
    Answerer,
    AnswerResult,
    Candidate,
    LLMClient,
    LLMRequest,
    QueryLang,
)

REFUSAL_TEXT: dict[str, str] = {
    "en": "The retrieved sources do not support an answer to this question.",
    "de": "Die abgerufenen Quellen belegen keine Antwort auf diese Frage.",
    "fr": "Les sources consultées ne permettent pas de répondre à cette question.",
    "it": "Le fonti consultate non consentono di rispondere a questa domanda.",
}


def _event(author: str, text: str, **state: Any) -> Event:
    return Event(
        author=author,
        content=genai_types.Content(role="model", parts=[genai_types.Part(text=text)]),
        actions=EventActions(state_delta=state),
    )


def refusal(query_lang: QueryLang, pointers: list[str] | None = None) -> AnswerResult:
    return AnswerResult(
        status="refused",
        answer=REFUSAL_TEXT.get(query_lang, REFUSAL_TEXT["en"]),
        citations=[],
        pointers=list(pointers or []),
        invalid_citations=[],
    )


def call_llm(
    llm: LLMClient, rt: RunState, agent: str, request: LLMRequest, **args: Any
) -> str | None:
    """One traced model call. Returns the text, or None when the call raised."""
    t0 = time.perf_counter()
    try:
        resp = llm.complete(request)
    except Exception as exc:
        rt.trace.add(
            agent, "llm", args=args, latency_s=time.perf_counter() - t0,
            error=f"{type(exc).__name__}: {exc}", model_id=getattr(llm, "model_id", ""),
        )  # fmt: skip
        return None
    rt.add_usage(resp.usage)
    rt.trace.add(
        agent, "llm", args=args, usage=resp.usage, latency_s=time.perf_counter() - t0,
        model_id=resp.model_id, cached=resp.cached,
    )  # fmt: skip
    return resp.text


class PlannerAgent(BaseAgent):
    """Decides query languages and sub-queries; revises the plan from checker feedback."""

    rt: Any = None
    llm: Any = None
    max_rounds: int = 3
    default_k: int = 8
    max_actions: int = 6

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        rt: RunState = self.rt
        user = prompts.planner_user(
            rt.query, rt.query_lang, rt.round, self.max_rounds,
            rt.pool.top(12), rt.past_queries, rt.feedback,
        )  # fmt: skip
        text = call_llm(
            self.llm, rt, self.name,
            LLMRequest(system=prompts.PLANNER_SYSTEM, user=user, max_output_tokens=700,
                       json_output=True),
            purpose="plan",
        )  # fmt: skip
        obj = parse_json_object(text or "")
        actions, done = parse_actions(obj, default_k=self.default_k, max_actions=self.max_actions)
        parsed_ok = obj is not None
        if rt.round == 1 and not actions:
            # Fallback keeps the system alive on a bad plan: search the question in every language.
            actions = [
                Action("search", lang, query=rt.query, k=self.default_k, query_lang=rt.query_lang)
                for lang in LANGS
            ]
            done = False
            rt.trace.add(self.name, "decision", args={"fallback": "search all languages"},
                         error=None if parsed_ok else "planner output not parsable")  # fmt: skip
        rt.plan, rt.planner_done = actions, done and rt.round > 1
        rt.past_queries += [a.query for a in actions if a.tool == "search"]
        summary = "; ".join(f"{a.tool}[{a.lang}] {a.query or a.article_id}" for a in actions)
        rt.trace.add(self.name, "decision", args={"actions": [vars(a) for a in actions]},
                     done=rt.planner_done, parse_ok=parsed_ok)  # fmt: skip
        yield _event(self.name, summary or "no actions", plan_size=len(actions), round=rt.round)


class RetrieverAgent(BaseAgent):
    """Executes the planned tool calls addressed to one language."""

    rt: Any = None
    toolbox: Any = None
    lang: str = "de"

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        rt: RunState = self.rt
        tb: Toolbox = self.toolbox
        mine = [a for a in rt.plan if a.lang == self.lang]
        n = 0
        for act in mine:
            n += len(tb.run_action(act))
        yield _event(self.name, f"{len(mine)} calls, {n} hits", round=rt.round)


class AnswerDrafterAgent(BaseAgent):
    """Drafts an answer with citations from the current evidence pool (injected Answerer)."""

    rt: Any = None
    answerer: Any = None
    answer_k: int = 12

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        rt: RunState = self.rt
        cands = rt.pool.top(self.answer_k)
        t0 = time.perf_counter()
        if not cands:
            rt.draft = refusal(rt.query_lang)
            rt.trace.add(self.name, "answer", args={"n_candidates": 0}, status="refused",
                         citations=[], invalid_citations=[], parse_failed=False)  # fmt: skip
        else:
            try:
                res: AnswerResult = self.answerer.answer(rt.query, rt.query_lang, cands)
                err = None
            except Exception as exc:
                res, err = refusal(rt.query_lang), f"{type(exc).__name__}: {exc}"
                res = replace(res, parse_failed=True)
            rt.add_usage(res.usage)
            rt.draft = res
            rt.trace.add(
                self.name, "answer", args={"n_candidates": len(cands), "answerer": getattr(
                    self.answerer, "name", "")}, usage=res.usage,
                latency_s=time.perf_counter() - t0, error=err, status=res.status,
                citations=list(res.citations), invalid_citations=list(res.invalid_citations),
                pointers=list(res.pointers), parse_failed=res.parse_failed,
            )  # fmt: skip
        yield _event(self.name, f"draft {rt.draft.status}", round=rt.round)


class CitationCheckerAgent(BaseAgent):
    """Rejects citations whose article text does not support the draft (code, then LLM)."""

    rt: Any = None
    llm: Any = None

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        rt: RunState = self.rt
        draft: AnswerResult = rt.draft
        rt.verified, rt.rejected, rt.missing_hint = [], [], ""
        cited = list(dict.fromkeys(draft.citations)) if draft.status == "answered" else []
        # Code rule: the id must be in the pool and carry text. Everything else goes to the LLM.
        texts = {a: rt.pool.get(a) for a in cited}
        no_text = [a for a, c in texts.items() if not any(x.text.strip() for x in c)]
        to_check = {a: c for a, c in texts.items() if a not in no_text}
        reasons: dict[str, str] = {a: "no retrieved text" for a in no_text}
        if to_check:
            verdict = self._llm_verdicts(rt, draft.answer, to_check)
            if verdict is None:  # fail closed: unreadable checker output verifies nothing
                reasons.update({a: "checker output unparsable" for a in to_check})
            else:
                supported, why, rt.missing_hint = verdict
                for a in to_check:
                    if a in supported:
                        rt.verified.append(a)
                    else:
                        reasons[a] = why.get(a, "not supported")
        rt.rejected = [a for a in cited if a not in rt.verified]
        rt.trace.add(
            self.name, "check", args={"cited": cited}, result_ids=list(rt.verified),
            rejected=list(rt.rejected), reasons=reasons, missing=rt.missing_hint,
        )  # fmt: skip
        yield _event(self.name, f"verified={rt.verified} rejected={rt.rejected}", round=rt.round)

    def _llm_verdicts(
        self, rt: RunState, answer: str, cited: dict[str, list[Candidate]]
    ) -> tuple[set[str], dict[str, str], str] | None:
        req = LLMRequest(
            system=prompts.CHECKER_SYSTEM, user=prompts.checker_user(answer, cited),
            max_output_tokens=600, json_output=True,
        )  # fmt: skip
        for attempt in (1, 2):  # invalid output is retried once
            text = call_llm(self.llm, rt, self.name, req, purpose="check", attempt=attempt)
            obj = parse_json_object(text or "")
            rows = obj.get("verdicts") if obj else None
            if isinstance(rows, list):
                ok: set[str] = set()
                why: dict[str, str] = {}
                for row in rows:
                    if isinstance(row, dict) and row.get("id") in cited:
                        if row.get("supported") is True:
                            ok.add(row["id"])
                        else:
                            why[row["id"]] = str(row.get("reason", ""))[:200]
                return ok, why, str(obj.get("missing", "") or "").strip()[:300]
        return None


class FinalAnswerAgent(BaseAgent):
    """Builds the final AnswerResult from verified citations only."""

    rt: Any = None
    answerer: Any = None

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        rt: RunState = self.rt
        draft: AnswerResult | None = rt.draft
        if draft is None:
            final = refusal(rt.query_lang)
        elif draft.status == "refused":
            moved = [c for c in draft.citations if c not in draft.pointers]
            final = AnswerResult(
                "refused", draft.answer, [], draft.pointers + moved, draft.invalid_citations,
                draft.parse_failed, draft.usage,
            )  # fmt: skip
        else:
            final = self._answered(rt, draft)
        rt.final = final
        yield _event(self.name, final.status, status=final.status, citations=final.citations)

    def _answered(self, rt: RunState, draft: AnswerResult) -> AnswerResult:
        verified = [c for c in dict.fromkeys(draft.citations) if c in rt.verified]
        if len(verified) == len(set(draft.citations)) and verified:
            return draft
        if not verified:
            return replace(
                refusal(rt.query_lang, draft.pointers),
                invalid_citations=draft.invalid_citations,
                parse_failed=draft.parse_failed,
                usage=draft.usage,
            )
        # Partly rejected: regenerate from the verified articles only, then re-filter by code.
        cands = [c for c in rt.pool.ordered() if c.article_id in verified]
        t0 = time.perf_counter()
        try:
            res: AnswerResult = self.answerer.answer(rt.query, rt.query_lang, cands)
        except Exception as exc:
            rt.trace.add(self.name, "answer", error=f"{type(exc).__name__}: {exc}")
            return refusal(rt.query_lang, draft.pointers)
        rt.add_usage(res.usage)
        rt.trace.add(self.name, "answer", args={"n_candidates": len(cands), "regenerate": True},
                     usage=res.usage, latency_s=time.perf_counter() - t0, status=res.status,
                     citations=list(res.citations), invalid_citations=list(res.invalid_citations),
                     parse_failed=res.parse_failed)  # fmt: skip
        keep = [c for c in dict.fromkeys(res.citations) if c in verified]
        if res.status != "answered" or not keep:
            return AnswerResult(
                "refused", res.answer if res.status == "refused" else
                refusal(rt.query_lang).answer, [], res.pointers, res.invalid_citations,
                res.parse_failed, res.usage,
            )  # fmt: skip
        return AnswerResult(
            "answered", res.answer, keep, res.pointers, res.invalid_citations,
            res.parse_failed, res.usage,
        )  # fmt: skip


def build_feedback(rt: RunState) -> str:
    parts: list[str] = []
    draft = rt.draft
    if draft is not None and draft.status == "refused":
        parts.append("The answerer refused with the current evidence; try different wording, "
                     "other languages, or follow references.")  # fmt: skip
    if rt.rejected:
        parts.append("The citation checker rejected these cited ids as unsupported: "
                     + ", ".join(rt.rejected) + ".")  # fmt: skip
    if draft is not None and draft.status == "answered" and not rt.verified:
        parts.append("No citation could be verified.")
    if rt.missing_hint:
        parts.append("Still missing: " + rt.missing_hint)
    return " ".join(parts)


class ResearchAgent(BaseAgent):
    """Root agent: planner -> language retrievers -> drafter -> citation checker, then final."""

    rt: Any = None
    max_rounds: int = 3

    def _sub(self, name: str) -> BaseAgent:
        agent = self.find_sub_agent(name)
        assert agent is not None
        return agent

    async def _run_sub(self, name: str, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        async for ev in self._sub(name).run_async(ctx):
            yield ev

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        rt: RunState = self.rt
        for rnd in range(1, self.max_rounds + 1):
            rt.round = rnd
            rt.trace.round = rnd
            async for ev in self._run_sub("planner", ctx):
                yield ev
            if rnd > 1 and (rt.planner_done or not rt.plan):
                rt.stop_reason = "planner_done" if rt.planner_done else "no_new_actions"
                break
            for lang in LANGS:
                async for ev in self._run_sub(f"retriever_{lang}", ctx):
                    yield ev
            new = rt.pool.article_ids()
            seen_before = sum(rt.new_ids_per_round)
            gained = len(new) - seen_before
            rt.new_ids_per_round.append(gained)
            rt.trace.add(self.name, "round_end", new_ids_count=gained, pool_size=len(new))
            if rnd > 1 and gained == 0:
                rt.stop_reason = "no_new_evidence"
                break
            async for ev in self._run_sub("answer_drafter", ctx):
                yield ev
            async for ev in self._run_sub("citation_checker", ctx):
                yield ev
            draft = rt.draft
            # Partly rejected citations are dropped by FinalAnswerAgent; a new round is only
            # for refusals, nothing verified, or an explicit gap named by the checker.
            needs_more = draft.status == "refused" or not rt.verified or bool(rt.missing_hint)
            if not needs_more:
                rt.stop_reason = "verified"
                break
            if rt.step_limit_hit:
                rt.stop_reason = "step_limit"
                break
            if rnd == self.max_rounds:
                rt.stop_reason = "max_rounds"
                rt.step_limit_hit = True
                rt.trace.add(self.name, "limit", error="max rounds reached")
                break
            rt.feedback = build_feedback(rt)
        if rt.draft is None:  # stopped before any draft (e.g. step limit on round 1)
            async for ev in self._run_sub("answer_drafter", ctx):
                yield ev
            async for ev in self._run_sub("citation_checker", ctx):
                yield ev
        async for ev in self._run_sub("final_answer", ctx):
            yield ev


def build_research_agent(
    rt: RunState,
    *,
    llm: LLMClient,
    answerer: Answerer,
    toolbox: Toolbox,
    max_rounds: int,
    default_k: int,
    answer_k: int,
) -> ResearchAgent:
    subs: list[BaseAgent] = [
        PlannerAgent(name="planner", description="Plans queries and languages.", rt=rt, llm=llm,
                     max_rounds=max_rounds, default_k=default_k),
        *[
            RetrieverAgent(name=f"retriever_{lang}", description=f"Retrieval in {lang}.", rt=rt,
                           toolbox=toolbox, lang=lang)
            for lang in LANGS
        ],
        AnswerDrafterAgent(name="answer_drafter", description="Drafts a cited answer.", rt=rt,
                           answerer=answerer, answer_k=answer_k),
        CitationCheckerAgent(name="citation_checker", description="Verifies citations.", rt=rt,
                             llm=llm),
        FinalAnswerAgent(name="final_answer", description="Builds the final answer.", rt=rt,
                         answerer=answerer),
    ]  # fmt: skip
    return ResearchAgent(
        name="fedlex_research", description="Fedlex research agent.", rt=rt,
        max_rounds=max_rounds, sub_agents=subs,
    )  # fmt: skip


__all__ = [
    "REFUSAL_TEXT",
    "AnswerDrafterAgent",
    "CitationCheckerAgent",
    "FinalAnswerAgent",
    "PlannerAgent",
    "ResearchAgent",
    "RetrieverAgent",
    "build_research_agent",
    "refusal",
]
