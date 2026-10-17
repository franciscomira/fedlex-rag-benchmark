"""End-to-end tests of FedlexAgentSystem on fakes (real ADK runner, no model calls)."""

from __future__ import annotations

import json

from agent_fakes import FakeAnswerer, FakeRetriever, ScriptedLLM, cand, plan, search

from fedlex_bench.agent import FedlexAgentSystem
from fedlex_bench.agent.failures import classify_trace
from fedlex_bench.types import System


def corpus(query: str, lang: str):
    """Query text decides the hit: 'gold' -> 220:1 and 220:2, otherwise a decoy."""
    if "gold" in query:
        return [cand("220:1", lang), cand("220:2", lang)]
    return [cand("999:1", lang)]


def make(plans, retr=corpus, cite=None, check=None, **kw):
    ret = FakeRetriever(retr)
    llm = ScriptedLLM(plans, check)
    ans = FakeAnswerer(cite or (lambda cs: [c.article_id for c in cs if c.article_id == "220:1"]))
    return FedlexAgentSystem(ret, ans, llm, **kw), ret, llm, ans


def test_happy_path_one_round():
    sys_, ret, llm, ans = make([plan(search("de", "gold de"), search("fr", "gold fr"))])
    assert isinstance(sys_, System)
    out = sys_.run("Frage?", "de")
    assert out.answer.status == "answered"
    assert out.answer.citations == ["220:1"]
    assert out.ranked_article_ids[0] == "220:1"
    assert len(set(out.ranked_article_ids)) == len(out.ranked_article_ids)
    assert {c[1] for c in ret.calls} == {"de", "fr"}  # the planner chose the languages
    assert len(ret.calls) == 2
    kinds = [s["kind"] for s in out.trace]
    assert kinds[-1] == "final" and "check" in kinds and "answer" in kinds
    fin = out.trace[-1]
    assert fin["rounds"] == 1 and fin["stop_reason"] == "verified" and not fin["step_limit_hit"]


def test_usage_is_summed_over_all_calls():
    sys_, *_ = make([plan(search("de", "gold"))])
    out = sys_.run("Frage?", "de")
    assert out.usage.calls == sum(s["usage"]["calls"] for s in out.trace)
    assert out.usage.calls == 4  # plan, one retrieval, answer, check
    assert out.usage.input_tokens == 30 and abs(out.usage.cost_usd - 0.003) < 1e-9


def test_trace_steps_have_required_fields():
    sys_, *_ = make([plan(search("it", "gold it"))])
    out = sys_.run("Domanda?", "it")
    for step in out.trace:
        for key in ("step", "round", "agent", "kind", "tool", "args", "result_ids", "usage",
                    "latency_s", "error"):  # fmt: skip
            assert key in step
    tool = next(s for s in out.trace if s["kind"] == "tool")
    assert tool["agent"] == "retriever_it" and tool["tool"] == "search"
    assert tool["args"]["query"] == "gold it" and tool["result_ids"] == ["220:1", "220:2"]


def test_checker_rejection_triggers_second_round():
    def check(user: str) -> str:
        ids = [ln[4:-4] for ln in user.splitlines() if ln.startswith("=== ")]
        ok = [i for i in ids if i != "999:1"]
        return json.dumps({
            "verdicts": [{"id": i, "supported": i in ok, "reason": "off topic"} for i in ids],
            "missing": "" if ok else "need the article on X",
        })  # fmt: skip

    def cite(cands):  # cites every decoy/gold article it is given (regeneration sees only 220:1)
        return [c.article_id for c in cands if c.article_id in {"999:1", "220:1"}]

    sys_, ret, llm, _ = make(
        [plan(search("de", "decoy")), plan(search("fr", "gold fr"))], cite=cite, check=check
    )
    out = sys_.run("Frage?", "de")
    assert out.trace[-1]["rounds"] == 2
    planner_users = [r.user for r in llm.requests if r.system.startswith("You plan")]
    assert "rejected" in planner_users[1] and "Still missing" in planner_users[1]
    assert out.answer.status == "answered"
    assert out.answer.citations == ["220:1"]  # 999:1 was rejected and removed by code
    assert out.trace[-1]["stop_reason"] in {"verified", "max_rounds"}


def test_hard_round_cap_and_refusal_when_nothing_verifies():
    def reject_all(user: str) -> str:
        return json.dumps({"verdicts": [], "missing": "more"})

    plans = [plan(search("de", f"q{i}")) for i in range(1, 6)]
    sys_, ret, llm, _ = make(
        plans,
        retr=lambda q, lang: [cand(f"5:{q[-1]}", lang)],
        cite=lambda cs: [cs[0].article_id],
        check=reject_all,
        max_rounds=3,
    )
    out = sys_.run("Frage?", "de")
    fin = out.trace[-1]
    assert fin["rounds"] == 3 and fin["step_limit_hit"] and fin["stop_reason"] == "max_rounds"
    assert out.answer.status == "refused" and out.answer.citations == []
    assert len(ret.calls) == 3


def test_malformed_plan_falls_back_to_all_languages():
    sys_, ret, *_ = make(["not json at all"])
    out = sys_.run("gold Frage", "de")
    assert len(ret.calls) == 3
    assert [c[1] for c in ret.calls] == ["de", "de", "de"]  # the original query language
    assert out.answer.status == "answered"
    assert any(s["kind"] == "decision" and "fallback" in s["args"] for s in out.trace)


def test_tool_call_cap():
    acts = [search("de", f"gold {i}") for i in range(6)]
    sys_, ret, *_ = make([plan(*acts)], max_tool_calls=2)
    out = sys_.run("Frage?", "de")
    assert len(ret.calls) == 2
    assert out.trace[-1]["step_limit_hit"]
    assert any(s["kind"] == "limit" for s in out.trace)


def test_refusal_path_has_no_citations_and_stops():
    sys_, *_ = make(
        [plan(search("de", "nothing")), plan(done=True)],
        retr=lambda q, lang: [],
        cite=lambda cs: [],
    )
    out = sys_.run("Wie hoch ist die Gebuehr?", "de")
    assert out.answer.status == "refused" and out.answer.citations == []
    assert out.trace[-1]["stop_reason"] in {"planner_done", "no_new_actions", "no_new_evidence"}


def test_invalid_citations_are_reported_never_cited():
    sys_, *_ = make([plan(search("de", "gold"))], cite=lambda cs: ["220:1", "777:7"])
    out = sys_.run("Frage?", "de")
    assert out.answer.citations == ["220:1"] and out.answer.invalid_citations == ["777:7"]


def test_checker_unparsable_fails_closed_after_one_retry():
    sys_, *_ = make([plan(search("de", "gold"))], check=lambda u: "garbage")
    out = sys_.run("Frage?", "de")
    assert out.answer.status == "refused" and out.answer.citations == []
    checks = [s for s in out.trace if s["kind"] == "llm" and s["args"].get("purpose") == "check"]
    assert len(checks) >= 2


def test_follow_reference_resolves_same_act_references():
    def retr(q, lang):
        if q == "start":
            return [cand("220:1", lang, "Es gilt Art. 5 und Art. 6a dieses Gesetzes.")]
        return [cand(q.split()[-1], lang)] if q.startswith("220:") else []

    follow = {"tool": "follow_reference", "lang": "de", "id": "220:1"}
    sys_, *_ = make([plan(search("de", "start"), follow)], retr=retr, cite=lambda cs: ["220:6a"])
    out = sys_.run("Frage?", "de")
    search_step = next(s for s in out.trace if s.get("tool") == "search")
    assert search_step["refs_seen"] == ["220:5", "220:6a"]
    follow_step = next(s for s in out.trace if s.get("tool") == "follow_reference")
    assert follow_step["result_ids"] == ["220:5", "220:6a"]
    assert out.answer.citations == ["220:6a"]


def test_traces_written_as_jsonl(tmp_path):
    sys_, *_ = make([plan(search("de", "gold"))], trace_dir=tmp_path)
    sys_.run("Frage 1?", "de")
    sys_.run("Question 2?", "fr")
    recs = [json.loads(x) for x in (tmp_path / "traces.jsonl").read_text("utf-8").splitlines()]
    assert [r["query"] for r in recs] == ["Frage 1?", "Question 2?"]
    assert recs[0]["trace"][-1]["kind"] == "final" and "usage" in recs[0]


def test_config_is_json_serialisable():
    sys_, *_ = make([plan(search("de", "gold"))])
    cfg = sys_.config()
    json.dumps(cfg)
    assert cfg["max_rounds"] == 3 and cfg["llm_model_id"] == "fake-llm-2026-01-01"
    assert set(cfg["retrievers"]) == {"de", "fr", "it"}


def test_classifier_accepts_real_trace():
    sys_, *_ = make([plan(search("de", "gold"))])
    out = sys_.run("Frage?", "de")
    assert classify_trace(out.trace, ["220:1"]).primary == "ok"
    assert classify_trace(out.trace, [], is_trap=True).primary == "answered_trap"
