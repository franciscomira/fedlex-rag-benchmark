"""Failure classification on hand-made traces, plus the state helpers it relies on."""

from __future__ import annotations

import json

from fedlex_bench.agent.failures import CATEGORIES, classify_trace, main
from fedlex_bench.agent.state import Pool, extract_references, parse_actions, parse_json_object
from fedlex_bench.types import Candidate


def tool(lang, ids, tool_name="search", refs=(), error=None, rnd=1, **args):
    return {"kind": "tool", "tool": tool_name, "round": rnd, "args": {"lang": lang, **args},
            "result_ids": list(ids), "refs_seen": list(refs), "error": error}  # fmt: skip


def round_end(n_new, rnd=1):
    return {"kind": "round_end", "round": rnd, "new_ids_count": n_new}


def final(status="answered", citations=(), invalid=(), stop="verified", rounds=1, **kw):
    return {"kind": "final", "status": status, "citations": list(citations),
            "invalid_citations": list(invalid), "stop_reason": stop, "rounds": rounds,
            "parse_failed": False, "step_limit_hit": False, "rejected": [], **kw}  # fmt: skip


ALL = [tool("de", ["1:1"]), tool("fr", ["1:2"]), tool("it", ["1:3"])]


def test_ok_answerable():
    rep = classify_trace([*ALL, final(citations=["1:1"])], ["1:1"])
    assert rep.primary == "ok" and rep.categories == []


def test_ok_trap_refused_clean():
    rep = classify_trace([*ALL, final("refused")], [], is_trap=True)
    assert rep.primary == "ok"


def test_no_gold_retrieved_when_budget_exhausted():
    trace = [*ALL, final("refused", stop="max_rounds", rounds=3, step_limit_hit=True)]
    rep = classify_trace(trace, ["9:9"])
    assert rep.primary == "no_gold_retrieved"
    assert rep.categories == ["no_gold_retrieved", "refused_answerable"]
    assert "step_limit_hit" in rep.flags


def test_wrong_language_retriever_when_a_language_was_never_tried():
    trace = [tool("de", ["1:1"]), tool("fr", ["1:2"]), final("refused", stop="max_rounds")]
    rep = classify_trace(trace, ["9:9"])
    assert rep.primary == "wrong_language_retriever"
    assert rep.details["langs_tried"] == ["de", "fr"]


def test_stopped_too_early():
    trace = [*ALL, final(citations=["1:1"], stop="verified")]
    rep = classify_trace(trace, ["9:9"])
    assert rep.primary == "stopped_too_early"


def test_gold_retrieved_but_not_cited():
    rep = classify_trace([*ALL, final(citations=["1:2"])], ["1:1"])
    assert rep.primary == "gold_retrieved_not_cited"
    assert rep.details["gold_retrieved"] == ["1:1"] and rep.details["gold_cited"] == []


def test_refused_answerable_with_gold_in_pool():
    rep = classify_trace([*ALL, final("refused")], ["1:1"])
    assert rep.primary == "refused_answerable" and rep.categories == ["refused_answerable"]


def test_invalid_citation_even_if_gold_cited():
    rep = classify_trace([*ALL, final(citations=["1:1"], invalid=["7:7"])], ["1:1"])
    assert rep.primary == "invalid_citation"


def test_answered_trap_by_status_and_by_citation():
    assert classify_trace([*ALL, final(citations=["1:1"])], [], True).primary == "answered_trap"
    refused_with_cite = final("refused", citations=["1:1"])
    assert classify_trace([*ALL, refused_with_cite], [], True).primary == "answered_trap"


def test_malformed_output_variants():
    assert classify_trace([*ALL], ["1:1"]).primary == "malformed_output"  # no final step
    bad = final(parse_failed=True)
    assert classify_trace([*ALL, bad], ["1:1"]).categories == ["malformed_output"]
    crashed = final("refused", error="RuntimeError: boom")
    assert classify_trace([*ALL, crashed], ["1:1"]).primary == "malformed_output"


def test_looped_without_new_evidence_needs_a_failure_and_two_rounds():
    trace = [*ALL, round_end(3, 1), round_end(0, 2),
             final("refused", stop="no_new_evidence", rounds=2)]  # fmt: skip
    rep = classify_trace(trace, ["1:1"])
    assert rep.categories == ["refused_answerable", "looped_without_new_evidence"]
    ok_trace = [*ALL, round_end(3, 1), round_end(0, 2), final(citations=["1:1"], rounds=2)]
    assert classify_trace(ok_trace, ["1:1"]).primary == "ok"


def test_flags_missed_reference_checker_rejected_tool_error_partial_gold():
    trace = [tool("de", ["1:1"], refs=["9:9"]), tool("fr", [], error="Timeout"),
             tool("it", ["1:3"]),
             final("refused", stop="max_rounds", rejected=["1:1"],
                   step_limit_hit=True)]  # fmt: skip
    rep = classify_trace(trace, ["9:9"])
    assert {"missed_reference", "checker_rejected", "tool_error", "step_limit_hit"} <= set(
        rep.flags
    )
    follow = tool("de", ["9:9"], tool_name="follow_reference", id="1:1")
    assert "missed_reference" not in classify_trace([*trace, follow], ["9:9"]).flags
    part = classify_trace([*ALL, final(citations=["1:1"])], ["1:1", "1:2"])
    assert part.primary == "ok" and part.flags == ["partial_gold"]


def test_errored_tool_results_do_not_count_as_retrieved():
    trace = [tool("de", ["1:1"], error="boom"), tool("fr", []), tool("it", []),
             final("refused", stop="max_rounds")]  # fmt: skip
    assert classify_trace(trace, ["1:1"]).primary == "no_gold_retrieved"


def test_all_nine_categories_are_reachable():
    assert len(CATEGORIES) == 9 and len(set(CATEGORIES)) == 9


def test_cli_matches_items_and_prints_counts(tmp_path, capsys):
    items = [
        {"id": "v1-0001", "question": "Q1", "query_lang": "de", "gold_article_ids": ["1:1"],
         "is_trap": False},
        {"id": "v1-0002", "question": "Q2", "query_lang": "fr", "gold_article_ids": [],
         "is_trap": True},
    ]  # fmt: skip
    traces = [
        {"query": "Q1", "query_lang": "de", "trace": [*ALL, final(citations=["1:1"])]},
        {"query": "Q2", "query_lang": "fr", "trace": [*ALL, final(citations=["1:1"])]},
        {"query": "unknown", "query_lang": "de", "trace": []},
    ]
    ip, tp, op = tmp_path / "items.jsonl", tmp_path / "traces.jsonl", tmp_path / "out.jsonl"
    ip.write_text("\n".join(json.dumps(i) for i in items), encoding="utf-8")
    tp.write_text("\n".join(json.dumps(t) for t in traces), encoding="utf-8")
    assert main(["--traces", str(tp), "--items", str(ip), "--out", str(op)]) == 0
    out = capsys.readouterr().out
    assert "items classified: 3" in out and "answered_trap" in out and "unmatched_item" in out
    rows = [json.loads(x) for x in op.read_text("utf-8").splitlines()]
    by_id = {r["id"]: r["primary"] for r in rows}
    assert by_id["v1-0001"] == "ok" and by_id["v1-0002"] == "answered_trap"


# -- state helpers ----------------------------------------------------------------------------
def test_extract_references_same_act_only():
    text = (
        "Nach Art. 5 und 6a sowie Art. 12 des Zivilgesetzbuches gilt Art. 7 dieses Gesetzes. "
        "Vgl. art. 20 du code civil et article 30."
    )
    refs = extract_references(text, "220:1")
    assert "220:5" in refs and "220:6a" in refs and "220:7" in refs and "220:30" in refs
    assert "220:12" not in refs and "220:20" not in refs
    assert "220:1" not in extract_references("Siehe Art. 1.", "220:1")


def test_parse_actions_validates_and_caps():
    obj = parse_json_object(
        '```json\n{"actions": [{"tool": "search", "lang": "de", "query": "x"},'
        '{"tool": "search", "lang": "en", "query": "x"},'
        '{"tool": "get_article", "lang": "fr", "id": "bad id"},'
        '{"tool": "get_article", "lang": "fr", "id": "281.1:16"},'
        '{"tool": "search", "lang": "de", "query": "x"}], "done": true}\n```'
    )
    acts, done = parse_actions(obj, default_k=8, max_actions=6)
    assert [(a.tool, a.lang) for a in acts] == [("search", "de"), ("get_article", "fr")]
    assert done is True
    assert parse_actions(None, default_k=8, max_actions=6) == ([], False)


def test_pool_ranks_by_best_position_and_dedupes_languages():
    pool = Pool()
    de = [Candidate("1:1", "de", "a", 0.5), Candidate("1:2", "de", "b", 0.4)]
    fr = [Candidate("1:2", "fr", "c", 0.9), Candidate("1:3", "fr", "d", 0.1)]
    assert pool.add(de) == ["1:1", "1:2"]
    assert pool.add(fr) == ["1:3"]
    assert pool.article_ids() == ["1:2", "1:1", "1:3"]
    assert len(pool.get("1:2")) == 2 and pool.get("1:2", "fr")[0].lang == "fr"
    pool.add([Candidate("1:3", "it", "e")], boost=True)
    assert pool.article_ids()[0] == "1:3"
