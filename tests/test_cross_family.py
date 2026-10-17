import json

from fedlex_bench.benchmark.cross_family import build_request, judge_items, parse_verdict
from fedlex_bench.llm.fake import FakeClient

ITEM = {
    "id": "v1-0001", "qtype": "single", "is_trap": False, "query_lang": "de",
    "question": "Frage?", "gold_article_ids": ["220:1"], "trap_reason": None,
    "gold_facts": [{"text": "Fact one.", "article_id": "220:1"}],
}
TRAP = {**ITEM, "id": "v1-0002", "qtype": "trap", "is_trap": True, "gold_article_ids": [],
        "gold_facts": [], "trap_reason": "Not regulated."}


def texts(ids):
    return {i: f"(de) text of {i}\n\n(fr) texte de {i}" for i in ids}


def test_request_contains_gold_text_facts_and_competitors_but_no_notes():
    req = build_request(ITEM, texts, competitors=["220:2"])
    assert "Fact one." in req.user and "text of 220:1" in req.user and "text of 220:2" in req.user
    assert "Frage?" in req.user and req.json_output and req.temperature == 0.0


def test_trap_request_has_reason_and_competitors_only():
    req = build_request(TRAP, texts, competitors=["281.1:16"])
    assert "Not regulated." in req.user and "text of 281.1:16" in req.user
    assert "text of 220:1" not in req.user


def test_parse_verdict_validates_fields():
    ok = parse_verdict('{"verdict":"reject","issues":["ambiguous"],"reason":"x","confidence":"low"}')
    assert ok == {"verdict": "reject", "issues": ["ambiguous"], "reason": "x", "confidence": "low"}
    assert parse_verdict('{"verdict":"maybe"}') is None
    assert parse_verdict("nope") is None
    assert parse_verdict('{"verdict":"approve","issues":["bogus_code"],"reason":"r"}')["issues"] == ["other"]


def test_judge_items_writes_bound_records_and_counts_failures():
    good = json.dumps({"verdict": "approve", "issues": [], "reason": "fine", "confidence": "high"})
    client = FakeClient([good, "garbage", "garbage"])
    recs, failed = judge_items([ITEM, TRAP], client, texts, lambda it: [])
    assert [r["id"] for r in recs] == ["v1-0001"] and failed == ["v1-0002"]
    assert len(recs[0]["content_sha256"]) == 64
