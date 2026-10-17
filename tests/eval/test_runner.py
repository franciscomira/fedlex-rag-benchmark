import json

from fedlex_bench.eval.judge import Judge
from fedlex_bench.eval.runner import aggregate, group_id, run_eval, score_item
from fedlex_bench.types import AnswerResult, LLMResponse, SystemOutput, Usage


class FakeJudgeClient:
    model_id = "fake-judge"

    def complete(self, request):
        if "Gold facts" in request.user:
            text = {"facts": [{"index": 0, "supported": True}]}
        elif "Cited articles" in request.user:
            text = {"claims": [{"text": "c", "supported": True}]}
        else:
            text = {"asserts_answer": False}
        return LLMResponse(json.dumps(text), "fake-judge", Usage(calls=1, cost_usd=0.001), False)


ITEM = {
    "id": "v1-0001", "split": "dev", "qtype": "single", "is_trap": False, "query_lang": "de",
    "question": "Frage?", "gold_article_ids": ["220:1"], "translation_of": None,
    "gold_facts": [{"text": "fact", "article_id": "220:1"}], "trap_reason": None,
}
TRAP = {**ITEM, "id": "v1-0002", "qtype": "trap", "is_trap": True, "gold_article_ids": [],
        "gold_facts": [], "trap_reason": "not in law"}


def out(ranked, status="answered", cit=("220:1",)):
    ans = AnswerResult(status, "txt", list(cit), [], [], False, Usage(cost_usd=0.1, calls=1))
    return SystemOutput(ranked, ans, Usage(cost_usd=0.1, latency_s=2.0, calls=1))


def texts(ids):
    return {i: "article text" for i in ids}


def test_group_id_uses_parent_for_translations():
    assert group_id(ITEM) == "v1-0001"
    assert group_id({**ITEM, "id": "v1-0003", "translation_of": "v1-0001"}) == "v1-0001"


def test_score_answerable_item():
    r = score_item(ITEM, out(["220:9", "220:1"]), Judge(FakeJudgeClient()), texts)
    assert r["recall@1"] == 0.0 and r["recall@5"] == 1.0 and r["mrr"] == 0.5
    assert r["fact_recall"] == 1.0 and r["cit_precision"] == 1.0 and r["faithfulness"] == 1.0
    assert r["over_refusal"] is False and r["trap_correct"] is None
    assert r["judge_cost_usd"] > 0 and r["cost_usd"] == 0.1


def test_score_refusal_on_answerable_is_over_refusal_with_zero_recall():
    r = score_item(ITEM, out(["220:1"], "refused", ()), Judge(FakeJudgeClient()), texts)
    assert r["over_refusal"] is True and r["fact_recall"] == 0.0 and r["faithfulness"] is None


def test_score_trap_correct_and_wrong():
    ok = score_item(TRAP, out([], "refused", ()), Judge(FakeJudgeClient()), texts)
    bad = score_item(TRAP, out([], "answered", ("220:1",)), Judge(FakeJudgeClient()), texts)
    assert ok["trap_correct"] is True and bad["trap_correct"] is False
    assert ok["recall@5"] is None


def test_aggregate_counts_traps_and_originals_only():
    rows = [
        score_item(ITEM, out(["220:1"]), Judge(FakeJudgeClient()), texts),
        score_item(TRAP, out([], "refused", ()), Judge(FakeJudgeClient()), texts),
    ]
    m = aggregate(rows)
    assert m["traps"] == {"correct": 1, "n": 1}
    assert m["overall"]["recall@5"]["n"] == 1
    assert m["by_query_lang"]["de"]["recall@5"]["mean"] == 1.0


def test_run_eval_writes_layout(tmp_path):
    class Sys:
        name = "fake"

        def run(self, q, lang):
            return out(["220:1"])

        def config(self):
            return {"kind": "fake"}

    d = run_eval(Sys(), [ITEM, TRAP], Judge(FakeJudgeClient()), tmp_path, run_id="r1",
                 split="dev", texts=texts, git_sha="abc")
    assert {p.name for p in d.iterdir()} == {"config.json", "per_item.jsonl", "metrics.json"}
    cfg = json.loads((d / "config.json").read_text())
    assert cfg["git_sha"] == "abc" and cfg["system"] == {"kind": "fake"} and "prompt_hashes" in cfg
    assert len((d / "per_item.jsonl").read_text().splitlines()) == 2
