import json

from fedlex_bench.eval.judge import Judge, prompt_hashes
from fedlex_bench.types import LLMRequest, LLMResponse, Usage


class FakeClient:
    model_id = "fake-judge-2026-01-01"

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests: list[LLMRequest] = []

    def complete(self, request):
        self.requests.append(request)
        return LLMResponse(self.replies.pop(0), self.model_id, Usage(calls=1, cost_usd=0.01), False)


FACTS = [{"text": "Notice period is 3 months.", "article_id": "220:335c"}]


def test_fact_verdicts_parsed_and_aligned_to_gold_facts():
    c = FakeClient([json.dumps({"facts": [{"index": 0, "supported": True}]})])
    r = Judge(c).fact_recall("q", "answer", FACTS)
    assert r.verdicts == [True] and r.score == 1.0 and r.usage.calls == 1
    assert "Notice period is 3 months." in c.requests[0].user
    assert c.requests[0].json_output and c.requests[0].temperature == 0.0


def test_fact_verdicts_missing_index_counts_unsupported():
    c = FakeClient([json.dumps({"facts": []})])
    assert Judge(c).fact_recall("q", "a", FACTS).score == 0.0


def test_unparseable_judge_reply_retries_once_then_fails():
    c = FakeClient(["not json", "still not"])
    r = Judge(c).fact_recall("q", "a", FACTS)
    assert r.failed and r.score is None and len(c.requests) == 2


def test_faithfulness_score_is_supported_over_claims():
    reply = {"claims": [{"text": "a", "supported": True}, {"text": "b", "supported": False}]}
    r = Judge(FakeClient([json.dumps(reply)])).faithfulness("a b", {"220:1": "text"})
    assert r.score == 0.5 and r.n_claims == 2


def test_faithfulness_no_claims_is_excluded_not_zero():
    r = Judge(FakeClient([json.dumps({"claims": []})])).faithfulness("hm", {"220:1": "t"})
    assert r.score is None and not r.failed


def test_trap_leak_flag():
    r = Judge(FakeClient([json.dumps({"asserts_answer": True})])).trap_leak("q", "a", "reason")
    assert r.leaked is True


def test_prompt_hashes_cover_all_roles():
    assert set(prompt_hashes()) == {"facts", "faithfulness", "trap"}
