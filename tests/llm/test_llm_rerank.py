import json

from fedlex_bench.llm.fake import FakeClient
from fedlex_bench.llm.jsonutil import extract_json
from fedlex_bench.llm.rerank import LLMReranker, prompt_hash
from fedlex_bench.types import Candidate, Reranker


def cands(n=5):
    langs = ["de", "fr", "it"]
    return [
        Candidate(f"220:{i}", langs[i % 3], f"Text of article {i}", score=1.0 - i / 10)
        for i in range(n)
    ]


def ranking(*nums):
    return json.dumps({"ranking": list(nums)})


def test_extract_json_handles_fences_and_prose():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    assert extract_json('Sure! Here:\n{"a": {"b": "}"}} thanks') == {"a": {"b": "}"}}
    assert extract_json("no json") is None
    assert extract_json("") is None
    assert extract_json('{"a": ') is None


def test_is_a_reranker():
    r = LLMReranker(FakeClient([ranking(1)]))
    assert isinstance(r, Reranker)
    assert r.name == "llm_rerank/fake-model"


def test_orders_candidates_by_model_ranking_and_truncates_to_k():
    client = FakeClient([ranking(3, 1, 5)])
    res = LLMReranker(client).rerank("q", "en", cands(5), k=2)
    assert [c.article_id for c in res.candidates] == ["220:2", "220:0"]
    assert res.candidates[0].score > res.candidates[1].score
    assert res.usage.calls == 1 and res.usage.cost_usd > 0
    # candidate objects keep their language and text
    assert res.candidates[0].lang == "it" and res.candidates[0].text == "Text of article 2"


def test_invalid_duplicate_and_missing_numbers_are_handled_by_code():
    client = FakeClient([ranking(2, 2, 99, 0, "x", 4)])
    res = LLMReranker(client).rerank("q", "de", cands(5), k=5)
    # valid, first-occurrence order: 2, 4, then unranked remain in original order: 1, 3, 5
    assert [c.article_id for c in res.candidates] == ["220:1", "220:3", "220:0", "220:2", "220:4"]


def test_prompt_numbers_candidates_names_languages_and_hides_article_ids():
    client = FakeClient([ranking(1)])
    LLMReranker(client).rerank("Wie hoch ist die Frist?", "de", cands(3), k=3)
    req = client.requests[0]
    assert req.json_output
    assert "Wie hoch ist die Frist?" in req.user
    assert "[1]" in req.user and "[3]" in req.user
    assert "Text of article 1" in req.user
    assert "German" in req.user  # query language named
    assert "220:1" not in req.user  # ids are code business; the model ranks by number


def test_passages_truncated():
    c = [Candidate("1:1", "de", "x" * 10_000)]
    client = FakeClient([ranking(1)])
    LLMReranker(client, max_chars_per_passage=100).rerank("q", "en", c, k=1)
    assert "x" * 101 not in client.requests[0].user


def test_parse_failure_retries_once_then_falls_back_to_input_order():
    client = FakeClient(["garbage", "still garbage"])
    r = LLMReranker(client)
    res = r.rerank("q", "fr", cands(4), k=3)
    assert [c.article_id for c in res.candidates] == ["220:0", "220:1", "220:2"]
    assert len(client.requests) == 2
    assert r.n_parse_failures == 1
    assert res.usage.calls == 2


def test_retry_success_after_first_failure():
    client = FakeClient(["garbage", ranking(2, 1)])
    r = LLMReranker(client)
    res = r.rerank("q", "it", cands(3), k=2)
    assert [c.article_id for c in res.candidates] == ["220:1", "220:0"]
    assert r.n_parse_failures == 0


def test_more_candidates_than_window_keep_tail_in_original_order():
    client = FakeClient([ranking(2, 1)])
    res = LLMReranker(client, max_candidates=3).rerank("q", "en", cands(6), k=5)
    assert [c.article_id for c in res.candidates] == ["220:1", "220:0", "220:2", "220:3", "220:4"]
    assert "Text of article 4" not in client.requests[0].user


def test_empty_candidates_make_no_call():
    client = FakeClient([ranking(1)])
    res = LLMReranker(client).rerank("q", "en", [], k=5)
    assert res.candidates == [] and client.requests == []


def test_prompt_hash_is_stable_hex():
    h = prompt_hash()
    assert h == prompt_hash() and len(h) == 64
