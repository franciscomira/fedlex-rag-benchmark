import json

import pytest

from fedlex_bench.answer import LLMAnswerer, parse_answer_json, prompt_hash
from fedlex_bench.llm.fake import FakeClient
from fedlex_bench.types import Answerer, Candidate


def cands():
    return [
        Candidate("220:257d", "de", "Der Arbeitgeber ... Frist von 30 Tagen."),
        Candidate("220:257d", "fr", "L'employeur ... delai de 30 jours."),
        Candidate("281.1:16", "it", "Il Consiglio federale fissa la tariffa."),
    ]


def reply(status="answered", answer="Ans", citations=("220:257d",), pointers=()):
    return json.dumps(
        {
            "status": status,
            "answer": answer,
            "citations": list(citations),
            "pointers": list(pointers),
        }
    )


# ---- parser ----


def test_parse_valid_and_lenient_wrappers():
    p = parse_answer_json("```json\n" + reply() + "\n```")
    assert p == {"status": "answered", "answer": "Ans", "citations": ["220:257d"], "pointers": []}
    assert parse_answer_json(reply(status="Refused"))["status"] == "refused"


def test_parse_missing_pointers_defaults_empty():
    text = json.dumps({"status": "refused", "answer": "x", "citations": []})
    assert parse_answer_json(text)["pointers"] == []


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "not json",
        "[1, 2]",
        json.dumps({"status": "maybe", "answer": "x", "citations": []}),
        json.dumps({"status": "answered", "citations": []}),
        json.dumps({"status": "answered", "answer": 5, "citations": []}),
        json.dumps({"status": "answered", "answer": "x", "citations": "220:1"}),
        json.dumps({"status": "answered", "answer": "x", "citations": [1]}),
        json.dumps({"status": "answered", "answer": "x", "citations": [], "pointers": "a"}),
    ],
)
def test_parse_rejects_malformed(bad):
    assert parse_answer_json(bad) is None


# ---- answerer ----


def test_is_an_answerer_with_stable_name():
    a = LLMAnswerer(FakeClient([reply()], model_id="m-1"))
    assert isinstance(a, Answerer)
    assert a.name == "answer/m-1"


def test_valid_answer_with_usage():
    client = FakeClient([reply(citations=["220:257d", "281.1:16"])])
    res = LLMAnswerer(client).answer("Frage?", "de", cands())
    assert res.status == "answered" and res.answer == "Ans"
    assert res.citations == ["220:257d", "281.1:16"]
    assert res.invalid_citations == [] and not res.parse_failed
    assert res.usage.calls == 1 and res.usage.cost_usd > 0


def test_invalid_ids_are_dropped_and_reported():
    client = FakeClient([reply(citations=["220:257d", "220:999", " 220:257d ", "220:999"])])
    res = LLMAnswerer(client).answer("q", "en", cands())
    assert res.citations == ["220:257d"]  # deduped, whitespace-normalised
    assert res.invalid_citations == ["220:999"]  # deduped


def test_id_must_match_exactly_not_by_prefix_or_case():
    client = FakeClient([reply(citations=["220:257", "220:257D", "220"])])
    res = LLMAnswerer(client).answer("q", "en", cands())
    assert res.citations == []
    assert res.invalid_citations == ["220:257", "220:257D", "220"]


def test_refusal_with_valid_pointer():
    client = FakeClient(
        [reply("refused", "Not in the sources; art. 16 sets the tariff.", [], ["281.1:16"])]
    )
    res = LLMAnswerer(client).answer("q", "en", cands())
    assert res.status == "refused" and res.citations == []
    assert res.pointers == ["281.1:16"] and res.invalid_citations == []


def test_invented_pointer_is_dropped_and_counted_invalid():
    client = FakeClient([reply("refused", "No.", [], ["281.1:16", "999:1"])])
    res = LLMAnswerer(client).answer("q", "en", cands())
    assert res.pointers == ["281.1:16"]
    assert res.invalid_citations == ["999:1"]


def test_prompt_numbers_candidates_with_ids_and_asks_for_question_language():
    client = FakeClient([reply()])
    LLMAnswerer(client).answer("Quelle est la duree ?", "fr", cands())
    req = client.requests[0]
    assert req.json_output
    assert "Quelle est la duree ?" in req.user
    assert "[1] id: 220:257d | language: de" in req.user
    assert "[3] id: 281.1:16 | language: it" in req.user
    assert "French" in req.user
    assert "Il Consiglio federale fissa la tariffa." in req.user
    assert "not in the sources" in req.system.lower()


def test_candidate_text_truncated():
    c = [Candidate("1:1", "de", "y" * 20_000)]
    client = FakeClient([reply(citations=["1:1"])])
    LLMAnswerer(client, max_chars_per_candidate=200).answer("q", "en", c)
    assert "y" * 201 not in client.requests[0].user


def test_parse_failure_retries_once_and_succeeds():
    client = FakeClient(["Sorry, here is prose", reply()])
    res = LLMAnswerer(client).answer("q", "en", cands())
    assert not res.parse_failed and res.status == "answered"
    assert len(client.requests) == 2
    assert res.usage.calls == 2  # both calls are paid for and reported
    assert client.requests[1].user != client.requests[0].user


def test_parse_failure_twice_is_flagged_not_hidden():
    client = FakeClient(["nope", "still nope"])
    res = LLMAnswerer(client).answer("q", "en", cands())
    assert res.parse_failed
    assert res.citations == [] and res.invalid_citations == [] and res.answer == ""
    assert len(client.requests) == 2 and res.usage.calls == 2


def test_empty_candidate_list_still_builds_a_prompt():
    client = FakeClient([reply("refused", "No sources.", [], [])])
    res = LLMAnswerer(client).answer("q", "en", [])
    assert res.status == "refused"
    assert "no passages" in client.requests[0].user.lower()


def test_prompt_hash_changes_with_prompt_files_only():
    assert prompt_hash() == prompt_hash() and len(prompt_hash()) == 64
