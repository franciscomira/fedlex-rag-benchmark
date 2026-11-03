"""Serving pipeline: same system as the harness, code-validated citations, notices (no network)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "retrieval"))
from retrieval_fixtures import FakeEmbedder, make_articles  # noqa: E402

from fedlex_bench.eval.metrics import dedupe_ids  # noqa: E402
from fedlex_bench.llm.fake import FakeClient  # noqa: E402
from fedlex_bench.retrieval import retrieve_articles  # noqa: E402
from fedlex_bench.serve import (
    ServingPipeline,  # noqa: E402
    notices,  # noqa: E402
)
from fedlex_bench.serve.pipeline import ANSWER_ARTICLES, RERANK_K, RETRIEVE_K  # noqa: E402
from fedlex_bench.types import LLMRequest  # noqa: E402


def llm_reply(request: LLMRequest) -> str:
    if "ranking" in request.user or "ranking" in request.system:
        return json.dumps({"ranking": [1, 2, 3]})
    return json.dumps(
        {
            "status": "answered",
            "answer": "Notice must be in writing.",
            "citations": ["220:1", "999:9"],  # the second id is invented
            "pointers": [],
        }
    )


@pytest.fixture
def pipe(tmp_path):
    return ServingPipeline(
        articles=make_articles(),
        embedder=FakeEmbedder(),
        index_dir=tmp_path,
        llm=FakeClient(llm_reply),
    )


def test_search_matches_the_harness_retriever(pipe):
    q = "Der Mieter zahlt den Mietzins"
    hits = pipe.search(q, k=3)
    expected = retrieve_articles(pipe.retriever, q, "en", 3)
    assert [h.article_id for h in hits] == expected
    assert [h.rank for h in hits] == [1, 2, 3]


def test_hits_carry_all_three_languages_unaltered(pipe):
    hit = pipe.search("Mietzins", k=1)[0]
    assert hit.article_id == "220:1"
    assert set(hit.texts) == {"de", "fr", "it"}
    assert hit.texts["de"] == "1 Der Mieter zahlt den Mietzins.\n2 Die Kuendigung ist schriftlich."


def test_get_article_and_unknown_id(pipe):
    assert pipe.get_article(" 220:2 ").article_id == "220:2"
    assert pipe.get_article("999:9") is None


def test_ask_drops_invented_citations_and_counts_them(pipe):
    res = pipe.ask("Wie ist die Kuendigung?", "de")
    assert res.status == "answered"
    assert [h.article_id for h in res.citations] == ["220:1"]
    assert res.n_invalid_ids == 1
    assert all(h.article_id in pipe.corpus for h in res.hits)


def test_ask_uses_the_frozen_serving_configuration(pipe):
    cfg = pipe.system.config()
    assert (cfg["retrieve_k"], cfg["rerank_k"], cfg["answer_articles"]) == (
        RETRIEVE_K,
        RERANK_K,
        ANSWER_ARTICLES,
    )
    assert cfg["retriever"] == "embedding/all_lang/article"
    assert cfg["reranker"].startswith("llm_rerank/") and cfg["answerer"].startswith("answer/")


def test_ask_ranking_equals_harness_run(pipe):
    q = "Der Kaeufer zahlt den Kaufpreis"
    res = pipe.ask(q, "de", n_hits=10)
    out = pipe.system.run(q, "de")
    assert [h.article_id for h in res.hits] == dedupe_ids(out.ranked_article_ids)[:10]


def test_refusal_is_passed_through(tmp_path):
    refusal = json.dumps(
        {"status": "refused", "answer": "Not in the sources.", "citations": [], "pointers": []}
    )
    pipe = ServingPipeline(
        articles=make_articles(),
        embedder=FakeEmbedder(),
        index_dir=tmp_path,
        llm=FakeClient(
            lambda r: json.dumps({"ranking": [1]}) if "ranking" in r.user + r.system else refusal
        ),
    )
    res = pipe.ask("What is the moon made of?", "en")
    assert res.status == "refused" and res.citations == []


@pytest.mark.parametrize("bad", ["", "   ", "x" * 1001])
def test_bad_questions_are_rejected(pipe, bad):
    with pytest.raises(ValueError):
        pipe.search(bad)
    with pytest.raises(ValueError):
        pipe.ask(bad, "en")


def test_bad_language_is_rejected(pipe):
    with pytest.raises(ValueError):
        pipe.ask("Frage", "es")  # type: ignore[arg-type]


def test_notices_text_and_check():
    assert notices.has_required(notices.notice_text())
    assert not notices.has_required("Not legal advice.")
    assert "Dies ist keine amtliche Veröffentlichung" in notices.notice_text()
