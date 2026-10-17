from __future__ import annotations

import pytest
from retrieval_fixtures import FakeEmbedder, make_articles

from fedlex_bench.retrieval import (
    INDEX_CONFIGS,
    RetrievalEnv,
    merge_article_ids,
    retrieve_articles,
)
from fedlex_bench.retrieval.fusion import rrf
from fedlex_bench.types import RetrievalResult, Retriever

ARTICLES = {"220:1", "220:2", "210:7"}


@pytest.fixture()
def env(tmp_path):
    return RetrievalEnv(make_articles(), embedder=FakeEmbedder(), cache_dir=tmp_path)


def test_names(env):
    assert env.retriever("hybrid", "all_lang", "article").name == "hybrid/all_lang/article"
    assert env.retriever("bm25", "single_lang_de", "window").name == "bm25/single_lang_de/window"
    assert env.retriever("embedding", "same_lang").name == "embedding/same_lang/article"


def test_implements_protocol(env):
    r = env.retriever("bm25", "all_lang")
    assert isinstance(r, Retriever)
    assert isinstance(r.retrieve("Mietzins", "de", 3), RetrievalResult)


def test_bad_arguments(env):
    with pytest.raises(ValueError, match="method"):
        env.retriever("tfidf", "all_lang")
    with pytest.raises(ValueError, match="index config"):
        env.retriever("bm25", "single_lang_en")
    with pytest.raises(ValueError, match="chunking"):
        env.retriever("bm25", "all_lang", "sentence")
    r = env.retriever("bm25", "all_lang")
    with pytest.raises(ValueError, match="query_lang"):
        r.retrieve("x", "es", 3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="k must"):
        r.retrieve("x", "de", 0)


def test_index_configs_constant():
    assert INDEX_CONFIGS == (
        "same_lang",
        "single_lang_de",
        "single_lang_fr",
        "single_lang_it",
        "all_lang",
    )


@pytest.mark.parametrize("method", ["bm25", "embedding", "hybrid"])
def test_same_lang_uses_query_language_and_rejects_english(env, method):
    r = env.retriever(method, "same_lang")
    for lang in ("de", "fr", "it"):
        res = r.retrieve("Mietzins loyer pigione", lang, 5)
        assert res.candidates and {c.lang for c in res.candidates} == {lang}
    with pytest.raises(ValueError, match="English"):
        r.retrieve("rent", "en", 5)


@pytest.mark.parametrize("method", ["bm25", "embedding", "hybrid"])
def test_single_lang_is_cross_lingual(env, method):
    r = env.retriever(method, "single_lang_fr")
    for ql in ("en", "de", "fr", "it"):
        res = r.retrieve("Mietzins loyer", ql, 5)
        assert {c.lang for c in res.candidates} <= {"fr"}


def test_all_lang_returns_every_language(env):
    res = env.retriever("embedding", "all_lang").retrieve("Mietzins loyer pigione", "de", 9)
    assert {c.lang for c in res.candidates} == {"de", "fr", "it"}


def test_never_returns_ineligible_articles(env):
    for method in ("bm25", "embedding", "hybrid"):
        res = env.retriever(method, "all_lang").retrieve(
            "Uebergangsrecht Wortlaut leer alt", "de", 100
        )
        assert {c.article_id for c in res.candidates} <= ARTICLES


def test_bm25_finds_lexical_match_and_ranks_best_first(env):
    res = env.retriever("bm25", "single_lang_de").retrieve("Kaufpreis Kaeufer", "de", 3)
    assert res.candidates[0].article_id == "220:2"
    scores = [c.score for c in res.candidates]
    assert scores == sorted(scores, reverse=True) and scores[0] > 0
    assert res.candidates[0].lang == "de" and "Kaufpreis" in res.candidates[0].text


def test_bm25_cross_lingual_without_overlap_returns_nothing(env):
    res = env.retriever("bm25", "single_lang_de").retrieve("prezzo", "it", 5)
    assert res.candidates == []


def test_embedding_scores_are_cosines_sorted(env):
    res = env.retriever("embedding", "single_lang_de").retrieve("Mietzins Mieter", "de", 3)
    scores = [c.score for c in res.candidates]
    assert scores == sorted(scores, reverse=True)
    assert res.candidates[0].article_id == "220:1"
    assert len(res.candidates) == 3


def test_hybrid_is_rrf_of_bm25_and_embedding(env):
    q, k = "Mieter zahlt Mietzins Kaufpreis", 5
    bm = env.retriever("bm25", "all_lang").retrieve(q, "de", 100).candidates
    em = env.retriever("embedding", "all_lang").retrieve(q, "de", 100).candidates

    def key(c):
        return (c.chunk_id, c.lang)

    fused = rrf([[key(c) for c in bm], [key(c) for c in em]], k=60)[:k]
    hy = env.retriever("hybrid", "all_lang").retrieve(q, "de", k).candidates
    assert [key(c) for c in hy] == [d for d, _ in fused]
    assert [c.score for c in hy] == pytest.approx([s for _, s in fused])


def test_deterministic(env):
    r = env.retriever("hybrid", "all_lang")
    a = r.retrieve("Ehe Zivilstandsamt", "de", 6).candidates
    b = r.retrieve("Ehe Zivilstandsamt", "de", 6).candidates
    assert a == b


def test_k_limits_candidates(env):
    res = env.retriever("embedding", "all_lang").retrieve("Mietzins", "de", 4)
    assert len(res.candidates) == 4


@pytest.mark.parametrize("chunking,suffix", [("paragraph", "#p"), ("window", "#w")])
def test_chunk_variants_map_back_to_article_ids(env, chunking, suffix):
    res = env.retriever("embedding", "single_lang_de", chunking).retrieve("Kuendigung", "de", 20)
    assert res.candidates
    for c in res.candidates:
        assert c.chunk_id.startswith(c.article_id + suffix)
        assert c.article_id in ARTICLES
    if chunking == "paragraph":
        ids = {c.chunk_id for c in res.candidates}
        assert {"220:1#p1", "220:1#p2"} <= ids


def test_article_chunking_chunk_id_is_article_id(env):
    res = env.retriever("bm25", "single_lang_de").retrieve("Mietzins", "de", 3)
    assert all(c.chunk_id == c.article_id for c in res.candidates)


def test_retrieve_articles_unique_ids_across_languages_and_chunks(env):
    r = env.retriever("embedding", "all_lang", "paragraph")
    ids = retrieve_articles(r, "Mietzins loyer pigione Kuendigung", "de", 3)
    assert len(ids) == len(set(ids)) == 3
    assert retrieve_articles(r, "Mietzins", "de", 50) == merge_article_ids(
        r.retrieve("Mietzins", "de", 1000).candidates
    )


def test_embeddings_cached_across_retrievers(env):
    env.retriever("embedding", "all_lang").retrieve("a", "de", 1)
    n = env.embedder.inner.passage_calls
    env.retriever("hybrid", "all_lang").retrieve("a", "de", 1)
    env.retriever("embedding", "single_lang_de").retrieve("a", "de", 1)
    assert env.embedder.inner.passage_calls == n


def test_embeddings_reused_from_disk_by_a_new_env(env, tmp_path):
    env.retriever("embedding", "all_lang").retrieve("a", "de", 1)
    fake = FakeEmbedder()
    env2 = RetrievalEnv(make_articles(), embedder=fake, cache_dir=tmp_path)
    a = env.retriever("embedding", "all_lang").retrieve("Ehe", "de", 5).candidates
    b = env2.retriever("embedding", "all_lang").retrieve("Ehe", "de", 5).candidates
    assert fake.passage_calls == 0
    assert a == b
