from fedlex_bench.eval.systems import PipelineSystem
from fedlex_bench.types import AnswerResult, Candidate, RetrievalResult, Usage


def cand(aid, lang="de"):
    return Candidate(aid, lang, f"text {aid} {lang}", 1.0, aid)


class FakeRetriever:
    name = "fake/all_lang/article"

    def retrieve(self, query, query_lang, k):
        cs = [cand("a:1"), cand("a:1", "fr"), cand("b:2"), cand("c:3"), cand("d:4")]
        return RetrievalResult(cs[:k], Usage(calls=0))


class ReverseReranker:
    name = "rev"

    def rerank(self, query, query_lang, candidates, k):
        return RetrievalResult(list(reversed(candidates))[:k], Usage(cost_usd=0.5, latency_s=1.0, calls=1))


class RecordingAnswerer:
    name = "ans"

    def __init__(self):
        self.seen = None

    def answer(self, query, query_lang, candidates):
        self.seen = candidates
        return AnswerResult("answered", "x", ["a:1"], [], [], False, Usage(cost_usd=0.25, latency_s=2.0, calls=1))


def test_retrieval_only_merges_languages_and_has_no_answer():
    out = PipelineSystem(FakeRetriever(), retrieve_k=10).run("q", "de")
    assert out.ranked_article_ids == ["a:1", "b:2", "c:3", "d:4"] and out.answer is None


def test_reranked_ids_come_first_then_unreranked_remainder():
    s = PipelineSystem(FakeRetriever(), reranker=ReverseReranker(), retrieve_k=5, rerank_k=2)
    out = s.run("q", "de")
    # reranker sees the first 2 chunks only (a:1 de, a:1 fr) and reverses them
    assert out.ranked_article_ids == ["a:1", "b:2", "c:3", "d:4"]
    assert out.usage.cost_usd == 0.5 and out.usage.calls == 1


def test_answerer_gets_top_unique_articles_one_chunk_per_article_language():
    ans = RecordingAnswerer()
    out = PipelineSystem(FakeRetriever(), answerer=ans, retrieve_k=5, answer_articles=2).run("q", "de")
    assert [(c.article_id, c.lang) for c in ans.seen] == [("a:1", "de"), ("a:1", "fr"), ("b:2", "de")]
    assert out.answer.status == "answered"
    assert out.usage.cost_usd == 0.25 and out.usage.latency_s >= 2.0


def test_config_is_json_ready():
    import json

    cfg = PipelineSystem(FakeRetriever(), answerer=RecordingAnswerer()).config()
    json.dumps(cfg)
    assert cfg["retriever"] == "fake/all_lang/article" and cfg["answerer"] == "ans"
