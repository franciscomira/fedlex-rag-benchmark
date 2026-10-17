from __future__ import annotations

import pytest

from fedlex_bench.retrieval.fusion import RRF_K, merge_article_ids, rrf
from fedlex_bench.types import Candidate


def test_rrf_default_k_is_60():
    assert RRF_K == 60


def test_rrf_scores_and_order():
    fused = rrf([["a", "b", "c"], ["b", "a", "d"]])
    ids = [d for d, _ in fused]
    assert ids[:2] == ["a", "b"]  # equal score; "a" seen first
    assert dict(fused)["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert dict(fused)["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert dict(fused)["c"] == pytest.approx(1 / 63)
    assert ids[2:] == ["c", "d"]  # equal score; "c" seen first


def test_rrf_single_list_and_empty():
    assert [d for d, _ in rrf([["x", "y"]])] == ["x", "y"]
    assert rrf([[], []]) == []


def test_rrf_custom_k():
    assert dict(rrf([["a"]], k=10))["a"] == pytest.approx(1 / 11)


def test_merge_first_occurrence_wins():
    assert merge_article_ids(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]


def test_merge_candidates_and_cap():
    cands = [
        Candidate("220:1", "de", "t", 3.0, "220:1#p1"),
        Candidate("220:1", "fr", "t", 2.9, "220:1#p1"),
        Candidate("220:2", "it", "t", 2.0, "220:2"),
        Candidate("210:7", "de", "t", 1.0, "210:7"),
    ]
    assert merge_article_ids(cands) == ["220:1", "220:2", "210:7"]
    assert merge_article_ids(cands, k=2) == ["220:1", "220:2"]
