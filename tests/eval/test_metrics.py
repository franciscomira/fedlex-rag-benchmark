import pytest

from fedlex_bench.eval.metrics import (
    citation_precision_recall,
    dedupe_ids,
    recall_at_k,
    reciprocal_rank,
    trap_code_checks,
)
from fedlex_bench.types import AnswerResult


def test_dedupe_keeps_first_occurrence():
    assert dedupe_ids(["a:1", "a:2", "a:1", "b:3"]) == ["a:1", "a:2", "b:3"]


def test_recall_at_k_single_and_cross_ref():
    ranked = ["220:1", "220:2", "210:5"]
    assert recall_at_k(ranked, ["220:2"], 1) == 0.0
    assert recall_at_k(ranked, ["220:2"], 2) == 1.0
    assert recall_at_k(ranked, ["220:2", "210:9"], 5) == 0.5
    assert recall_at_k([], ["220:2"], 5) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x:1", "220:2"], ["220:2", "y:1"]) == 0.5
    assert reciprocal_rank(["x:1"], ["220:2"]) == 0.0


def test_citation_precision_recall():
    p, r = citation_precision_recall(["220:1", "220:9"], ["220:1", "220:2"])
    assert (p, r) == (0.5, 0.5)
    assert citation_precision_recall([], ["220:1"]) == (0.0, 0.0)


def _ans(status, citations=(), pointers=(), parse_failed=False):
    return AnswerResult(status, "t", list(citations), list(pointers), [], parse_failed)


@pytest.mark.parametrize(
    "answer,expected",
    [
        (_ans("refused"), True),
        (_ans("refused", pointers=["281.1:16"]), True),
        (_ans("answered", ["220:1"]), False),
        (_ans("answered"), False),
        (_ans("refused", ["220:1"]), False),
        (_ans("refused", parse_failed=True), False),
    ],
)
def test_trap_code_checks(answer, expected):
    assert trap_code_checks(answer) is expected
