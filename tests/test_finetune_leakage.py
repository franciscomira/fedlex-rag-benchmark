"""Leakage guard for fine-tuning data: nothing the benchmark grades on, sits next to, or nearly
repeats may be in the training pool. The corpus-level tests skip when the data files are missing;
they fail on any overlap."""

from __future__ import annotations

import pandas as pd
import pytest

from fedlex_bench.benchmark.common import id_key
from fedlex_bench.chunk import ARTICLES_PARQUET
from fedlex_bench.eligibility import retrieval_eligible
from fedlex_bench.finetune.exclusions import (
    FINETUNE_DIR,
    TWIN_THRESHOLD,
    neighbours,
    read_gold_ids,
    twins_of_gold,
)

POOL_FILE = FINETUNE_DIR / "train_pool_ids.txt"


def test_neighbours_same_act_only_and_gold_excluded():
    ids = ["220:1", "220:2", "220:3", "220:3a", "210:1", "210:2"]
    assert neighbours(ids, {"220:2"}) == {"220:1", "220:3"}
    assert neighbours(ids, {"220:3"}) == {"220:2", "220:3a"}
    assert neighbours(ids, {"220:1", "220:2"}) == {
        "220:3"
    }  # gold never counts as its own neighbour
    assert neighbours(ids, {"210:1"}) == {"210:2"}  # does not cross into another act
    assert neighbours(ids, {"999:1"}) == set()  # unknown gold id


def test_twins_of_gold_threshold():
    texts = {
        "1:1": "Der Mieter hat den Mietzins am Ende jedes Monats zu bezahlen.",
        "1:2": "Der Mieter hat den Mietzins am Ende jedes Monats zu bezahlen, spätestens am 1.",
        "1:3": "Die Aktiengesellschaft wird durch die Generalversammlung vertreten und geleitet.",
    }
    assert set(twins_of_gold(texts, {"1:1"}, 0.5)) == {"1:2"}
    assert twins_of_gold(texts, {"1:1"}, 0.99) == {}


@pytest.fixture(scope="module")
def pool() -> set[str]:
    if not POOL_FILE.exists():
        pytest.skip("run `python -m fedlex_bench.finetune.exclusions` first")
    return {line for line in POOL_FILE.read_text(encoding="utf-8").splitlines() if line}


def test_pool_has_no_gold(pool):
    assert pool & read_gold_ids() == set()


def test_pool_has_no_gold_neighbours(pool):
    # Independent of exclusions.neighbours(): adjacency in the sorted full eligible id list.
    if not ARTICLES_PARQUET.exists():
        pytest.skip("articles.parquet missing")
    ids = sorted(
        retrieval_eligible(pd.read_parquet(ARTICLES_PARQUET))["article_id"].unique(), key=id_key
    )
    gold = read_gold_ids()
    near = set()
    for i, a in enumerate(ids):
        if a in gold:
            near |= {
                ids[j]
                for j in (i - 1, i + 1)
                if 0 <= j < len(ids) and ids[j].split(":")[0] == a.split(":")[0]
            }
    assert pool & near == set()


def test_pool_has_no_gold_twins(pool):
    if not ARTICLES_PARQUET.exists():
        pytest.skip("articles.parquet missing")
    eligible = retrieval_eligible(pd.read_parquet(ARTICLES_PARQUET))
    german = eligible[eligible["lang"] == "de"].drop_duplicates("article_id")
    texts = dict(zip(german["article_id"], german["text"], strict=True))
    twins = twins_of_gold(texts, read_gold_ids(), TWIN_THRESHOLD)
    assert pool & set(twins) == set()
