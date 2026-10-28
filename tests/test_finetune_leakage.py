"""Leakage guard for fine-tuning data: nothing the benchmark grades on, sits next to, or nearly
repeats may be in the training pool. The corpus-level tests skip when the data files are missing;
they fail on any overlap."""

from __future__ import annotations

import json

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


# --- the files that actually feed training: pairs (positives) and hard negatives -----------------

PAIRS_FILE = FINETUNE_DIR / "pairs.jsonl"
HARDNEG_FILE = FINETUNE_DIR / "hardneg.jsonl"


def _rows(path):
    if not path.exists():
        pytest.skip(f"{path.name} missing")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_every_positive_is_in_the_pool_and_no_article_is_in_both_splits(pool):
    pairs = _rows(PAIRS_FILE)
    assert {p["article_id"] for p in pairs} <= pool
    train = {p["article_id"] for p in pairs if p["split"] == "train"}
    val = {p["article_id"] for p in pairs if p["split"] == "val"}
    assert train and val and train & val == set()


def test_hard_negatives_are_in_the_pool_never_the_positive_and_never_gold(pool):
    pairs = {p["pair_id"]: p for p in _rows(PAIRS_FILE)}
    gold = read_gold_ids()
    for row in _rows(HARDNEG_FILE):
        for neg in row["negatives"]:
            assert neg in pool
            assert neg not in gold
            assert neg != pairs[row["pair_id"]]["article_id"]
        assert pairs[row["pair_id"]]["split"] == "train"


def _grams(text: str) -> set[str]:
    """Own implementation (not the code under test): character 5-grams of lowercased letters."""
    t = " ".join("".join(c if c.isalnum() else " " for c in text.lower()).split())
    return {t[i : i + 5] for i in range(max(len(t) - 4, 1))}


def test_pool_has_no_gold_twins_independent_check(pool):
    if not ARTICLES_PARQUET.exists():
        pytest.skip("articles.parquet missing")
    eligible = retrieval_eligible(pd.read_parquet(ARTICLES_PARQUET))
    german = eligible[eligible["lang"] == "de"].drop_duplicates("article_id")
    text = dict(zip(german["article_id"], german["text"], strict=True))
    gold = {g: _grams(text[g]) for g in read_gold_ids()}
    for a in sorted(pool):
        grams = _grams(text[a])
        for g, gg in gold.items():
            assert len(grams & gg) / len(grams | gg) < TWIN_THRESHOLD, (a, g)


def test_training_runs_used_the_files_that_are_tested(pool):
    """models/<run>/config.json records SHA-256 hashes of the data files: they must still match."""
    import hashlib

    from fedlex_bench.ingest import PROJECT_ROOT

    configs = sorted((PROJECT_ROOT / "models").glob("*/config.json"))
    if not configs:
        pytest.skip("no trained models on this machine")
    files = {
        "pairs.jsonl": PAIRS_FILE,
        "train_pool_ids.txt": POOL_FILE,
        "hardneg.jsonl": HARDNEG_FILE,
    }
    for cfg in configs:
        c = json.loads(cfg.read_text(encoding="utf-8"))
        for name, path in files.items():
            if name == "hardneg.jsonl" and not c["args"].get("hard_negatives"):
                continue  # an in-batch run did not read the negatives
            assert c["data_sha256"][name] == hashlib.sha256(path.read_bytes()).hexdigest(), (
                cfg,
                name,
            )
