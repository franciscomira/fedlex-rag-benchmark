"""Which articles may be used to build fine-tuning data (phase 3).

The training pool is every retrieval-eligible article id minus three groups, so that nothing the
benchmark grades on, or sits next to, or nearly repeats, can reach the model:

1. **Gold**: every id in ``benchmark/v1/dev_gold_articles.txt`` and ``test_gold_articles.txt``.
2. **Neighbours**: the previous and the next retrieval-eligible article of the same act, in
   natural article order, for each gold article. ("One article number either side"; letter
   suffixes such as 257d/257e count as separate articles, repealed gaps are skipped.)
3. **Twins**: any article whose German text has a character 5-gram Jaccard of at least
   ``TWIN_THRESHOLD`` with a gold article. The benchmark itself used 0.7 (``benchmark/twins.py``)
   to keep gold unique; for leakage we are stricter, because a 60% copy of a gold article still
   teaches the model that article's wording.

Deterministic: sorted ids, no randomness. Run ``uv run python -m fedlex_bench.finetune.exclusions``
to write ``data/finetune/excluded_ids.json`` and ``data/finetune/train_pool_ids.txt``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from fedlex_bench.benchmark.common import id_key
from fedlex_bench.benchmark.twins import char_ngrams, jaccard, normalise_text
from fedlex_bench.chunk import ARTICLES_PARQUET
from fedlex_bench.eligibility import retrieval_eligible
from fedlex_bench.ingest import PROJECT_ROOT

TWIN_THRESHOLD = 0.5
BENCHMARK_DIR = PROJECT_ROOT / "benchmark" / "v1"
FINETUNE_DIR = PROJECT_ROOT / "data" / "finetune"
GOLD_FILES = ("dev_gold_articles.txt", "test_gold_articles.txt")


def read_gold_ids(benchmark_dir: Path = BENCHMARK_DIR) -> set[str]:
    gold: set[str] = set()
    for name in GOLD_FILES:
        lines = (benchmark_dir / name).read_text(encoding="utf-8").splitlines()
        gold |= {line.strip() for line in lines if line.strip()}
    return gold


def neighbours(eligible_ids: Iterable[str], gold: Iterable[str]) -> set[str]:
    """Previous and next eligible article of the same act for each gold id (gold excluded)."""
    ordered = sorted(set(eligible_ids), key=id_key)
    position = {a: i for i, a in enumerate(ordered)}
    gold = set(gold)
    found: set[str] = set()
    for g in gold & position.keys():
        sr = g.partition(":")[0]
        for j in (position[g] - 1, position[g] + 1):
            if 0 <= j < len(ordered) and ordered[j].partition(":")[0] == sr:
                found.add(ordered[j])
    return found - gold


def twins_of_gold(
    german_texts: dict[str, str], gold: Iterable[str], threshold: float = TWIN_THRESHOLD
) -> dict[str, float]:
    """``{article_id: best jaccard to a gold article}`` for non-gold articles >= ``threshold``."""
    gold = set(gold) & german_texts.keys()
    grams = {a: char_ngrams(normalise_text(t)) for a, t in german_texts.items()}
    twins: dict[str, float] = {}
    for a in sorted(set(german_texts) - gold, key=id_key):
        best = max((jaccard(grams[a], grams[g]) for g in gold), default=0)
        if best >= threshold:
            twins[a] = float(best)
    return twins


def build_exclusions(
    articles: pd.DataFrame, gold: set[str], threshold: float = TWIN_THRESHOLD
) -> dict:
    """Return the exclusion sets and the training pool for a corpus frame (see module docstring)."""
    eligible = retrieval_eligible(articles)
    ids = sorted(eligible["article_id"].unique(), key=id_key)
    german = eligible[eligible["lang"] == "de"].drop_duplicates("article_id")
    texts = dict(zip(german["article_id"], german["text"], strict=True))
    nb = neighbours(ids, gold)
    tw = twins_of_gold(texts, gold, threshold)
    excluded = (gold | nb | set(tw)) & set(ids)
    return {
        "eligible": ids,
        "gold": sorted(gold & set(ids), key=id_key),
        "neighbours": sorted(nb, key=id_key),
        "twins": {a: round(s, 4) for a, s in tw.items()},
        "twin_threshold": threshold,
        "train_pool": [a for a in ids if a not in excluded],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--threshold", type=float, default=TWIN_THRESHOLD)
    args = parser.parse_args(argv)
    result = build_exclusions(pd.read_parquet(ARTICLES_PARQUET), read_gold_ids(), args.threshold)
    FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in result.items() if k not in ("eligible", "train_pool")}
    (FINETUNE_DIR / "excluded_ids.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    (FINETUNE_DIR / "train_pool_ids.txt").write_text(
        "\n".join(result["train_pool"]) + "\n", encoding="utf-8"
    )
    print(
        f"eligible {len(result['eligible'])}  gold {len(result['gold'])}  "
        f"neighbours {len(result['neighbours'])}  "
        f"twins(>={args.threshold}) {len(result['twins'])}  "
        f"train pool {len(result['train_pool'])}"
    )


if __name__ == "__main__":
    main()
