"""BM25 recall@10 baseline on the DEV split (never the test split).

Indexes: one per corpus language (de, fr, it) and ``all`` (the three language versions of every
article as separate documents). Traps are excluded. A query is run against each index with the
package's BM25 (``bm25.py``).

- Recall of an item = share of its gold ids found in the top 10. For a ``single`` item that is
  a hit or a miss; for ``cross_ref`` it is 0, 0.5 or 1. In the ``all`` index a gold id counts as
  found if any of its language versions is in the top 10.
- Only documents with a positive BM25 score count. The tie-break order of zero-score documents
  (article order) would otherwise make low article numbers look like hits.
- English queries against the de/fr/it indexes are cross-lingual by construction (a lexical
  baseline can only match shared tokens such as names, numbers and Latin terms); the same holds
  for any query language against a different-language index.

Output: a table on stdout and ``benchmark/baseline_dev.json``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from fedlex_bench.benchmark.bm25 import Bm25Index, build_indexes
from fedlex_bench.benchmark.common import (
    BENCHMARK_DIR,
    CORPUS_LANGS,
    QUERY_LANGS,
    SELECTED_DIR,
    load_corpus,
    read_jsonl,
    sha256_file,
)
from fedlex_bench.chunk import ARTICLES_PARQUET
from fedlex_bench.eligibility import retrieval_eligible

K = 10
INDEX_NAMES: tuple[str, ...] = (*CORPUS_LANGS, "all")
DEFAULT_OUT = BENCHMARK_DIR / "baseline_dev.json"
NOTE = (
    "English queries against the de/fr/it indexes are cross-lingual by construction: BM25 only "
    "matches tokens the question shares with the article (names, numbers, Latin terms). The same "
    "holds for a query in another language than the index. Documents with a zero score never "
    "count as hits."
)
_SEP = "#"


def build_all_indexes(df: pd.DataFrame) -> dict[str, Bm25Index]:
    """The de, fr and it indexes plus ``all``: one document per (article, language) with the id
    ``"<article_id>#<lang>"``."""
    indexes = build_indexes(df)
    live = retrieval_eligible(df)
    docs = {
        f"{r.article_id}{_SEP}{r.lang}": f"{r.title}\n{r.text}"
        for r in live.itertuples(index=False)
        if r.lang in CORPUS_LANGS
    }
    indexes["all"] = Bm25Index(docs)
    return indexes


def found_ids(index: Bm25Index, question: str, k: int = K) -> set[str]:
    """Article ids with a positively scored document in the top ``k``."""
    scores = index.scores(question)
    top = [d for d in index.top(question, k) if d in scores]
    return {d.split(_SEP, 1)[0] for d in top}


def item_recall(item: Mapping, indexes: Mapping[str, Bm25Index]) -> dict[str, float]:
    """Recall@10 of one answerable item against every index."""
    gold = list(item["gold_article_ids"])
    if not gold:
        raise ValueError(f"{item.get('id')}: recall needs gold ids (traps are excluded)")
    return {
        name: sum(g in found_ids(indexes[name], item["question"]) for g in gold) / len(gold)
        for name in INDEX_NAMES
    }


def _cell(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "recall": None}
    return {"n": len(values), "recall": round(sum(values) / len(values), 4)}


def summarise(
    items: list[dict], indexes: Mapping[str, Bm25Index], include_translations: bool = False
) -> dict:
    """Recall@10 per index and query language, and overall. Traps are counted and left out.
    Translations (``translation_of`` set) are left out of the table unless
    ``include_translations``; their number is reported either way (``n_translations``)."""
    n_translations = sum(1 for i in items if i.get("translation_of") and not i["is_trap"])
    if not include_translations:
        items = [i for i in items if not i.get("translation_of")]
    answerable = [i for i in items if not i["is_trap"]]
    per_item = [(i["query_lang"], item_recall(i, indexes)) for i in answerable]
    table: dict[str, dict] = {}
    for name in INDEX_NAMES:
        row = {lang: _cell([r[name] for q, r in per_item if q == lang]) for lang in QUERY_LANGS}
        row["overall"] = _cell([r[name] for _, r in per_item])
        table[name] = row
    return {
        "k": K,
        "n_items": len(answerable),
        "n_traps_excluded": len(items) - len(answerable),
        "n_translations": n_translations,
        "translations_included": include_translations,
        "recall_at_10": table,
    }


def format_table(summary: Mapping) -> str:
    rows = [("query lang", *INDEX_NAMES, "n")]
    table = summary["recall_at_10"]
    for label in (*QUERY_LANGS, "overall"):
        cells = [table[name][label] for name in INDEX_NAMES]
        rows.append(
            (
                label,
                *("-" if c["recall"] is None else f"{c['recall']:.3f}" for c in cells),
                str(cells[0]["n"]),
            )
        )
    widths = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
    lines = [
        "  ".join(
            v.rjust(w) if c else v.ljust(w) for c, (v, w) in enumerate(zip(r, widths, strict=True))
        )
        for r in rows
    ]
    return "BM25 recall@10, dev split, by index\n" + "\n".join(lines)


def run(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="benchmark baseline", description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="dev", help="only dev is allowed")
    ap.add_argument("--dir", type=Path, default=SELECTED_DIR, help="folder with dev.jsonl")
    ap.add_argument("--parquet", type=Path, default=ARTICLES_PARQUET)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--include-translations",
        action="store_true",
        help="count translated items (translation_of set) in the table; default: leave them out",
    )
    args = ap.parse_args(argv)
    if args.split != "dev":
        print(
            f"refused: the baseline runs on the dev split only, not the {args.split} split "
            "(the test split is never used for tuning or for development numbers)"
        )
        return 2
    dev_path = args.dir / "dev.jsonl"
    items = read_jsonl(dev_path)
    summary = summarise(
        items,
        build_all_indexes(load_corpus(args.parquet)),
        include_translations=args.include_translations,
    )
    summary.update(split="dev", dev_sha256=sha256_file(dev_path), note=NOTE)
    print(format_table(summary))
    print(
        f"{summary['n_items']} answerable items, {summary['n_traps_excluded']} traps excluded.\n"
        f"Note: {NOTE}"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(summary, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    args.out.write_bytes(text.encode("utf-8"))
    print(f"wrote {args.out}")
    return 0
