"""Command line for the retrieval layer.

    uv run python -m fedlex_bench.retrieval build [--chunking article|paragraph|window|all]
        embed every eligible chunk once and cache the vectors in data/processed/index/
    uv run python -m fedlex_bench.retrieval query "text" --lang de [--method hybrid]
        [--config all_lang] [--chunking article] [--k 10]
    uv run python -m fedlex_bench.retrieval sanity [--methods ...] [--configs ...] [--chunking ...]
        recall@1/5/10 and MRR on benchmark/v1/dev.jsonl answerable items (dev only: this module
        has no option to read another split)

Embeddings need ~470 MB of model weights from the Hugging Face cache (downloaded on first use).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict

from fedlex_bench.ingest import PROJECT_ROOT
from fedlex_bench.retrieval.chunking import CHUNKINGS
from fedlex_bench.retrieval.index import (
    INDEX_CONFIGS,
    METHODS,
    RetrievalEnv,
    retrieve_articles,
)

DEV_PATH = PROJECT_ROOT / "benchmark" / "v1" / "dev.jsonl"  # the only split this tool reads
LIST_LENGTH = 20


def load_dev_items() -> list[dict]:
    with DEV_PATH.open(encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    return sorted((i for i in items if not i["is_trap"]), key=lambda i: i["id"])


def _score(ranked: list[str], gold: list[str]) -> tuple[dict[int, float], float]:
    pos = {a: r for r, a in enumerate(ranked, 1)}
    recall = {k: sum(1 for g in gold if pos.get(g, 10**9) <= k) / len(gold) for k in (1, 5, 10)}
    best = min((pos[g] for g in gold if g in pos), default=None)
    return recall, (1.0 / best if best else 0.0)


def sanity(env: RetrievalEnv, methods, configs, chunking: str) -> list[dict]:
    items = load_dev_items()
    rows = []
    for method in methods:
        for config in configs:
            r = env.retriever(method, config, chunking)
            agg: dict[str, list] = defaultdict(list)
            for it in items:
                ql = it["query_lang"]
                if config == "same_lang" and ql == "en":
                    continue
                ranked = retrieve_articles(r, it["question"], ql, LIST_LENGTH)
                rec, rr = _score(ranked, it["gold_article_ids"])
                agg["r1"].append(rec[1])
                agg["r5"].append(rec[5])
                agg["r10"].append(rec[10])
                agg["mrr"].append(rr)
            n = len(agg["mrr"])
            rows.append(
                {
                    "name": r.name,
                    "n": n,
                    **{m: round(sum(v) / n, 3) for m, v in agg.items()},
                }
            )
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="embed and cache all chunks")
    b.add_argument("--chunking", default="article", choices=[*CHUNKINGS, "all"])
    q = sub.add_parser("query")
    q.add_argument("text")
    q.add_argument("--lang", required=True, choices=["en", "de", "fr", "it"])
    q.add_argument("--method", default="hybrid", choices=METHODS)
    q.add_argument("--config", default="all_lang", choices=INDEX_CONFIGS)
    q.add_argument("--chunking", default="article", choices=CHUNKINGS)
    q.add_argument("--k", type=int, default=10)
    s = sub.add_parser("sanity", help="dev-set recall/MRR")
    s.add_argument("--methods", nargs="*", default=list(METHODS), choices=METHODS)
    s.add_argument("--configs", nargs="*", default=list(INDEX_CONFIGS), choices=INDEX_CONFIGS)
    s.add_argument("--chunking", default="article", choices=CHUNKINGS)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    env = RetrievalEnv()
    if args.cmd == "build":
        for ch in CHUNKINGS if args.chunking == "all" else (args.chunking,):
            env.warm(ch)
    elif args.cmd == "query":
        res = env.retriever(args.method, args.config, args.chunking).retrieve(
            args.text, args.lang, args.k
        )
        for c in res.candidates:
            print(f"{c.score:8.4f}  {c.chunk_id:<24} {c.lang}  {c.text[:80]!r}")
    else:
        for row in sanity(env, args.methods, args.configs, args.chunking):
            print(row)


if __name__ == "__main__":
    main()
