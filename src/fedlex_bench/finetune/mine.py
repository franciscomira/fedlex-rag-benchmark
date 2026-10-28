"""Mine hard-negative **candidates** for every training question. No filtering happens here.

For each pair, two ranked lists of up to ``DEPTH`` articles of the training pool, best first:

- ``emb``: the base multilingual-e5-small (the model we will fine-tune), cosine of the question
  against each article's best language version;
- ``bm25``: the project's BM25 over the all-language article index.

Only articles of the training pool can appear (dev/test gold, their neighbours and twins are never
shown to the model, not even as negatives), and the positive article itself is removed (all its
language versions are the same article). For each question we also store how the positive scores
(``pos_score``: its best language version; ``pos_rank``: its rank among pool articles by the
embedding), because the false-negative filter needs the positive as a reference.

Output ``data/finetune/mined.jsonl`` (git-ignored, ~15 MB). Run:
``uv run python -m fedlex_bench.finetune.mine [--limit N]``.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from fedlex_bench.finetune.dataset import PAIRS_FILE, read_jsonl
from fedlex_bench.finetune.exclusions import FINETUNE_DIR
from fedlex_bench.finetune.synth import LANGS, read_pool
from fedlex_bench.retrieval.index import RetrievalEnv

DEPTH = 30
BATCH = 256
MINED_FILE = FINETUNE_DIR / "mined.jsonl"


def _article_matrix_layout(index, pool: list[str]):
    """Positions of each pool article's de/fr/it chunk rows in ``index.matrix``."""
    where = {(c.article_id, c.lang): i for i, c in enumerate(index.chunks)}
    return np.array([[where[(a, lang)] for a in pool] for lang in LANGS])  # (3, n_pool)


def mine(limit: int = 0) -> list[dict]:
    pairs = read_jsonl(PAIRS_FILE)
    if limit:
        pairs = pairs[:: max(len(pairs) // limit, 1)][:limit]
    pool = read_pool()
    pos_of = {a: i for i, a in enumerate(pool)}
    env = RetrievalEnv()
    index = env.index("article", "all")
    rows_by_lang = _article_matrix_layout(index, pool)  # (3, n_pool)
    bm25 = index.bm25
    chunk_article = [c.article_id for c in index.chunks]
    pool_set = set(pool)
    out: list[dict] = []
    t0 = time.perf_counter()
    for start in range(0, len(pairs), BATCH):
        batch = pairs[start : start + BATCH]
        q = env.embedder.embed_queries([p["query"] for p in batch])
        sims = np.stack([q @ index.matrix[rows_by_lang[k]].T for k in range(len(LANGS))])  # (3,B,n)
        best = sims.max(axis=0)  # (B, n_pool): best language version per article
        for b, p in enumerate(batch):
            pi = pos_of[p["article_id"]]
            pos_score = float(best[b, pi])
            pos_rank = int((best[b] > pos_score).sum()) + 1
            masked = best[b].copy()
            masked[pi] = -np.inf
            top = np.argsort(-masked, kind="stable")[:DEPTH]
            emb = [[pool[i], round(float(masked[i]), 4)] for i in top]
            lex: dict[str, float] = {}
            for key, score in sorted(bm25.scores(p["query"]).items(), key=lambda kv: -kv[1]):
                art = chunk_article[index._pos[key]]
                if art in pool_set and art != p["article_id"] and art not in lex:
                    lex[art] = round(score, 3)
                    if len(lex) == DEPTH:
                        break
            out.append(
                {
                    "pair_id": p["pair_id"],
                    "pos_score": round(pos_score, 4),
                    "pos_rank": pos_rank,
                    "emb": emb,
                    "bm25": [[a, s] for a, s in lex.items()],
                }
            )
        print(f"{start + len(batch)}/{len(pairs)}  {time.perf_counter() - t0:.0f}s", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    rows = mine(args.limit)
    path = MINED_FILE if not args.limit else FINETUNE_DIR / "mined_sample.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    ranks = np.array([r["pos_rank"] for r in rows])
    print(
        f"{len(rows)} questions -> {path.name}; base-model rank of the positive among pool "
        f"articles: top1 {np.mean(ranks == 1):.2f}, top10 {np.mean(ranks <= 10):.2f}, "
        f"top30 {np.mean(ranks <= 30):.2f}"
    )


if __name__ == "__main__":
    main()
