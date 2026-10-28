"""Choose hard negatives for the training pairs from the mined candidates, and write a review sheet.

``mine.py`` produced, for each question, two ranked lists of articles of the training pool (the
base model's ``emb`` and BM25's ``bm25``). Some of those are **false negatives**: articles that
also answer the question. ``pick_hard_negatives`` is the filter that decides which candidates are
safe to train against. The rest of this file is routine: it applies the filter to the training
split, counts what survives, and writes ``data/finetune/review_hardneg.md`` with the first N
examples (query, positive and chosen negatives with text) for a manual read before any training.

Run: ``uv run python -m fedlex_bench.finetune.hardneg [--show 20]``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from fedlex_bench.finetune.dataset import PAIRS_FILE, read_jsonl
from fedlex_bench.finetune.exclusions import FINETUNE_DIR
from fedlex_bench.finetune.mine import MINED_FILE
from fedlex_bench.finetune.synth import load_pool_texts, read_pool

HARDNEG_FILE = FINETUNE_DIR / "hardneg.jsonl"
REVIEW_FILE = FINETUNE_DIR / "review_hardneg.md"
MAX_NEGATIVES = 3  # kept per question, best first; training uses the first (more are spare)


def pick_hard_negatives(
    pair: dict, mined: dict, articles: dict[str, tuple[str, str]], k: int = MAX_NEGATIVES
) -> list[str]:
    """Return up to ``k`` article ids that are safe to use as hard negatives, best first.

    Inputs (all for ONE training question):
    - ``pair``: ``article_id`` (the positive), ``query``, ``query_lang``, ``style``,
      ``passage_lang``.
    - ``mined``: ``pos_score`` (cosine of the base model for the positive), ``pos_rank`` (its rank
      among the 3,201 pool articles), ``emb`` and ``bm25``: up to 30 ``[article_id, score]`` each,
      best first; the positive is never in them.
    - ``articles``: ``{article_id: (german title, german text)}`` for the whole pool, for text
      checks (the helpers ``char_ngrams`` / ``jaccard`` in ``fedlex_bench.benchmark.twins`` and
      ``id_key`` in ``fedlex_bench.benchmark.common`` may help).

    Numbers from the real data (12,800 questions): the base model's cosine is very compressed.
    The positive scores 0.874 on average, the best wrong article 0.892, the 30th-best wrong
    article 0.873. The positive is the base model's top-1 for only 20% of questions, top-10 for
    39% (legal_wording questions 76%, keywords 13%). BM25 and embedding top-10 share 0.44
    articles on average; 231 questions have no BM25 candidate at all.

    Failure modes to weigh: too loose lets false negatives through (training then pushes a correct
    article away); too strict leaves no hard negatives (they then teach nothing).
    """
    # TODO(human): implement the false-negative filter.
    raise NotImplementedError("pick_hard_negatives is the human-written filter")


def _snippet(text: str, n: int = 330) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n].rstrip() + " ..."


def write_review(rows: list[dict], pairs: dict, articles: dict, path: Path) -> None:
    out = ["# Hard-negative review sheet (German text shown)\n"]
    for r in rows:
        p = pairs[r["pair_id"]]
        pos_title, pos_text = articles[p["article_id"]]
        out.append(
            f"## {r['pair_id']}  [{p['query_lang']} / {p['style']}]  pos_rank {r['pos_rank']}"
        )
        out.append(f"**Q:** {p['query']}\n")
        out.append(f"**Positive {p['article_id']}** {pos_title}: {_snippet(pos_text)}\n")
        for a in r["negatives"]:
            title, text = articles[a]
            out.append(f"- **Negative {a}** {title}: {_snippet(text)}")
        if not r["negatives"]:
            out.append("- (no negative kept)")
        out.append("")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--show", type=int, default=20)
    parser.add_argument("--mined", type=Path, default=MINED_FILE)
    args = parser.parse_args()
    pairs = {r["pair_id"]: r for r in read_jsonl(PAIRS_FILE)}
    mined = read_jsonl(args.mined)
    pool = read_pool()
    german = load_pool_texts(pool)
    articles = {a: german[(a, "de")] for a in pool}
    rows = []
    for m in mined:
        p = pairs[m["pair_id"]]
        if p["split"] != "train":
            continue
        rows.append(
            {
                "pair_id": m["pair_id"],
                "pos_rank": m["pos_rank"],
                "negatives": pick_hard_negatives(p, m, articles),
            }
        )
    HARDNEG_FILE.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    kept = Counter(len(r["negatives"]) for r in rows)
    print(f"train questions {len(rows)}; negatives kept per question: {dict(sorted(kept.items()))}")
    # spread the review over the file instead of taking the first articles (all act 210)
    step = max(len(rows) // args.show, 1)
    write_review(rows[::step][: args.show], pairs, articles, REVIEW_FILE)
    print(f"review sheet: {REVIEW_FILE}")


if __name__ == "__main__":
    main()
