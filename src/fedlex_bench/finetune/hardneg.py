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

from fedlex_bench.benchmark.common import id_key
from fedlex_bench.benchmark.twins import char_ngrams, jaccard, normalise_text
from fedlex_bench.finetune.dataset import PAIRS_FILE, _h, read_jsonl
from fedlex_bench.finetune.exclusions import FINETUNE_DIR
from fedlex_bench.finetune.mine import MINED_FILE
from fedlex_bench.finetune.synth import load_pool_texts, read_pool

HARDNEG_FILE = FINETUNE_DIR / "hardneg.jsonl"
REVIEW_FILE = FINETUNE_DIR / "review_hardneg.md"
MAX_NEGATIVES = 3  # kept per question, best first; training uses the first (more are spare)
NEIGHBOUR_DISTANCE = 3  # same act, within this many pool articles of the positive: dropped
JACCARD_MAX = 0.3  # character 5-gram Jaccard with the positive at or above this: dropped
AMBIGUITY_ZONE = 3  # embedding ranks skipped when the base model ranks the positive below them
MIN_CHARS = 100  # candidates with shorter German text are dropped (unless the positive is as short)
HUB_CAP = 25  # an article is chosen as a negative for at most this many questions

_ORDER_CACHE: dict[int, dict[str, int]] = {}
_GRAMS_CACHE: dict[int, dict[str, set[str]]] = {}


def _pool_order(articles: dict) -> dict[str, int]:
    """Position of every article in natural id order (cached per ``articles`` object)."""
    if id(articles) not in _ORDER_CACHE:
        _ORDER_CACHE[id(articles)] = {a: i for i, a in enumerate(sorted(articles, key=id_key))}
    return _ORDER_CACHE[id(articles)]


def _pool_grams(articles: dict) -> dict[str, set[str]]:
    if id(articles) not in _GRAMS_CACHE:
        _GRAMS_CACHE[id(articles)] = {
            a: char_ngrams(normalise_text(text)) for a, (_, text) in articles.items()
        }
    return _GRAMS_CACHE[id(articles)]


def pick_hard_negatives(
    pair: dict,
    mined: dict,
    articles: dict[str, tuple[str, str]],
    k: int = MAX_NEGATIVES,
    used: Counter | None = None,
) -> list[str]:
    """Return up to ``k`` article ids that are safe to use as hard negatives, best first.

    Inputs (all for ONE training question):
    - ``pair``: ``article_id`` (the positive), ``query``, ``query_lang``, ``style``,
      ``passage_lang``.
    - ``mined``: ``pos_score`` (cosine of the base model for the positive), ``pos_rank`` (its rank
      among the 3,201 pool articles), ``emb`` and ``bm25``: up to 30 ``[article_id, score]`` each,
      best first; the positive is never in them.
    - ``used``: how often each article has already been chosen (``None``: no cap). The caller
      owns it; this function adds the articles it returns.
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
    # DRAFT written by Claude (Francisco asked for a draft to review, 2026-10-28). Francisco chose
    # to add rules 1 and 2 after the first review: hub cap and minimum length.
    # TODO(human): review the rule and the thresholds above; change what you disagree with.
    positive = pair["article_id"]
    order, grams = _pool_order(articles), _pool_grams(articles)

    min_chars = min(MIN_CHARS, len(articles[positive][1]))

    def too_short(candidate: str) -> bool:
        return len(articles[candidate][1]) < min_chars

    def at_cap(candidate: str) -> bool:
        return used is not None and used[candidate] >= HUB_CAP

    def too_close(candidate: str) -> bool:
        """Structural near-copy of the positive: adjacent article, or largely the same text."""
        if candidate.partition(":")[0] == positive.partition(":")[0]:
            if abs(order[candidate] - order[positive]) <= NEIGHBOUR_DISTANCE:
                return True
        return jaccard(grams[candidate], grams[positive]) >= JACCARD_MAX

    # The base model scored these above the labelled positive: the likeliest false negatives.
    skip = 0 if mined["pos_rank"] == 1 else AMBIGUITY_ZONE
    emb = [a for a, _ in mined["emb"][skip:]]
    in_both = set(emb) & {a for a, _ in mined["bm25"]}
    safe = [a for a in emb if not too_short(a) and not at_cap(a) and not too_close(a)]
    # real confusions (close in meaning and in words) first, then the rest in embedding order
    safe.sort(key=lambda a: a not in in_both)  # stable: keeps embedding order inside each group
    chosen = safe[:k]
    if used is not None:
        used.update(chosen)
    return chosen


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
    # The hub cap is first come, first served, so go through the questions in a seeded random
    # order (not file order, which would favour act 210), and write the result in file order.
    todo = [m for m in mined if pairs[m["pair_id"]]["split"] == "train"]
    chosen: dict[str, list[str]] = {}
    used: Counter = Counter()
    for m in sorted(todo, key=lambda m: _h("hardneg", m["pair_id"])):
        chosen[m["pair_id"]] = pick_hard_negatives(pairs[m["pair_id"]], m, articles, used=used)
    rows = [
        {"pair_id": m["pair_id"], "pos_rank": m["pos_rank"], "negatives": chosen[m["pair_id"]]}
        for m in todo
    ]
    HARDNEG_FILE.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    kept = Counter(len(r["negatives"]) for r in rows)
    print(f"train questions {len(rows)}; negatives kept per question: {dict(sorted(kept.items()))}")
    first = Counter(r["negatives"][0] for r in rows if r["negatives"])
    top100 = sum(v for _, v in first.most_common(100)) / max(sum(first.values()), 1)
    lengths = sorted(len(articles[r["negatives"][0]][1]) for r in rows if r["negatives"])
    print(
        f"distinct first negatives {len(first)}; top-100 articles fill {top100:.2f}; "
        f"max uses {max(used.values())}; median length of first negative "
        f"{lengths[len(lengths) // 2]} chars"
    )
    # spread the review over the file instead of taking the first articles (all act 210)
    step = max(len(rows) // args.show, 1)
    write_review(rows[::step][: args.show], pairs, articles, REVIEW_FILE)
    print(f"review sheet: {REVIEW_FILE}")


if __name__ == "__main__":
    main()
