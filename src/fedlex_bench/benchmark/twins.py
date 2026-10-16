"""Twin articles: pairs of articles with (almost) the same text.

Some statutes repeat a rule word for word in neighbouring articles (the same rule for two kinds of
company, say). A question written from one of them is also answered by the other, so the benchmark
would have two correct answers. ``plan`` therefore removes every article whose German text has a
character 5-gram Jaccard similarity of at least 0.7 with another article of the pool, both as
primary and as secondary gold. (This is not part of ``eligibility.py``: the articles stay in the
retrieval corpus.)

Similarity: Jaccard of the sets of character 5-grams of the normalised text (lowercase, letters
and digits, single spaces; a text shorter than 5 characters is its own single gram). All pairs
are compared exactly; the search uses prefix filtering (a pair with Jaccard >= t shares a gram
among the ``len - ceil(t * len) + 1`` rarest grams of each), which finds every such pair without
the quadratic cost.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from fractions import Fraction

from fedlex_bench.benchmark.common import id_key

THRESHOLD = 0.7


def normalise_text(text: str) -> str:
    """Lowercase, letters and digits only, single spaces."""
    return " ".join(re.sub(r"[\W_]+", " ", text.lower()).split())


def char_ngrams(text: str, n: int = 5) -> set[str]:
    return {text[i : i + n] for i in range(max(0, len(text) - n + 1))} or {text}


def jaccard(a: set[str], b: set[str]) -> Fraction:
    union = len(a | b)
    return Fraction(len(a & b), union) if union else Fraction(1)


def find_twins(texts: dict[str, str], threshold: float = THRESHOLD) -> dict[str, tuple[str, float]]:
    """``{article_id: (twin_id, jaccard)}`` for every article that has a partner with a Jaccard of
    at least ``threshold``. The twin is the partner with the highest similarity (ties: the lowest
    article id in natural order)."""
    t = Fraction(str(threshold))
    ids = sorted(texts, key=id_key)
    grams = [char_ngrams(normalise_text(texts[i])) for i in ids]
    frequency: Counter = Counter()
    for g in grams:
        frequency.update(g)
    ordered = [sorted(g, key=lambda x: (frequency[x], x)) for g in grams]

    num, den = t.numerator, t.denominator
    sizes = [len(g) for g in grams]
    best: dict[int, tuple[Fraction, int]] = {}

    def offer(a: int, b: int, shared: int, union: int) -> None:
        if a not in best:
            best[a] = (Fraction(shared, union), b)
            return
        j = Fraction(shared, union)
        if (j, -b) > (best[a][0], -best[a][1]):
            best[a] = (j, b)

    # Probe each document against the prefixes of the smaller (or equal) ones indexed before it.
    prefix_index: dict[str, list[int]] = {}
    for a in sorted(range(len(ids)), key=lambda n: (sizes[n], n)):
        size_a = sizes[a]
        prefix = ordered[a][: size_a - math.ceil(t * size_a) + 1]
        seen: set[int] = set()
        for g in prefix:
            for b in prefix_index.get(g, ()):
                if b in seen:
                    continue
                seen.add(b)
                size_b = sizes[b]
                if size_a * den < num * size_b:
                    continue  # b is too small to reach the threshold with a
                shared = len(grams[a] & grams[b])
                union = size_a + size_b - shared
                if shared * den < num * union:
                    continue
                offer(a, b, shared, union)
                offer(b, a, shared, union)
        for g in prefix:
            prefix_index.setdefault(g, []).append(a)
    return {ids[a]: (ids[b], float(j)) for a, (j, b) in best.items()}


def twin_records(found: dict[str, tuple[str, float]]) -> list[dict]:
    """Rows for ``twins.jsonl``: article id, twin id, Jaccard (4 decimals), in natural order."""
    return [
        {"article_id": a, "twin_id": found[a][0], "jaccard": round(found[a][1], 4)}
        for a in sorted(found, key=id_key)
    ]
