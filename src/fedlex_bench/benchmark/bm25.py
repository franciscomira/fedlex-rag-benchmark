"""BM25, own implementation (docs/benchmark_spec.md, section 6).

- Tokens: lowercase, ``\\w+`` runs with at least 2 characters, no stemming, no stop words.
- Parameters: ``k1 = 1.5``, ``b = 0.75``.
- IDF: ``ln(1 + (N - n + 0.5) / (n + 0.5))`` (never negative). The spec does not fix the IDF
  variant; this is the Lucene/BM25+ form.
- A query term that is repeated counts each time. Terms are summed in query order, so the floats
  are identical on every run.
- Rank of a document: 1 + the number of documents with a higher score + the number of documents
  with an equal score that come first in article order (natural order of ``"<SR>:<number>"``).
  The tie rule is deterministic and applies to the many zero-score documents of a cross-lingual
  query: they follow in article order. The rank of an item is the best (smallest) rank of its gold
  articles.
- A gold article that scores 0 (no shared token) has **no rank**: ``rank_scored`` returns
  ``None`` and ``bm25_rank[lang]`` is null. ``lexical_overlap`` is the rank in the question's own
  language index (null if that rank is null) and null for English questions (n/a).
- Index: every retrieval-eligible article of one language, one document per article, text of
  ``title + "\\n" + text``.

Nothing is cached on disk.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping

import pandas as pd

from fedlex_bench.benchmark.common import CORPUS_LANGS, id_key
from fedlex_bench.eligibility import retrieval_eligible

K1 = 1.5
B = 0.75
_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lowercased ``\\w+`` tokens of at least 2 characters, in text order."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 2]


class Bm25Index:
    """BM25 index over ``{doc_id: text}``. Documents are ordered by natural article order, which
    defines the tie-break (see the module docstring)."""

    def __init__(self, docs: Mapping[str, str], k1: float = K1, b: float = B) -> None:
        if not docs:
            raise ValueError("cannot build a BM25 index over zero documents")
        self.k1, self.b = k1, b
        self.ids: list[str] = sorted(docs, key=id_key)
        self._ordinal = {doc_id: i for i, doc_id in enumerate(self.ids)}
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._lengths: list[int] = []
        for i, doc_id in enumerate(self.ids):
            tokens = tokenize(docs[doc_id])
            self._lengths.append(len(tokens))
            for term, tf in sorted(Counter(tokens).items()):
                self._postings.setdefault(term, []).append((i, tf))
        self.n_docs = len(self.ids)
        self.avgdl = sum(self._lengths) / self.n_docs or 1.0

    def _idf(self, term: str) -> float:
        n = len(self._postings.get(term, ()))
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def scores(self, query: str) -> dict[str, float]:
        """Positive scores only: ``{doc_id: score}`` for every document sharing a term."""
        acc: dict[int, float] = {}
        for term in tokenize(query):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf(term)
            for i, tf in postings:
                norm = 1 - self.b + self.b * self._lengths[i] / self.avgdl
                acc[i] = acc.get(i, 0.0) + idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return {self.ids[i]: s for i, s in acc.items() if s > 0}

    def _ranked_positive(self, query: str) -> list[tuple[float, int]]:
        scored = self.scores(query)
        return sorted(
            ((s, self._ordinal[d]) for d, s in scored.items()), key=lambda x: (-x[0], x[1])
        )

    def top(self, query: str, k: int) -> list[str]:
        """The ``k`` best documents in rank order (zero-score documents follow in article order)."""
        ranked = [self.ids[i] for _, i in self._ranked_positive(query)]
        if len(ranked) < k:
            seen = set(ranked)
            ranked += [d for d in self.ids if d not in seen]
        return ranked[:k]

    def rank(self, query: str, gold_ids: Iterable[str]) -> int:
        """1-based rank of the best-ranked gold article."""
        gold = list(gold_ids)
        if not gold:
            raise ValueError("rank() needs at least one gold id")
        for g in gold:
            if g not in self._ordinal:
                raise KeyError(g)
        positive = self._ranked_positive(query)
        position = {ordinal: r for r, (_, ordinal) in enumerate(positive, 1)}
        positive_ordinals = sorted(position)
        best = None
        for g in gold:
            ordinal = self._ordinal[g]
            if ordinal in position:
                rank = position[ordinal]
            else:
                # zero score: after all positives, then in article order among the zero-score docs
                positives_before = _count_below(positive_ordinals, ordinal)
                rank = len(positive) + 1 + ordinal - positives_before
            best = rank if best is None else min(best, rank)
        assert best is not None
        return best

    def rank_scored(self, query: str, gold_ids: Iterable[str]) -> int | None:
        """Like :meth:`rank`, but ``None`` when no gold article has a positive score. A zero score
        means the question shares no token with the article: there is no rank to report, and the
        article-order position of a zero-score document would be an artefact."""
        gold = list(gold_ids)
        scores = self.scores(query)
        if not any(g in scores for g in gold if g in self._ordinal):
            for g in gold:
                if g not in self._ordinal:
                    raise KeyError(g)
            if not gold:
                raise ValueError("rank_scored() needs at least one gold id")
            return None
        scored = [g for g in gold if g in scores]
        return self.rank(query, scored)


def _count_below(sorted_values: list[int], value: int) -> int:
    lo, hi = 0, len(sorted_values)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_values[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


def build_indexes(df: pd.DataFrame) -> dict[str, Bm25Index]:
    """One index per corpus language over the retrieval-eligible articles of ``df``."""
    live = retrieval_eligible(df)
    out: dict[str, Bm25Index] = {}
    for lang in CORPUS_LANGS:
        sub = live[live["lang"] == lang]
        docs = {r.article_id: f"{r.title}\n{r.text}" for r in sub.itertuples(index=False)}
        out[lang] = Bm25Index(docs)
    return out


def lexical_overlap(ranks: Mapping[str, int | None], query_lang: str) -> int | None:
    """``bm25_rank`` of the question's own language index (``None`` if that rank is null). For an
    English question there is no own index: ``None`` (n/a); the three ranks are still stored."""
    if query_lang == "en":
        return None
    return ranks[query_lang]


def compute_ranks(
    question: str,
    gold_ids: Iterable[str],
    query_lang: str,
    indexes: Mapping[str, Bm25Index],
) -> tuple[dict[str, int | None], int | None]:
    """``(bm25_rank per language, lexical_overlap)`` of a question against its gold articles. A
    rank is ``None`` where the best gold article scores 0 (no shared token)."""
    gold = list(gold_ids)
    ranks = {lang: indexes[lang].rank_scored(question, gold) for lang in CORPUS_LANGS}
    return ranks, lexical_overlap(ranks, query_lang)
