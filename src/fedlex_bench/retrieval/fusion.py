"""Reciprocal rank fusion and the merge of language copies into unique article ids."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from typing import TypeVar

from fedlex_bench.types import Candidate

RRF_K = 60

T = TypeVar("T", bound=Hashable)


def rrf(rankings: Sequence[Sequence[T]], k: int = RRF_K) -> list[tuple[T, float]]:
    """Reciprocal rank fusion: ``score(d) = sum over rankings of 1 / (k + rank)``, rank from 1.
    Best first. Ties keep the order of first appearance (ranking by ranking, top to bottom), so
    the output is deterministic."""
    scores: dict[T, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, 1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    order = {doc: i for i, doc in enumerate(scores)}  # dict keeps first-appearance order
    return sorted(scores.items(), key=lambda kv: (-kv[1], order[kv[0]]))


def merge_article_ids(items: Iterable[Candidate | str], k: int | None = None) -> list[str]:
    """Unique article ids from a ranked list of candidates (or plain ids): the first occurrence of
    an id wins, so a DE, FR or IT copy or any chunk of an article counts at its best position.
    ``k`` caps the number of ids returned."""
    seen: dict[str, None] = {}
    for it in items:
        aid = it if isinstance(it, str) else it.article_id
        if aid not in seen:
            seen[aid] = None
            if k is not None and len(seen) >= k:
                break
    return list(seen)
