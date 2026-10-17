"""Pure-code metrics (docs/eval_spec.md sections 2, 3, 4). No model calls."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from fedlex_bench.types import AnswerResult


def dedupe_ids(ids: Iterable[str]) -> list[str]:
    """Unique article ids, first occurrence wins (merges language copies)."""
    seen: set[str] = set()
    out: list[str] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def recall_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Fraction of gold articles within the first k unique ids."""
    top = set(dedupe_ids(ranked)[:k])
    return sum(g in top for g in gold) / len(gold)


def reciprocal_rank(ranked: Sequence[str], gold: Sequence[str]) -> float:
    gold_set = set(gold)
    for rank, article_id in enumerate(dedupe_ids(ranked), start=1):
        if article_id in gold_set:
            return 1.0 / rank
    return 0.0


def citation_precision_recall(
    citations: Sequence[str], gold: Sequence[str]
) -> tuple[float, float]:
    """Article-level precision and recall of valid citations; 0 for both if none cited."""
    cited = set(citations)
    if not cited:
        return 0.0, 0.0
    hits = len(cited & set(gold))
    return hits / len(cited), hits / len(set(gold))


def trap_code_checks(answer: AnswerResult) -> bool:
    """Criteria 1 and 2 of the trap rule: refused and no citations. Criterion 3 is the judge's."""
    return (
        (not answer.parse_failed)
        and answer.status == "refused"
        and not answer.citations
        and not answer.invalid_citations  # a made-up id is still a citation attempt
    )
