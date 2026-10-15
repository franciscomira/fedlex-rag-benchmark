"""Article ids: parsing, ordering and range-id resolution.

An article id is ``"<SR>:<number>"`` (e.g. ``"220:257d"``). It is the alignment key across DE, FR
and IT (see :mod:`fedlex_bench.chunk`). This module is pure code (no I/O, no randomness): citations
and ids are never produced by the model.

Number grammar (canonical form, as made by ``ingest.normalize_article_number``)::

    number  := [block "/"] article ["-" article]  |  block
    block   := "u" digits            transitional/final-provision block, e.g. "u12"
    article := digits letters*       "257", "257d", "6bis", "6abis", "697l"

So ``220:257d`` is an article, ``220:u12/3`` is article 3 of transitional block 12, ``220:u12`` is a
whole block chunked without articles, and ``210:135-149`` is a **range id**.

Range ids
---------
Fedlex collapses runs of repealed articles into one element (e.g. SR 210 "Art. 135-149
aufgehoben"). The chunker keeps that as a single, text-less, repealed chunk whose id is the range
(42 such ids per language in the Phase 0 corpus). A lookup of ``210:140`` therefore finds nothing by
exact match; :func:`resolve_id` maps it to ``210:135-149``.

Ordering of suffixed numbers (shared with :func:`fedlex_bench.chunk.natural_key`)
-------------------------------------------------------------------------------
Numbers sort by their integer part first, then by suffix: the bare number, then single letters
(``a < b < ... < z``), then Latin multiplicatives by ordinal (``bis < ter < quater < ...``). Letters
before multiplicatives, and a longer suffix after its prefix::

    6 < 6a < 6abis < 6b < 6bis < 6ter < 6quater < 7 < 10 < 257 < 257a < 257d < 257e < 257bis < 258

A range ``lo-hi`` covers every number ``x`` of the same act and block with ``lo <= x <= hi`` in this
order (both ends inclusive). Consequences, all deliberate:

- ``226a-226d`` covers 226a, 226b, 226c, 226d, but not the bare 226.
- ``274-274g`` starts at the bare 274 and covers 274a..274g, but not 274bis (multiplicatives sort
  after all single letters).
- ``135-149`` also covers ``140a`` and ``140bis`` (they sort between 140 and 141) but not ``149a``.
  :func:`expand_range` lists only integers and single-letter suffixes; :func:`range_covers` and
  :func:`resolve_id` use the full order.
- Ranges never cross acts or blocks: ``220:3`` is not in ``220:u12/2-4``.

What happens when an id cannot be resolved
------------------------------------------
:func:`resolve_id` returns ``None`` (never a guess) when the query is not a well-formed id, when no
known id and no range covers it, when it is itself a range or a block-only id and not an exact
known id, or when **two or more** known ranges cover it (ambiguous; overlapping ranges would be a
data problem and are reported by the corpus-level test). An exact match always wins.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import NamedTuple

from fedlex_bench.chunk import natural_key
from fedlex_bench.ingest import normalize_article_number

__all__ = [
    "ArticleIdError",
    "IdResolver",
    "NumberSpec",
    "expand_range",
    "is_range_id",
    "normalize_article_id",
    "number_key",
    "parse_article_id",
    "range_covers",
    "resolve_id",
    "split_number",
]


class ArticleIdError(ValueError):
    """A string is not a well-formed canonical article id."""


class NumberSpec(NamedTuple):
    """The number part of an id. ``block`` is ``""`` for main text or ``"u12"``; ``start`` is the
    article number (``""`` for a block-only id such as ``u3``); ``end`` is the range end or None."""

    block: str
    start: str
    end: str | None


_SR_RE = re.compile(r"\d+(?:\.\d+)*")
_ARTICLE = r"\d+[a-z]*"
_BLOCK_ONLY_RE = re.compile(r"(u\d+)")
_NUMBER_RE = re.compile(rf"(?:(u\d+)/)?({_ARTICLE})(?:-({_ARTICLE}))?")
_SPLIT_RE = re.compile(r"(\d+)([a-z]*)")


def number_key(number: str) -> tuple:
    """Sort key of a bare article number (no block, no range): see the module docstring."""
    return natural_key(number)


def split_number(number: str) -> NumberSpec:
    """Split the number part of an id into block, start and end. Raises :class:`ArticleIdError`
    for anything outside the grammar, including descending or empty ranges (``149-135``,
    ``5-5``)."""
    m = _BLOCK_ONLY_RE.fullmatch(number)
    if m:
        return NumberSpec(m.group(1), "", None)
    m = _NUMBER_RE.fullmatch(number)
    if not m:
        raise ArticleIdError(f"malformed article number {number!r}")
    block, start, end = m.group(1) or "", m.group(2), m.group(3)
    if end is not None and not number_key(start) < number_key(end):
        raise ArticleIdError(f"range {number!r} is empty or descending")
    return NumberSpec(block, start, end)


def parse_article_id(article_id: str) -> tuple[str, str]:
    """``"220:257d"`` -> ``("220", "257d")``. Strict: the id must already be canonical (see
    :func:`normalize_article_id`). Raises :class:`ArticleIdError` otherwise."""
    sr, sep, number = article_id.partition(":")
    if not sep or not _SR_RE.fullmatch(sr):
        raise ArticleIdError(f"malformed article id {article_id!r}: expected '<SR>:<number>'")
    split_number(number)  # validates
    return sr, number


def normalize_article_id(raw: str) -> str:
    """Canonical form of a user/citation-style id: ``"220:Art. 257 d"`` -> ``"220:257d"``,
    ``"210:135 à 149"`` -> ``"210:135-149"``. Idempotent on canonical ids. Raises
    :class:`ArticleIdError` when the result is not a well-formed id."""
    sr, sep, number = raw.partition(":")
    if not sep:
        raise ArticleIdError(f"malformed article id {raw!r}: expected '<SR>:<number>'")
    canonical = f"{sr.strip()}:{normalize_article_number(number)}"
    parse_article_id(canonical)
    return canonical


def is_range_id(article_id: str) -> bool:
    """True for a well-formed range id such as ``210:135-149``; False for anything else."""
    try:
        _, number = parse_article_id(article_id)
        return split_number(number).end is not None
    except ArticleIdError:
        return False


def range_covers(range_id: str, article_id: str) -> bool:
    """True when ``article_id`` is an article of the same act and block that lies within
    ``range_id`` (inclusive, natural order). A range or block-only id is never covered. Both
    arguments must be well-formed (:class:`ArticleIdError` otherwise); ``range_id`` that is not a
    range gives False."""
    r_sr, r_number = parse_article_id(range_id)
    r = split_number(r_number)
    a_sr, a_number = parse_article_id(article_id)
    a = split_number(a_number)
    if r.end is None or a.end is not None or not a.start:
        return False
    if (r_sr, r.block) != (a_sr, a.block):
        return False
    return number_key(r.start) <= number_key(a.start) <= number_key(r.end)


def _int_and_suffix(article: str) -> tuple[int, str]:
    m = _SPLIT_RE.fullmatch(article)
    assert m is not None  # callers pass validated numbers
    return int(m.group(1)), m.group(2)


def _is_letter(suffix: str) -> bool:
    return len(suffix) == 1 and suffix.isalpha()


def expand_range(range_id: str) -> list[str]:
    """List the ids a range id stands for, in order: all integers of ``135-149``; the letters of
    ``226a-226d``; the bare number plus letters of ``274-274g``.

    Only these shapes are enumerable (integers, single-letter suffixes within one integer).
    Latin-multiplicative or mixed shapes (``5a-7``, ``5-7bis``) raise :class:`ArticleIdError`
    rather than inventing members. Numbers that :func:`range_covers` accepts but that are not
    listed here (``140a`` in ``135-149``) are still covered. Members are what the range *would*
    contain, not a promise that each exists as an article."""
    sr, number = parse_article_id(range_id)
    spec = split_number(number)
    if spec.end is None:
        raise ArticleIdError(f"{range_id!r} is not a range id")
    prefix = f"{sr}:{spec.block + '/' if spec.block else ''}"
    (lo_n, lo_s), (hi_n, hi_s) = _int_and_suffix(spec.start), _int_and_suffix(spec.end)
    if lo_n == hi_n:
        if (lo_s and not _is_letter(lo_s)) or not _is_letter(hi_s):
            raise ArticleIdError(f"cannot enumerate {range_id!r}")
        first = lo_s or "a"
        letters = [chr(c) for c in range(ord(first), ord(hi_s) + 1)]
        members = ([""] if not lo_s else []) + letters
        return [f"{prefix}{lo_n}{s}" for s in members]
    if lo_s or hi_s:
        raise ArticleIdError(f"cannot enumerate {range_id!r}")
    return [f"{prefix}{n}" for n in range(lo_n, hi_n + 1)]


class IdResolver:
    """Resolve ids against a fixed set of known chunk ids; build once, query many times.

    ``skipped`` lists known ids that look like ranges (a ``-`` in the number) but are malformed
    and therefore cannot be used for range resolution (reported, not silently dropped)."""

    def __init__(self, known_ids: Iterable[str]) -> None:
        known = known_ids if isinstance(known_ids, set | frozenset) else frozenset(known_ids)
        self._known = known
        self._ranges: dict[tuple[str, str], list[tuple[tuple, tuple, str]]] = {}
        self.skipped: list[str] = []
        for kid in sorted(k for k in known if isinstance(k, str)):
            if "-" not in kid.partition(":")[2]:
                continue
            try:
                sr, number = parse_article_id(kid)
                spec = split_number(number)
            except ArticleIdError:
                self.skipped.append(kid)
                continue
            if spec.end is not None:
                entry = (number_key(spec.start), number_key(spec.end), kid)
                self._ranges.setdefault((sr, spec.block), []).append(entry)

    def resolve(self, article_id: str) -> str | None:
        """The known chunk id for ``article_id``, or None (see the module docstring)."""
        try:
            canonical = normalize_article_id(article_id)
            sr, number = parse_article_id(canonical)
            spec = split_number(number)
        except ArticleIdError:
            return None
        if canonical in self._known:
            return canonical
        if spec.end is not None or not spec.start:
            return None  # a range or block-only query matches only exactly
        key = number_key(spec.start)
        hits = [rid for lo, hi, rid in self._ranges.get((sr, spec.block), ()) if lo <= key <= hi]
        return hits[0] if len(hits) == 1 else None


def resolve_id(article_id: str, known_ids: Iterable[str]) -> str | None:
    """Map an article id to the known chunk id that holds it: itself if known, else the single
    known range id that covers it, else None. ``known_ids`` may be any iterable of ids (consumed
    once); for many lookups build an :class:`IdResolver` instead."""
    try:
        canonical = normalize_article_id(article_id)
    except ArticleIdError:
        return None
    if isinstance(known_ids, set | frozenset) and canonical in known_ids:
        return canonical
    return IdResolver(known_ids).resolve(canonical)
