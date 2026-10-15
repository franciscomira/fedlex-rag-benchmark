"""Article ids: parsing, ordering and range-id resolution.

An article id is ``"<SR>:<number>"`` (e.g. ``"220:257d"``). It is the alignment key across DE, FR
and IT (see :mod:`fedlex_bench.chunk`). This module is pure code (no I/O, no randomness): citations
and ids are never produced by the model.

Number grammar (canonical form, as made by ``ingest.normalize_article_number``)::

    number  := [block "/"] article ["-" article]  |  block
    block   := "u" digits            transitional/final-provision block, e.g. "u12"
    article := digits suffix?
    suffix  := letter | ordinal | letter ordinal
    letter  := "a" .. "z"            exactly one
    ordinal := "bis" | "ter" | "quater" | ... | "vicies"   (``ingest.LATIN_ORDINAL``)

So ``220:257d``, ``220:6bis``, ``220:6abis`` and ``220:12ater`` are articles. Anything else after
the digits (``135ff``, ``140xyz``, ``5aa``, ``5bisbis``) is malformed, not a number.
``220:u12/3`` is article 3 of transitional block 12, ``220:u12`` is a whole block chunked
without articles, and ``210:135-149`` is a **range id**.

Range ids
---------
Fedlex collapses runs of repealed articles into one element (e.g. SR 210 "Art. 135-149
aufgehoben"). The chunker keeps that as a single, text-less, repealed chunk whose id is the range
(42 such ids per language in the Phase 0 corpus). A lookup of ``210:140`` therefore finds nothing by
exact match; :func:`resolve_id` maps it to ``210:135-149``. The eId of Fedlex does not tell a range
("135 bis 149") from a conjunction ("50 und 51"); in the corpus all conjunctions are adjacent
pairs, so both are covered the same way here.

Ordering of suffixed numbers (shared with :func:`fedlex_bench.chunk.natural_key`)
-------------------------------------------------------------------------------
Numbers sort by their integer part first, then by suffix: the bare number, then single letters
(``a < b < ... < z``), then Latin multiplicatives by ordinal (``bis < ter < quater < ...``). Letters
before multiplicatives, and a longer suffix after its prefix::

    6 < 6a < 6abis < 6b < 6bis < 6ter < 6quater < 7 < 10 < 257 < 257a < 257d < 257e < 257bis < 258

This order is used for validating ranges (``lo < hi``) and for sorting. What a range **covers** is
deliberately narrower than the order span, so that nothing is guessed:

- An integer range ``135-149`` covers the bare integers 135..149 only. It does not cover
  ``140a``, ``140z`` or ``140bis``: the repealed block does not say whether an inserted 140a
  belongs to it, so such an id resolves to ``None`` unless it is a chunk of its own.
- A letter range ``226a-226d`` covers 226a..226d (single letters of one integer), not the bare
  226 and not 226bis.
- ``274-274g`` starts at the bare 274 and covers 274, 274a..274g, but not 274bis.
- Range shapes that cannot be enumerated (``5a-7``, ``5-7bis``) cover only their two end points.
- Ranges never cross acts or blocks: ``220:3`` is not in ``220:u12/2-4``.

:func:`expand_range` lists exactly what a range covers.

What happens when an id cannot be resolved
------------------------------------------
:func:`resolve_id` returns ``None`` (never a guess) when the query is not a well-formed id, when no
known id and no range covers it, when it is itself a range or a block-only id and not an exact
known id, or when **two or more** known ranges cover it (ambiguous; overlapping ranges would be a
data problem and are reported by the corpus-level test). An exact match always wins. Passing a
single string as ``known_ids`` is a ``TypeError``. Resolving an id says nothing about whether the
article may be used as a gold label; see :func:`fedlex_bench.eligibility.resolve_gold`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import NamedTuple

from fedlex_bench.chunk import natural_key
from fedlex_bench.ingest import LATIN_ORDINAL, normalize_article_number

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
_ORDINAL = "|".join(sorted(LATIN_ORDINAL, key=len, reverse=True))
_ARTICLE = rf"\d+(?:[a-z]?(?:{_ORDINAL})|[a-z])?"
_BLOCK_ONLY_RE = re.compile(r"(u\d+)")
_NUMBER_RE = re.compile(rf"(?:(u\d+)/)?({_ARTICLE})(?:-({_ARTICLE}))?")
_SPLIT_RE = re.compile(r"(\d+)([a-z]*)")


def number_key(number: str) -> tuple:
    """Sort key of a bare article number (no block, no range): see the module docstring."""
    return natural_key(number)


def split_number(number: str) -> NumberSpec:
    """Split the number part of an id into block, start and end. Raises :class:`ArticleIdError`
    for anything outside the grammar (``135ff``, ``140xyz``), including descending or empty
    ranges (``149-135``, ``5-5``)."""
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


def _covers(lo: str, hi: str, x: str) -> bool:
    """Does the range of article numbers ``lo``..``hi`` cover article number ``x``? See the module
    docstring: integers of an integer range, single letters (plus the bare number when ``lo`` is
    bare) of a same-integer range, otherwise only the end points. ``lo < hi`` is assumed."""
    (ln, ls), (hn, hs), (xn, xs) = _int_and_suffix(lo), _int_and_suffix(hi), _int_and_suffix(x)
    if ln == hn:
        if (ls and not _is_letter(ls)) or not _is_letter(hs):
            return x in (lo, hi)
        if xn != ln:
            return False
        if not xs:
            return not ls  # the bare number only when the range starts at it
        return _is_letter(xs) and (ls or "a") <= xs <= hs
    if ls or hs:
        return x in (lo, hi)
    return ln <= xn <= hn and not xs


def range_covers(range_id: str, article_id: str) -> bool:
    """True when ``article_id`` is an article of the same act and block that ``range_id`` covers
    (inclusive; exactly the members of :func:`expand_range`, see the module docstring). A range
    or block-only id is never covered. Both arguments must be well-formed
    (:class:`ArticleIdError` otherwise); a ``range_id`` that is not a range gives False."""
    r_sr, r_number = parse_article_id(range_id)
    r = split_number(r_number)
    a_sr, a_number = parse_article_id(article_id)
    a = split_number(a_number)
    if r.end is None or a.end is not None or not a.start:
        return False
    if (r_sr, r.block) != (a_sr, a.block):
        return False
    return _covers(r.start, r.end, a.start)


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
    rather than inventing members. The list is exactly what :func:`range_covers` accepts (apart
    from the end points of non-enumerable shapes). Members are what the range *would* contain,
    not a promise that each exists as an article."""
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
        _reject_single_string(known_ids)
        known = known_ids if isinstance(known_ids, set | frozenset) else frozenset(known_ids)
        self._known = known
        self._ranges: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
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
                self._ranges.setdefault((sr, spec.block), []).append((spec.start, spec.end, kid))

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
        hits = [
            rid
            for lo, hi, rid in self._ranges.get((sr, spec.block), ())
            if _covers(lo, hi, spec.start)
        ]
        return hits[0] if len(hits) == 1 else None


def _reject_single_string(known_ids: object) -> None:
    if isinstance(known_ids, str | bytes):
        raise TypeError(
            "known_ids must be an iterable of article ids, not a single string "
            f"(got {known_ids!r}); wrap it in a list"
        )


def resolve_id(article_id: str, known_ids: Iterable[str]) -> str | None:
    """Map an article id to the known chunk id that holds it: itself if known, else the single
    known range id that covers it, else None. ``known_ids`` may be any iterable of ids (consumed
    once; a single string is a ``TypeError``); for many lookups build an :class:`IdResolver`."""
    _reject_single_string(known_ids)
    try:
        canonical = normalize_article_id(article_id)
    except ArticleIdError:
        return None
    if isinstance(known_ids, set | frozenset) and canonical in known_ids:
        return canonical
    return IdResolver(known_ids).resolve(canonical)
