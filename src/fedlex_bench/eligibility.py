"""Which chunks may be retrieved, and which articles may anchor a benchmark question.

Two levels, both deterministic (sorted output, no randomness):

- **Retrieval-eligible** (:func:`retrieval_eligible`): ``kind == "article"`` (main text, not
  transitional/final provisions), not ``is_repealed``, not ``is_historical``, non-empty ``text``.
  This is what goes into the retrieval index.
- **Benchmark-eligible** (:func:`benchmark_eligible`): an article id that is retrieval-eligible in
  **all three languages** and whose text has at least ``min_chars`` characters in **every**
  language. The result is an article-level pool (one row per ``article_id``) with the per-language
  lengths; :func:`pool_rows` / :func:`benchmark_rows` return the chunk rows of the pool per
  language. Because the pool is defined once on article ids, the DE, FR and IT pools are identical
  by construction.

About the 200-character default (``DEFAULT_MIN_CHARS``): it is a judgement call, not a measured
optimum. Very short articles give a question too little text to be answerable from the article
alone, and the shortest language decides because the same question must be answerable in every
language. It is a parameter; ``pool_size_table`` shows how the pool shrinks at 100/200/300
(Phase 0 corpus: 3,502 / 2,877 / 2,162 of 3,663). It was not tuned on any test split.

Run: ``uv run python -m fedlex_bench.eligibility [--min-chars N]`` prints the counts.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from fedlex_bench.chunk import ARTICLES_PARQUET, natural_key, sort_articles
from fedlex_bench.ingest import LANGS

log = logging.getLogger("fedlex_bench.eligibility")

DEFAULT_MIN_CHARS = 200
DEFAULT_THRESHOLDS: tuple[int, ...] = (100, 200, 300)
REQUIRED_COLUMNS = ("sr", "article_id", "lang", "text", "kind", "is_repealed", "is_historical")


def _require_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"articles frame is missing column(s): {', '.join(missing)}")


def _check_min_chars(min_chars: int) -> None:
    if isinstance(min_chars, bool) or not isinstance(min_chars, int) or min_chars < 0:
        raise ValueError(f"min_chars must be a non-negative integer, got {min_chars!r}")


def _ineligible_reason(r) -> str:
    """First rule an article chunk fails ("" when it passes). ``r`` is a row or itertuples item."""
    if r.kind != "article":
        return "not_article"
    if r.is_repealed:
        return "repealed"
    if r.is_historical:
        return "historical"
    if not _has_text(r.text):
        return "empty"
    return ""


def _has_text(value) -> bool:
    return isinstance(value, str) and value.strip() != ""


def retrieval_eligible(df: pd.DataFrame) -> pd.DataFrame:
    """Main-text article chunks that are in force and have text: ``kind == "article"``, not
    ``is_repealed``, not ``is_historical``, non-empty ``text`` (missing or whitespace-only counts
    as empty). Returns a new frame in deterministic order (sr, natural article order, language);
    the input is not modified."""
    _require_columns(df)
    mask = (
        (df["kind"] == "article")
        & ~df["is_repealed"].astype(bool)
        & ~df["is_historical"].astype(bool)
        & df["text"].map(_has_text).astype(bool)
    )
    return sort_articles(df[mask])


def _pool_columns(langs: tuple[str, ...]) -> list[str]:
    return ["article_id", "sr", *[f"chars_{lang}" for lang in langs], "min_chars"]


def _natural_id_key(article_id: str) -> tuple:
    sr, _, number = article_id.partition(":")
    return (natural_key(sr), natural_key(number), article_id)


def _lengths_wide(df: pd.DataFrame, langs: tuple[str, ...]) -> pd.DataFrame:
    """article_id x language -> text length, for retrieval-eligible chunks only. Raises when one
    (article_id, lang) occurs twice, since the alignment key would be ambiguous."""
    elig = retrieval_eligible(df)
    elig = elig[elig["lang"].isin(langs)]
    dup = elig[elig.duplicated(["article_id", "lang"], keep=False)]
    if not dup.empty:
        pairs = zip(dup["article_id"], dup["lang"], strict=True)
        sample = ", ".join(sorted({f"{a}/{lang}" for a, lang in pairs})[:5])
        raise ValueError(f"duplicate (article_id, lang) rows: {sample}")
    lengths = elig.assign(n=elig["text"].str.len())
    wide = lengths.pivot(index="article_id", columns="lang", values="n")
    return wide.reindex(columns=list(langs))


def benchmark_eligible(
    df: pd.DataFrame, min_chars: int = DEFAULT_MIN_CHARS, langs: tuple[str, ...] = LANGS
) -> pd.DataFrame:
    """Article-level benchmark pool: ids retrieval-eligible in every language of ``langs`` with
    ``len(text) >= min_chars`` in every one of them (inclusive).

    One row per ``article_id`` with columns ``article_id``, ``sr``, ``chars_<lang>`` per language
    and ``min_chars`` (the shortest of the per-language lengths). Sorted by act and natural
    article order. ``min_chars=0`` still requires non-empty text."""
    _check_min_chars(min_chars)
    _require_columns(df)
    langs = tuple(langs)
    wide = _lengths_wide(df, langs).dropna()  # present and retrieval-eligible in every language
    wide = wide[(wide >= min_chars).all(axis=1)].astype(int)
    pool = pd.DataFrame(
        {
            "article_id": wide.index.to_numpy(dtype=object),
            "sr": [a.partition(":")[0] for a in wide.index],
            **{f"chars_{lang}": wide[lang].to_numpy() for lang in langs},
            "min_chars": wide.min(axis=1).to_numpy(dtype=int),
        },
        columns=_pool_columns(langs),
    )
    order = sorted(range(len(pool)), key=lambda i: _natural_id_key(pool["article_id"].iloc[i]))
    return pool.iloc[order].reset_index(drop=True)


def pool_rows(df: pd.DataFrame, pool: pd.DataFrame, lang: str) -> pd.DataFrame:
    """The chunk rows of one language for the articles of ``pool``, in deterministic order. All
    original columns are kept."""
    if lang not in LANGS:
        raise ValueError(f"unknown language {lang!r}; expected one of {LANGS}")
    _require_columns(df)
    sub = retrieval_eligible(df[df["lang"] == lang])
    return sub[sub["article_id"].isin(set(pool["article_id"]))].reset_index(drop=True)


def benchmark_rows(
    df: pd.DataFrame, min_chars: int = DEFAULT_MIN_CHARS, langs: tuple[str, ...] = LANGS
) -> dict[str, pd.DataFrame]:
    """Per-language chunk rows of the benchmark pool: ``{lang: rows}``, with the same article ids
    in the same order for every language."""
    pool = benchmark_eligible(df, min_chars, langs)
    return {lang: pool_rows(df, pool, lang) for lang in langs}


def eligibility_mismatches(df: pd.DataFrame, langs: tuple[str, ...] = LANGS) -> pd.DataFrame:
    """Article ids that are retrieval-eligible in some languages but not all: what cannot be
    aligned and is therefore left out of the pool. Columns: ``article_id``, ``sr``,
    ``eligible_in`` (comma-separated languages) and ``reason_<lang>`` ("ok", "absent",
    "repealed", "historical", "empty" or "not_article"). Ids ineligible everywhere are
    consistent and are not listed. Empty when every id is eligible in all languages or none."""
    _require_columns(df)
    langs = tuple(langs)
    art = df[df["lang"].isin(langs)]
    reasons: dict[str, dict[str, str]] = {}
    for r in art.itertuples(index=False):
        reasons.setdefault(r.article_id, {})[r.lang] = _ineligible_reason(r) or "ok"
    rows = []
    for aid in sorted(reasons, key=_natural_id_key):
        per = {lang: reasons[aid].get(lang, "absent") for lang in langs}
        ok = [lang for lang in langs if per[lang] == "ok"]
        if not ok or len(ok) == len(langs):
            continue
        rows.append(
            {
                "article_id": aid,
                "sr": aid.partition(":")[0],
                "eligible_in": ",".join(ok),
                **{f"reason_{lang}": per[lang] for lang in langs},
            }
        )
    return pd.DataFrame(
        rows,
        columns=["article_id", "sr", "eligible_in", *[f"reason_{lang}" for lang in langs]],
    )


def pool_size_table(
    df: pd.DataFrame,
    thresholds: tuple[int, ...] = DEFAULT_THRESHOLDS,
    langs: tuple[str, ...] = LANGS,
) -> pd.DataFrame:
    """Pool size per act (rows, plus a final ``ALL``) for several ``min_chars`` thresholds.
    ``eligible_all_langs`` is the number of articles retrieval-eligible in all languages (the
    pool before any length rule)."""
    _require_columns(df)
    wide = _lengths_wide(df, tuple(langs)).dropna()
    sr_of = pd.Series([a.partition(":")[0] for a in wide.index], index=wide.index)
    srs = sorted({a.partition(":")[0] for a in df["article_id"]}, key=natural_key)
    cols = ["eligible_all_langs", *[f"min_chars_{t}" for t in thresholds]]
    rows = []
    for sr in srs:
        w = wide[sr_of == sr]
        rows.append([len(w), *[int((w >= t).all(axis=1).sum()) for t in thresholds]])
    table = pd.DataFrame(rows, index=pd.Index(srs), columns=cols, dtype=int)
    table.loc["ALL"] = table.sum()
    return table


def log_counts(df: pd.DataFrame, min_chars: int, thresholds: tuple[int, ...]) -> None:
    log.info("acts: %d (%s)", df["sr"].nunique(), ", ".join(sorted(df["sr"].unique())))
    elig = retrieval_eligible(df)
    log.info(
        "retrieval-eligible chunks per language: %s",
        {lang: int((elig["lang"] == lang).sum()) for lang in LANGS},
    )
    mism = eligibility_mismatches(df)
    log.info("articles eligible in some but not all languages (not aligned): %d", len(mism))
    for r in mism.itertuples(index=False):
        log.warning("not aligned: %s eligible in %s | %s", r.article_id, r.eligible_in, r)
    pool = benchmark_eligible(df, min_chars)
    log.info("benchmark pool at min_chars=%d: %d articles", min_chars, len(pool))
    table = pool_size_table(df, thresholds)
    for line in table.to_string().splitlines():
        log.info("%s", line)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parquet", type=Path, default=ARTICLES_PARQUET)
    ap.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    ap.add_argument("--thresholds", type=int, nargs="*", default=list(DEFAULT_THRESHOLDS))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = pd.read_parquet(args.parquet)
    log_counts(df, args.min_chars, tuple(args.thresholds))


if __name__ == "__main__":
    main()
