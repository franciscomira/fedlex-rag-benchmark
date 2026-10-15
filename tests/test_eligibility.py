"""Retrieval / benchmark eligibility rules. Synthetic frames plus corpus-level checks (the latter
skipped when ``data/processed/articles.parquet`` is missing). No network access."""

from __future__ import annotations

import pandas as pd
import pytest

from fedlex_bench.chunk import ARTICLES_PARQUET
from fedlex_bench.eligibility import (
    DEFAULT_MIN_CHARS,
    benchmark_eligible,
    benchmark_rows,
    eligibility_mismatches,
    pool_rows,
    pool_size_table,
    retrieval_eligible,
)

LANGS = ("de", "fr", "it")


def row(
    article_id: str,
    lang: str,
    text: str = "x" * 250,
    kind: str = "article",
    repealed: bool = False,
    historical: bool = False,
) -> dict:
    return {
        "sr": article_id.partition(":")[0],
        "article_id": article_id,
        "lang": lang,
        "title": "t",
        "text": "" if repealed else text,
        "source_url": "u",
        "consolidation_date": "2026-01-01",
        "kind": kind,
        "is_repealed": repealed,
        "is_historical": historical,
        "block_heading": "",
        "footnotes": "",
    }


def frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def all_langs(article_id: str, **kw) -> list[dict]:
    return [row(article_id, lang, **kw) for lang in LANGS]


# ------------------------------------------------------------------ retrieval_eligible


def test_retrieval_eligible_keeps_plain_article():
    df = frame(all_langs("220:1"))
    assert len(retrieval_eligible(df)) == 3


def test_retrieval_eligible_drops_transitional():
    df = frame(all_langs("220:u1/3", kind="transitional") + all_langs("220:2"))
    out = retrieval_eligible(df)
    assert set(out["article_id"]) == {"220:2"}


def test_retrieval_eligible_drops_repealed():
    df = frame(all_langs("220:3", repealed=True) + all_langs("220:4"))
    assert set(retrieval_eligible(df)["article_id"]) == {"220:4"}


def test_retrieval_eligible_drops_historical():
    df = frame(all_langs("210:5", historical=True) + all_langs("210:6"))
    assert set(retrieval_eligible(df)["article_id"]) == {"210:6"}


def test_retrieval_eligible_drops_empty_and_whitespace_text():
    df = frame(
        [
            row("220:7", "de", text=""),
            row("220:8", "de", text="  \n\t "),
            row("220:9", "de", text="Art"),
        ]
    )
    assert list(retrieval_eligible(df)["article_id"]) == ["220:9"]


def test_retrieval_eligible_treats_missing_text_as_empty():
    df = frame([row("220:7", "de"), row("220:8", "de")])
    df["text"] = df["text"].astype(object)
    df.loc[0, "text"] = None
    assert list(retrieval_eligible(df)["article_id"]) == ["220:8"]


def test_retrieval_eligible_range_chunk_is_not_eligible():
    # Repealed range chunks have no text and is_repealed=True.
    df = frame(all_langs("210:135-149", repealed=True))
    assert retrieval_eligible(df).empty


def test_retrieval_eligible_is_sorted_naturally_and_does_not_mutate():
    rows = []
    for n in ("10", "257d", "2", "257", "257a", "6bis", "6"):
        rows += all_langs(f"220:{n}")
    rows += all_langs("210:9")
    df = frame(rows).sample(frac=1.0, random_state=1).reset_index(drop=True)
    before = df.copy()
    out = retrieval_eligible(df)
    assert list(out.loc[out["lang"] == "de", "article_id"]) == [
        "210:9",
        "220:2",
        "220:6",
        "220:6bis",
        "220:10",
        "220:257",
        "220:257a",
        "220:257d",
    ]
    pd.testing.assert_frame_equal(df, before)  # input untouched
    # all languages of one id are adjacent and in fixed language order
    assert list(out["lang"].iloc[:3]) == ["de", "fr", "it"]


def test_retrieval_eligible_missing_columns_raise():
    with pytest.raises(ValueError, match="is_historical"):
        retrieval_eligible(frame(all_langs("220:1")).drop(columns=["is_historical"]))


# ------------------------------------------------------------------ benchmark_eligible


def test_default_threshold_is_200():
    assert DEFAULT_MIN_CHARS == 200


def test_benchmark_eligible_basic_columns_and_lengths():
    rows = [
        row("220:1", "de", text="a" * 300),
        row("220:1", "fr", text="b" * 250),
        row("220:1", "it", text="c" * 210),
    ]
    pool = benchmark_eligible(frame(rows))
    assert list(pool["article_id"]) == ["220:1"]
    r = pool.iloc[0]
    assert (r["sr"], r["chars_de"], r["chars_fr"], r["chars_it"], r["min_chars"]) == (
        "220",
        300,
        250,
        210,
        210,
    )


def test_benchmark_eligible_threshold_is_inclusive():
    rows = [row("220:1", lang, text="x" * 200) for lang in LANGS]
    assert len(benchmark_eligible(frame(rows), min_chars=200)) == 1
    assert len(benchmark_eligible(frame(rows), min_chars=201)) == 0


def test_benchmark_eligible_requires_length_in_every_language():
    for short_lang in LANGS:
        rows = [
            row("220:1", lang, text="x" * (50 if lang == short_lang else 400)) for lang in LANGS
        ]
        assert benchmark_eligible(frame(rows)).empty, short_lang


def test_benchmark_eligible_requires_presence_in_every_language():
    for missing in LANGS:
        rows = [row("220:1", lang) for lang in LANGS if lang != missing]
        assert benchmark_eligible(frame(rows)).empty, missing


def test_benchmark_eligible_requires_retrieval_eligibility_in_every_language():
    for bad in LANGS:
        rows = [row("220:1", lang, repealed=(lang == bad)) for lang in LANGS]
        assert benchmark_eligible(frame(rows)).empty, bad
    rows = [row("210:1", lang, historical=(lang == "fr")) for lang in LANGS]
    assert benchmark_eligible(frame(rows)).empty


def test_benchmark_eligible_excludes_transitional_even_if_long():
    df = frame(all_langs("220:u1/3", kind="transitional", text="x" * 900))
    assert benchmark_eligible(df).empty


def test_benchmark_eligible_min_chars_parameter_changes_pool():
    rows = []
    rows += [row("220:1", lang, text="x" * 150) for lang in LANGS]
    rows += [row("220:2", lang, text="x" * 250) for lang in LANGS]
    rows += [row("220:3", lang, text="x" * 350) for lang in LANGS]
    df = frame(rows)
    assert list(benchmark_eligible(df, min_chars=100)["article_id"]) == ["220:1", "220:2", "220:3"]
    assert list(benchmark_eligible(df, min_chars=200)["article_id"]) == ["220:2", "220:3"]
    assert list(benchmark_eligible(df, min_chars=300)["article_id"]) == ["220:3"]


def test_benchmark_eligible_validates_min_chars():
    df = frame(all_langs("220:1"))
    with pytest.raises(ValueError):
        benchmark_eligible(df, min_chars=-1)
    with pytest.raises(ValueError):
        benchmark_eligible(df, min_chars=True)  # a bool is not a threshold
    with pytest.raises(ValueError):
        benchmark_eligible(df, min_chars=1.5)


def test_benchmark_eligible_min_chars_zero_still_needs_text():
    rows = [row("220:1", lang, text="") for lang in LANGS]
    assert benchmark_eligible(frame(rows), min_chars=0).empty


def test_benchmark_eligible_one_row_per_id_sorted_and_deterministic():
    rows = []
    for n in ("10", "257d", "2", "257", "257a", "6bis", "6"):
        rows += all_langs(f"220:{n}")
    rows += all_langs("210:9")
    df = frame(rows)
    shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    a, b = benchmark_eligible(df), benchmark_eligible(shuffled)
    pd.testing.assert_frame_equal(a, b)
    assert list(a["article_id"]) == [
        "210:9",
        "220:2",
        "220:6",
        "220:6bis",
        "220:10",
        "220:257",
        "220:257a",
        "220:257d",
    ]
    assert a["article_id"].is_unique


def test_benchmark_eligible_duplicate_id_in_one_language_raises():
    rows = all_langs("220:1") + [row("220:1", "de")]
    with pytest.raises(ValueError, match="duplicate"):
        benchmark_eligible(frame(rows))


def test_benchmark_eligible_empty_frame():
    pool = benchmark_eligible(frame(all_langs("220:1", repealed=True)))
    assert pool.empty
    assert list(pool.columns) == [
        "article_id",
        "sr",
        "chars_de",
        "chars_fr",
        "chars_it",
        "min_chars",
    ]


# ------------------------------------------------------------------ per-language rows


def test_pool_rows_are_identical_across_languages_by_construction():
    rows = []
    rows += all_langs("220:1")
    rows += [row("220:2", "de"), row("220:2", "fr")]  # missing in IT
    rows += [row("220:3", "de"), row("220:3", "fr"), row("220:3", "it", text="short")]
    rows += all_langs("220:4")
    df = frame(rows)
    pool = benchmark_eligible(df)
    per_lang = benchmark_rows(df)
    assert set(per_lang) == set(LANGS)
    for lang in LANGS:
        out = per_lang[lang]
        assert list(out["article_id"]) == list(pool["article_id"]) == ["220:1", "220:4"]
        assert set(out["lang"]) == {lang}
        assert list(out.columns) == list(df.columns)  # full chunk rows, all original columns


def test_pool_rows_single_language_uses_given_pool():
    df = frame(all_langs("220:1") + all_langs("220:2"))
    pool = benchmark_eligible(df).iloc[:1]
    out = pool_rows(df, pool, "fr")
    assert list(out["article_id"]) == ["220:1"]
    assert list(out["lang"]) == ["fr"]


def test_pool_rows_unknown_language_raises():
    df = frame(all_langs("220:1"))
    with pytest.raises(ValueError):
        pool_rows(df, benchmark_eligible(df), "en")


# ------------------------------------------------------------------ unaligned ids are reported


def test_eligibility_mismatches_reports_partial_eligibility():
    rows = []
    rows += all_langs("220:1")  # fully eligible: not reported
    rows += all_langs("220:2", repealed=True)  # ineligible everywhere: not a mismatch
    rows += [row("220:3", "de"), row("220:3", "fr")]  # missing in IT
    rows += [row("220:4", "de"), row("220:4", "fr"), row("220:4", "it", repealed=True)]
    df = frame(rows)
    out = eligibility_mismatches(df)
    assert list(out["article_id"]) == ["220:3", "220:4"]
    r3, r4 = out.iloc[0], out.iloc[1]
    assert r3["eligible_in"] == "de,fr" and r3["reason_it"] == "absent"
    assert r4["eligible_in"] == "de,fr" and r4["reason_it"] == "repealed"


def test_eligibility_mismatches_ignores_transitional():
    df = frame([row("220:u1/1", "de", kind="transitional")])
    assert eligibility_mismatches(df).empty


# ------------------------------------------------------------------ size table


def test_pool_size_table_per_act_and_total():
    rows = []
    rows += [row("220:1", lang, text="x" * 150) for lang in LANGS]
    rows += [row("220:2", lang, text="x" * 250) for lang in LANGS]
    rows += [row("210:1", lang, text="x" * 350) for lang in LANGS]
    rows += all_langs("210:2", repealed=True)
    t = pool_size_table(frame(rows), thresholds=(100, 200, 300))
    assert list(t.index) == ["210", "220", "ALL"]
    assert list(t.columns) == [
        "eligible_all_langs",
        "min_chars_100",
        "min_chars_200",
        "min_chars_300",
    ]
    assert t.loc["210"].tolist() == [1, 1, 1, 1]
    assert t.loc["220"].tolist() == [2, 2, 1, 0]
    assert t.loc["ALL"].tolist() == [3, 3, 2, 1]


def test_pool_size_table_is_monotone_in_threshold():
    rows = []
    for i, n in enumerate((50, 150, 250, 350, 450)):
        rows += [row(f"220:{i + 1}", lang, text="x" * n) for lang in LANGS]
    t = pool_size_table(frame(rows), thresholds=(100, 200, 300, 400))
    vals = t.loc["ALL"].tolist()
    assert vals == sorted(vals, reverse=True)


# ------------------------------------------------------------------ corpus-level

corpus = pytest.mark.skipif(
    not ARTICLES_PARQUET.exists(),
    reason="data/processed/articles.parquet missing (run `uv run python -m fedlex_bench.chunk`)",
)


@pytest.fixture(scope="module")
def corpus_df() -> pd.DataFrame:
    return pd.read_parquet(ARTICLES_PARQUET)


@corpus
def test_corpus_retrieval_eligible_matches_phase0_report(corpus_df):
    """Phase 0 report, section 3: 3,844 in-force chunks per language, of which 3,663 are
    main-text articles (the rest are in-force transitional/final provisions)."""
    out = retrieval_eligible(corpus_df)
    per_lang = out.groupby("lang").size().to_dict()
    assert per_lang == {"de": 3663, "fr": 3663, "it": 3663}
    assert (out["kind"] == "article").all()
    live_all_kinds = corpus_df[~corpus_df["is_repealed"] & ~corpus_df["is_historical"]]
    assert live_all_kinds.groupby("lang").size().to_dict() == {"de": 3844, "fr": 3844, "it": 3844}


@corpus
def test_corpus_retrieval_eligible_id_sets_are_identical_across_languages(corpus_df):
    out = retrieval_eligible(corpus_df)
    ids = {lang: set(g["article_id"]) for lang, g in out.groupby("lang")}
    assert ids["de"] == ids["fr"] == ids["it"]
    assert eligibility_mismatches(corpus_df).empty


@corpus
def test_corpus_pool_is_identical_across_languages_and_consistent(corpus_df):
    pool = benchmark_eligible(corpus_df)
    retrieval_ids = set(retrieval_eligible(corpus_df)["article_id"])
    assert pool["article_id"].is_unique
    assert set(pool["article_id"]) <= retrieval_ids
    per_lang = benchmark_rows(corpus_df)
    for lang in LANGS:
        assert list(per_lang[lang]["article_id"]) == list(pool["article_id"])
        lengths = per_lang[lang]["text"].str.len()
        assert (lengths >= DEFAULT_MIN_CHARS).all()
        assert list(lengths) == list(pool[f"chars_{lang}"])
    assert (pool["min_chars"] >= DEFAULT_MIN_CHARS).all()
    assert len(pool) > 0


@corpus
def test_corpus_pool_shrinks_monotonically_with_threshold(corpus_df):
    t = pool_size_table(corpus_df, thresholds=(100, 200, 300))
    all_row = t.loc["ALL"]
    assert all_row["eligible_all_langs"] == 3663
    assert (
        all_row["eligible_all_langs"]
        >= all_row["min_chars_100"]
        >= all_row["min_chars_200"]
        >= all_row["min_chars_300"]
    )
    # per-act rows add up to the total
    per_act = t.drop(index="ALL")
    assert per_act.sum().tolist() == all_row.tolist()


@corpus
def test_corpus_pool_is_deterministic(corpus_df):
    pd.testing.assert_frame_equal(
        benchmark_eligible(corpus_df), benchmark_eligible(corpus_df.sample(frac=1, random_state=3))
    )
