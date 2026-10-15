"""Chunking, deterministic ordering and DE/FR/IT alignment. No network access."""

from __future__ import annotations

import pandas as pd

from akn_fixtures import ART_1, ART_10_REPEALED, ART_12_BIS, ART_257_D, wrap
from fedlex_bench import alignment, chunk
from fedlex_bench.ingest import ParsedArticle, parse_akn

COLS = chunk.COLUMNS


def _pa(
    number: str, text: str = "x", repealed: bool = False, kind: str = "article"
) -> ParsedArticle:
    return ParsedArticle(
        number=number,
        kind=kind,
        eid=f"art_{number}",
        title="t",
        text="" if repealed else text,
        is_repealed=repealed,
        repeal_note="Aufgehoben" if repealed else "",
    )


def _df(spec: dict[str, list[ParsedArticle]], sr: str = "220") -> pd.DataFrame:
    rows: list[dict] = []
    for lang, arts in spec.items():
        rows += chunk.articles_to_rows(arts, sr, lang, f"https://x/{lang}.xml", "2026-10-01")
    return chunk.sort_articles(pd.DataFrame(rows, columns=COLS))


# ---------------------------------------------------------------- chunk


def test_rows_have_expected_columns_and_ids() -> None:
    arts = parse_akn(wrap(ART_1 + ART_257_D + ART_10_REPEALED))
    rows = chunk.articles_to_rows(arts, "220", "de", "https://x/de.xml", "2026-10-01")
    assert list(rows[0]) == COLS
    assert [r["article_id"] for r in rows] == ["220:1", "220:257d", "220:10"]
    assert rows[0]["source_url"] == "https://x/de.xml#art_1"
    assert rows[0]["consolidation_date"] == "2026-10-01"
    assert rows[2]["is_repealed"] is True and rows[2]["text"] == ""  # kept, not dropped


def test_duplicate_ids_are_disambiguated_not_dropped() -> None:
    rows = chunk.articles_to_rows([_pa("5"), _pa("5")], "220", "de", "u", "d")
    assert [r["article_id"] for r in rows] == ["220:5", "220:5#2"]


def test_natural_sort_order_of_article_numbers() -> None:
    nums = ["257", "6bis", "10", "6", "257a", "6a", "2", "257d", "u1/2", "u1/10"]
    ordered = sorted(nums, key=chunk.natural_key)
    assert ordered == ["2", "6", "6a", "6bis", "10", "257", "257a", "257d", "u1/2", "u1/10"]


def test_sort_articles_is_deterministic_and_input_order_independent() -> None:
    arts = [_pa("10"), _pa("2"), _pa("2bis"), _pa("1", kind="transitional")]
    a = _df({"de": arts, "fr": arts})
    b = _df({"fr": list(reversed(arts)), "de": list(reversed(arts))})
    pd.testing.assert_frame_equal(a, b)
    first = a[a["lang"] == "de"]["article_id"].tolist()
    assert first == ["220:2", "220:2bis", "220:10", "220:1"]  # body before transitional
    assert a["lang"].tolist()[:2] == ["de", "fr"]  # same id: languages sorted


def test_sr_sorting_is_numeric_aware() -> None:
    df = pd.concat([_df({"de": [_pa("1")]}, sr="235.1"), _df({"de": [_pa("1")]}, sr="220")])
    assert chunk.sort_articles(df.reset_index(drop=True))["sr"].tolist() == ["220", "235.1"]


def test_parquet_roundtrip_is_byte_identical(tmp_path) -> None:
    df = _df({"de": [_pa("1"), _pa("2", repealed=True)], "fr": [_pa("1"), _pa("2", repealed=True)]})
    p = tmp_path / "a.parquet"
    h1 = chunk.write_parquet(df, p)
    h2 = chunk.write_parquet(df, p)
    assert h1 == h2
    back = pd.read_parquet(p)
    pd.testing.assert_frame_equal(back, df)
    assert back["is_repealed"].dtype == bool


# ---------------------------------------------------------------- alignment


def test_full_alignment_rate() -> None:
    arts = [_pa("1"), _pa("257d"), _pa("12bis")]
    df = _df({"de": arts, "fr": arts, "it": arts})
    assert alignment.alignment_rate(df) == (3, 3, 1.0)
    assert alignment.missing_ids(df) == []


def test_missing_id_with_reason_text_vs_repealed() -> None:
    de = [_pa("1"), _pa("2"), _pa("3", repealed=True)]
    fr = [_pa("1"), _pa("2"), _pa("3", repealed=True)]
    it = [_pa("1")]  # IT lacks 2 (live elsewhere) and 3 (repealed elsewhere)
    df = _df({"de": de, "fr": fr, "it": it})
    assert alignment.alignment_rate(df) == (1, 3, 1 / 3)
    miss = {(m["lang"], m["article_id"]): m["reason"] for m in alignment.missing_ids(df)}
    assert set(miss) == {("it", "220:2"), ("it", "220:3")}
    assert "numbering difference or parsing issue" in miss[("it", "220:2")]
    assert miss[("it", "220:3")].startswith("repealed/elided in de,fr")


def test_numbering_difference_shows_up_in_both_directions() -> None:
    df = _df({"de": [_pa("12bis")], "fr": [_pa("12bis")], "it": [_pa("12a")]})
    miss = alignment.missing_ids(df)
    assert {(m["lang"], m["article_id"]) for m in miss} == {
        ("it", "220:12bis"),
        ("de", "220:12a"),
        ("fr", "220:12a"),
    }
    assert alignment.alignment_rate(df)[0] == 0


def test_alignment_is_per_act() -> None:
    a = _df({"de": [_pa("1")], "fr": [_pa("1")], "it": [_pa("1")]}, sr="210")
    b = _df({"de": [_pa("1"), _pa("2")], "fr": [_pa("1")], "it": [_pa("1")]}, sr="220")
    df = pd.concat([a, b], ignore_index=True)
    assert alignment.alignment_rate(df, "210")[2] == 1.0
    assert alignment.alignment_rate(df, "220")[2] == 0.5
    assert alignment.alignment_rate(df)[:2] == (2, 3)


def test_repealed_flag_disagreement_is_reported() -> None:
    de = [_pa("1"), _pa("2", repealed=True)]
    it = [_pa("1"), _pa("2")]
    df = _df({"de": de, "fr": de, "it": it})
    dis = alignment.flag_disagreements(df)
    assert list(dis.index) == ["220:2"]


def test_sample_is_seeded_live_aligned_and_in_force_only() -> None:
    arts = [_pa(str(n)) for n in range(1, 40)] + [_pa("99", repealed=True)]
    df = _df({"de": arts, "fr": arts, "it": arts})
    s1 = alignment.sample_aligned(df, 5, seed=42)
    s2 = alignment.sample_aligned(df, 5, seed=42)
    assert s1 == s2 and len(s1) == 5 and len(set(s1)) == 5
    assert "220:99" not in s1
    assert alignment.sample_aligned(df, 5, seed=43) != s1


def test_sample_excludes_transitional_and_partially_aligned() -> None:
    de = [_pa("1"), _pa("u1/1", kind="transitional"), _pa("2")]
    it = [_pa("1"), _pa("u1/1", kind="transitional")]  # "2" missing in it
    df = _df({"de": de, "fr": de, "it": it})
    assert alignment.sample_aligned(df, 5) == ["220:1"]


def test_length_ratio_outlier_detection() -> None:
    long = "x" * 600
    de = [_pa("1", long), _pa("2", long)]
    fr = [_pa("1", long), _pa("2", "y" * 100)]  # content mismatch
    it = [_pa("1", long), _pa("2", long)]
    df = _df({"de": de, "fr": fr, "it": it})
    assert alignment.length_ratio_outliers(df) == ["220:2"]


def test_report_mentions_pass_and_fail(tmp_path) -> None:
    arts = [_pa("1", "Text eins."), _pa("2", "Text zwei.")]
    df = _df({"de": arts, "fr": arts, "it": arts})
    manifest = {
        "220": {
            "consolidation_date": "2026-10-01",
            "consolidation_end": None,
            "as_of": "2026-10-15",
            "consolidation": "c",
            "files": {lang: {"format": "xml", "url": "u"} for lang in ("de", "fr", "it")},
        }
    }
    for lang in ("de", "fr", "it"):
        p = tmp_path / "220" / f"{lang}_2026-10-01.xml"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(wrap(ART_12_BIS))
    report = alignment.build_report(df, manifest, raw_dir=tmp_path)
    assert "100.00%" in report and "Threshold 95%: PASS" in report
    assert "None: every article_id exists" in report

    broken = _df({"de": arts, "fr": arts, "it": arts[:1]})
    assert "FAIL for 220, ALL" in alignment.build_report(broken, manifest, raw_dir=tmp_path)
