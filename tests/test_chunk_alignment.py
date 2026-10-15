"""Chunking, deterministic ordering and DE/FR/IT alignment. No network access."""

from __future__ import annotations

import pandas as pd
import pytest

from akn_fixtures import ART_1, ART_10_REPEALED, ART_12_BIS, ART_257_D, wrap
from fedlex_bench import alignment, chunk, ingest
from fedlex_bench.ingest import ParsedArticle, parse_akn

COLS = chunk.COLUMNS


def _pa(
    number: str,
    text: str = "x",
    repealed: bool = False,
    kind: str = "article",
    historical: bool = False,
    n_paragraphs: int = 1,
    n_items: int = 0,
    displayed: str | None = None,
    footnotes: str = "",
) -> ParsedArticle:
    return ParsedArticle(
        number=number,
        kind=kind,
        eid=f"art_{number}",
        title="t",
        text="" if repealed else text,
        is_repealed=repealed,
        repeal_note="Aufgehoben" if repealed else "",
        is_historical=historical,
        block_heading="Block" if kind == "transitional" else "",
        displayed_number=number.rsplit("/", 1)[-1] if displayed is None else displayed,
        n_paragraphs=n_paragraphs,
        n_items=n_items,
        footnotes=footnotes,
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


def test_footnotes_column_is_last_and_filled_for_repealed_rows() -> None:
    assert COLS[-1] == "footnotes"
    arts = parse_akn(wrap(ART_1 + ART_10_REPEALED + ART_257_D))
    rows = {r["article_id"]: r for r in chunk.articles_to_rows(arts, "220", "de", "u", "d")}
    assert rows["220:1"]["footnotes"] == "Fassung gemaess X."
    assert "Fassung gemaess" not in rows["220:1"]["text"]
    assert rows["220:10"]["text"] == ""
    assert rows["220:10"]["footnotes"].startswith("Aufgehoben durch Anhang 1")
    assert rows["220:257d"]["footnotes"] == ""


def test_duplicate_ids_raise_instead_of_being_renamed() -> None:
    with pytest.raises(ValueError, match="duplicate article id 220:5"):
        chunk.articles_to_rows([_pa("5"), _pa("5")], "220", "de", "u", "d")


def test_new_columns_historical_and_block_heading() -> None:
    arts = [_pa("1"), _pa("u2/178", kind="transitional", historical=True)]
    df = _df({"de": arts})
    assert {"is_historical", "block_heading"} <= set(df.columns)
    assert df["is_historical"].dtype == bool
    old = df[df["article_id"] == "220:u2/178"].iloc[0]
    assert old["is_historical"] and old["block_heading"] == "Block"
    assert chunk.in_force(df)["article_id"].tolist() == ["220:1"]


def test_in_force_excludes_repealed_and_historical() -> None:
    arts = [_pa("1"), _pa("2", repealed=True), _pa("u1/1", kind="transitional", historical=True)]
    assert chunk.in_force(_df({"de": arts}))["article_id"].tolist() == ["220:1"]


def test_natural_sort_order_of_article_numbers() -> None:
    nums = ["257", "6bis", "10", "6", "257a", "6a", "2", "257d", "u1/2", "u1/10"]
    ordered = sorted(nums, key=chunk.natural_key)
    assert ordered == ["2", "6", "6a", "6bis", "10", "257", "257a", "257d", "u1/2", "u1/10"]


def test_latin_suffixes_sort_by_ordinal_not_alphabetically() -> None:
    latin = ["decies", "quinquies", "quater", "ter", "bis", "novies", "sexies", "undecies"]
    shuffled = [f"12{w}" for w in latin]
    expected = [f"12{w}" for w in "bis ter quater quinquies sexies novies decies undecies".split()]
    assert sorted(shuffled, key=chunk.natural_key) == expected
    # letters before Latin multiplicatives, base numbers numerically
    mixed = ["13", "12bis", "12b", "12", "12abis", "12a", "2"]
    assert sorted(mixed, key=chunk.natural_key) == [
        "2",
        "12",
        "12a",
        "12abis",
        "12b",
        "12bis",
        "13",
    ]


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
    assert alignment.flag_disagreements(df, "is_historical").empty


def test_historical_chunks_are_not_sampled_or_counted_in_force() -> None:
    arts = [_pa("1"), _pa("2")] + [
        _pa(f"u2/{n}", kind="transitional", historical=True) for n in (1, 2)
    ]
    df = _df({"de": arts, "fr": arts, "it": arts})
    assert alignment.alignment_rate(df) == (4, 4, 1.0)  # historical ids still align
    assert alignment.alignment_rate(chunk.in_force(df)) == (2, 2, 1.0)
    assert set(alignment.sample_aligned(df, 5)) == {"220:1", "220:2"}


def _struct(**per_lang: list[ParsedArticle]) -> pd.DataFrame:
    rows = []
    for lang, arts in per_lang.items():
        for a in arts:
            rows.append(
                {
                    "sr": "220",
                    "lang": lang,
                    "article_id": f"220:{a.number}",
                    "eid": a.eid,
                    "number": a.number,
                    "displayed_number": a.displayed_number,
                    "n_paragraphs": a.n_paragraphs,
                    "n_items": a.n_items,
                }
            )
    return pd.DataFrame(rows)


def test_paragraph_and_item_count_mismatches_are_listed() -> None:
    de = [_pa("644", n_paragraphs=2), _pa("1033", n_paragraphs=2, n_items=3), _pa("5")]
    fr = [_pa("644", n_paragraphs=2), _pa("1033", n_paragraphs=1, n_items=3), _pa("5")]
    it = [_pa("644", n_paragraphs=1), _pa("1033", n_paragraphs=1, n_items=3), _pa("5")]
    sm = alignment.structure_mismatches(_struct(de=de, fr=fr, it=it))
    assert sorted(sm.index) == ["220:1033", "220:644"]
    assert sm.loc["220:644", "it"] == "1p/0i" and sm.loc["220:644", "de"] == "2p/0i"


def test_structure_check_ignores_ids_missing_in_a_language() -> None:
    de, fr = [_pa("1", n_paragraphs=2)], [_pa("1", n_paragraphs=2)]
    assert alignment.structure_mismatches(_struct(de=de, fr=fr, it=[_pa("2")])).empty


def test_eid_vs_displayed_number_mismatch_detection() -> None:
    ok = _pa("257d")
    bad = _pa("697l-697m", displayed="697l")
    unit = _pa("u15", kind="transitional", displayed="")  # block pseudo-article: skipped
    trans = _pa("u1/12cbis", kind="transitional", displayed="12cbis")  # prefix stripped
    mm = alignment.displayed_number_mismatches(_struct(it=[ok, bad, unit, trans]))
    assert mm["article_id"].tolist() == ["220:697l-697m"]


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


def test_length_ratio_threshold_is_tight() -> None:
    de = [_pa("1", "x" * 300), _pa("2", "x" * 300), _pa("3", "x" * 150)]
    fr = [_pa("1", "x" * 200), _pa("2", "x" * 170), _pa("3", "x" * 100)]  # 1.5, 1.76, 1.5
    it = [_pa("1", "x" * 300), _pa("2", "x" * 300), _pa("3", "x" * 150)]
    df = _df({"de": de, "fr": fr, "it": it})
    assert alignment.length_ratio_outliers(df) == ["220:2"]  # >1.6 only; 220:3 is under 200 chars
    assert alignment.length_ratio_outliers(df, ratio=1.4) == ["220:1", "220:2"]


def test_length_ratio_outlier_detection() -> None:
    long = "x" * 600
    de = [_pa("1", long), _pa("2", long)]
    fr = [_pa("1", long), _pa("2", "y" * 100)]  # content mismatch
    it = [_pa("1", long), _pa("2", long)]
    df = _df({"de": de, "fr": fr, "it": it})
    assert alignment.length_ratio_outliers(df) == ["220:2"]


def test_footnote_coverage_counts_per_act_and_language() -> None:
    de = [_pa("1", footnotes="N1"), _pa("2"), _pa("3", repealed=True, footnotes="Aufgehoben")]
    fr = [_pa("1", footnotes="N1"), _pa("2"), _pa("3", repealed=True, footnotes="Abrogé")]
    it = [_pa("1"), _pa("2"), _pa("3", repealed=True, footnotes="Abrogato")]  # 1 lacks its note
    df = _df({"de": de, "fr": fr, "it": it})
    lines = alignment.footnote_coverage_lines(df)
    assert lines[0].split() == ["SR", "de", "fr", "it", "differ"]
    assert lines[1].split() == ["220", "2/3", "2/3", "1/3", "1"]
    assert lines[-1].split() == ["ALL", "2/3", "2/3", "1/3", "1"]
    legacy = df.drop(columns=["footnotes"])
    assert "rebuild" in alignment.footnote_coverage_lines(legacy)[0]


def _manifest_and_raw(tmp_path) -> dict:
    manifest = {
        "220": {
            "sr": "220",
            "consolidation_date": "2026-10-01",
            "consolidation_end": None,
            "as_of": "2026-10-15",
            "consolidation": "c",
            "files": {
                lang: {"format": "xml", "url": f"u/{lang}-xml-3.xml", "sha256": "ab" * 32}
                for lang in ("de", "fr", "it")
            },
        }
    }
    for lang in ("de", "fr", "it"):
        p = ingest.file_path(manifest["220"], lang, tmp_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(wrap(ART_12_BIS))
    return manifest


def test_report_states_what_id_alignment_shows_and_passes_or_fails(tmp_path) -> None:
    arts = [_pa("1", "Text eins."), _pa("2", "Text zwei.")]
    df = _df({"de": arts, "fr": arts, "it": arts})
    manifest = _manifest_and_raw(tmp_path)
    parquet = tmp_path / "a.parquet"
    digest = chunk.write_parquet(df, parquet)
    report = alignment.build_report(df, manifest, raw_dir=tmp_path, parquet_path=parquet)
    assert "100.00%" in report and "Threshold 95% on id alignment: PASS" in report
    assert "None: every article_id exists" in report
    assert "same article structure" in report and "does NOT show" in report
    assert f"articles.parquet: sha256 {digest}" in report  # parquet hash recorded
    assert "ab" * 32 in report  # raw file hashes recorded
    for section in ("5a.", "5b.", "5c.", "6b."):
        assert section in report

    assert "8. FOOTNOTE COVERAGE" in report
    assert "9. FIVE RANDOM" in report

    broken = _df({"de": arts, "fr": arts, "it": arts[:1]})
    assert "FAIL for 220, ALL" in alignment.build_report(broken, manifest, raw_dir=tmp_path)
