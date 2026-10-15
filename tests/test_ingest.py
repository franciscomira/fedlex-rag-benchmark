"""Parsing, normalisation and discovery logic. No network access."""

from __future__ import annotations

import json

import httpx
import pytest

from akn_fixtures import (
    ART_1,
    ART_10_REPEALED,
    ART_257_D,
    ART_PARTIAL_REPEAL,
    TRANSITIONAL,
    full_act,
    wrap,
)
from fedlex_bench import ingest
from fedlex_bench.ingest import (
    normalize_article_number,
    number_from_eid,
    parse_akn,
    pick_consolidation,
    pick_manifestations,
    unarticled_units,
)

# ---------------------------------------------------------------- id normalisation


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Art. 257 d", "257d"),
        ("Art. 257d", "257d"),
        ("art. 257 d", "257d"),  # non-breaking space
        ("Art. 12bis", "12bis"),
        ("Art. 12 bis", "12bis"),
        ("Art. 40 ter", "40ter"),
        ("Art. 9 quater", "9quater"),
        ("Art. 6a bis", "6abis"),
        ("Art. 6 a bis", "6abis"),
        ("Art. 1", "1"),
        ("Art. 1", "1"),
        ("  ART. 12  ", "12"),
        ("Art. 135–149", "135-149"),  # en dash range
        ("Art. 135 à 149", "135-149"),  # FR range
        ("Art. 135 a 149", "135-149"),  # IT range (connector, not a suffix)
        ("Art. 50 und 51", "50-51"),
        ("Art. 50 et 51", "50-51"),
        ("Art. 50 e 51", "50-51"),
        ("Art. 28d à 28f", "28d-28f"),
        ("Art. 2 4", "2-4"),
        ("", ""),
    ],
)
def test_normalize_article_number(raw: str, expected: str) -> None:
    assert normalize_article_number(raw) == expected


@pytest.mark.parametrize(
    ("eid", "expected"),
    [
        ("art_257_d", "257d"),
        ("art_1", "1"),
        ("art_12_bis", "12bis"),
        ("art_6_a_bis", "6abis"),
        ("art_135_149", "135-149"),  # repealed range
        ("art_226_a_226_d", "226a-226d"),
        ("art_274_274_g", "274-274g"),
        ("art_329_g_bis", "329gbis"),
        ("disp_u12/art_2_4", "2-4"),
        ("disp_u1/art_12_c_bis", "12cbis"),
    ],
)
def test_number_from_eid(eid: str, expected: str) -> None:
    assert number_from_eid(eid) == expected


def test_eid_and_displayed_number_agree_on_simple_cases() -> None:
    for art, eid in [(ART_257_D, "art_257_d"), (ART_1, "art_1")]:
        (parsed,) = [a for a in parse_akn(wrap(art)) if a.eid == eid]
        assert parsed.number == number_from_eid(eid)


# ---------------------------------------------------------------- parsing


@pytest.fixture(scope="module")
def parsed() -> dict:
    return {a.eid: a for a in parse_akn(full_act())}


def test_parses_all_articles_in_document_order(parsed: dict) -> None:
    assert list(parsed)[:3] == ["art_1", "art_257_d", "art_12_bis"]
    assert len(parsed) == 16


def test_body_article_text_title_and_footnote_removal(parsed: dict) -> None:
    a = parsed["art_1"]
    assert a.kind == "article"
    assert a.title == "Abschluss"  # from the enclosing marginal level
    assert a.text == (
        "1 Zum Abschluss ist Willenseinigung erforderlich.\n2 Sie kann ausdruecklich sein."
    )
    assert "Fassung gemaess" not in a.text  # footnote dropped
    assert not a.is_repealed and a.repeal_note == ""


@pytest.mark.parametrize(
    ("eid", "number"),
    [
        ("art_257_d", "257d"),
        ("art_12_bis", "12bis"),
        ("art_6_a_bis", "6abis"),
        ("art_40_ter", "40ter"),
        ("art_9_quater", "9quater"),
    ],
)
def test_suffix_articles(parsed: dict, eid: str, number: str) -> None:
    a = parsed[eid]
    assert a.number == number
    assert a.text == f"Text {number}."
    assert not a.is_repealed


@pytest.mark.parametrize(
    "eid",
    [
        "art_10",  # note says Aufgehoben
        "art_40_g",  # "Eingefügt ... Aufgehoben ..." history
        "art_135_149",  # repealed range
        "art_77",  # "Privo d'oggetto" (obsolete), empty <content/>
        "art_78",  # explicit stub body "Abrogé"
        "art_697_l_697_m",  # IT split-number quirk, note inside <heading>
        "disp_u12/art_2_4",  # "…" elision mark in a transitional block
    ],
)
def test_repealed_or_elided_articles_are_flagged_not_dropped(parsed: dict, eid: str) -> None:
    a = parsed[eid]
    assert a.is_repealed
    assert a.text == ""
    assert a.repeal_note  # reason kept


def test_repeal_note_text(parsed: dict) -> None:
    assert parsed["art_10"].repeal_note.startswith("Aufgehoben durch Anhang 1")
    assert "Eingefügt" in parsed["art_40_g"].repeal_note


def test_partially_repealed_article_stays_in_force(parsed: dict) -> None:
    a = parsed["art_56"]
    assert not a.is_repealed
    assert a.text.splitlines() == ["1 Haftung des Tierhalters.", "2 …"]


def test_split_number_heading_is_not_used_as_title(parsed: dict) -> None:
    assert parsed["art_697_l_697_m"].title == ""
    assert parsed["art_697_l_697_m"].number == "697l-697m"


def test_transitional_articles_get_separate_namespace(parsed: dict) -> None:
    body = parsed["art_1"]
    trans = parsed["disp_u1/art_1"]
    assert body.number == "1" and body.kind == "article"
    assert trans.number == "u1/1" and trans.kind == "transitional"
    assert parsed["disp_u1/art_12_c_bis"].number == "u1/12cbis"
    assert parsed["disp_u12/art_2_4"].number == "u12/2-4"


def test_numbers_are_unique_within_a_document() -> None:
    numbers = [a.number for a in parse_akn(full_act())]
    assert len(numbers) == len(set(numbers))


def test_unarticled_transitional_units_are_reported() -> None:
    units = unarticled_units(full_act())
    assert [eid for eid, _ in units] == ["disp_u4"]
    assert units[0][1].startswith("Uebergangsbestimmung")


def test_parse_accepts_str_and_ignores_external_entities() -> None:
    xml = wrap(ART_10_REPEALED + ART_PARTIAL_REPEAL).decode("utf-8")
    assert [a.number for a in parse_akn(xml)] == ["10", "56"]


def test_transitional_only_document() -> None:
    (a, b) = parse_akn(wrap(TRANSITIONAL))
    assert (a.number, b.number) == ("u1/1", "u1/12cbis")


# ---------------------------------------------------------------- SPARQL result selection


def test_pick_consolidation_latest_applicable_on_date() -> None:
    rows = [
        {"cons": "c/20260101", "start": "2026-01-01", "end": "2026-09-30", "work": "w"},
        {"cons": "c/20261001", "start": "2026-10-01", "end": "2027-06-30", "work": "w"},
        {"cons": "c/20270701", "start": "2027-07-01", "work": "w"},  # future
    ]
    assert pick_consolidation(rows, "2026-10-15")["cons"] == "c/20261001"
    assert pick_consolidation(rows, "2026-10-01")["cons"] == "c/20261001"  # boundary
    assert pick_consolidation(rows, "2026-09-30")["cons"] == "c/20260101"
    assert pick_consolidation(rows, "2030-01-01")["cons"] == "c/20270701"  # open-ended


def test_pick_consolidation_ignores_repealed_work_with_same_sr() -> None:
    old = {"cons": "old/20190301", "start": "2019-03-01", "end": "2023-08-31", "work": "old"}
    new = {"cons": "new/20250707", "start": "2025-07-07", "work": "new"}
    assert pick_consolidation([old, new], "2026-10-15")["work"] == "new"


def test_pick_consolidation_raises_when_none_applicable() -> None:
    rows = [{"cons": "c", "start": "2019-03-01", "end": "2023-08-31", "work": "w"}]
    with pytest.raises(LookupError):
        pick_consolidation(rows, "2026-10-15")


def test_pick_manifestations_prefers_xml_and_latest_revision() -> None:
    rows = [
        {"lang": "de", "fmt": "html", "url": "u/de.html"},
        {"lang": "de", "fmt": "xml", "url": "u/de-xml-2.xml"},
        {"lang": "de", "fmt": "xml", "url": "u/de-xml-3.xml"},
        {"lang": "fr", "fmt": "html", "url": "u/fr.html"},  # no xml: fall back to html
        {"lang": "rm", "fmt": "xml", "url": "u/rm.xml"},  # not a target language
    ]
    out = pick_manifestations(rows)
    assert out["de"] == {"format": "xml", "url": "u/de-xml-3.xml"}
    assert out["fr"] == {"format": "html", "url": "u/fr.html"}
    assert "it" not in out and "rm" not in out


# ---------------------------------------------------------------- idempotent download


def test_ensure_raw_uses_cache_and_never_touches_network(tmp_path, monkeypatch) -> None:
    manifest = {
        "220": {
            "sr": "220",
            "as_of": "2026-10-15",
            "consolidation": "c",
            "consolidation_date": "2026-10-01",
            "consolidation_end": None,
            "work": "w",
            "files": {
                lang: {"format": "xml", "url": f"u/{lang}.xml"} for lang in "de fr it".split()
            },
        }
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for lang in ("de", "fr", "it"):
        p = ingest.raw_path("220", lang, "2026-10-01", "xml", tmp_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"<x/>")

    def boom(*_a, **_k):  # any HTTP client construction means a network attempt
        raise AssertionError("network access attempted")

    monkeypatch.setattr(httpx, "Client", boom)
    out = ingest.ensure_raw(("220",), "2026-10-15", raw_dir=tmp_path)
    assert out == manifest
