"""Parsing, normalisation and discovery logic. No network access."""

from __future__ import annotations

import pytest

from akn_fixtures import (
    ART_1,
    ART_10_REPEALED,
    ART_257_D,
    ART_PARTIAL_REPEAL,
    TRANSITIONAL,
    full_act,
    note,
    wrap,
)
from fedlex_bench import ingest
from fedlex_bench.ingest import (
    normalize_article_number,
    normalize_article_numbers,
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
        ("Art. 135 bis 149", "135-149"),  # German "to"
        ("Art. 257 e", "257e"),  # Italian letter suffix, not "and"
        ("Art. 12 bis", "12bis"),
        # a conjunction is never turned into a range: the two numbers stay separate
        ("Art. 50 und 51", "50,51"),
        ("Art. 50 et 51", "50,51"),
        ("Art. 50 e 51", "50,51"),
        ("Art. 50 und 52", "50,52"),
        ("Art. 28d à 28f", "28d-28f"),
        ("Art. 2 4", "2-4"),
        ("", ""),
    ],
)
def test_normalize_article_number(raw: str, expected: str) -> None:
    assert normalize_article_number(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Art. 50 und 51", ["50", "51"]),
        ("Art. 50 und 52", ["50", "52"]),  # not adjacent: must not become 50-52
        ("Art. 50 et 52", ["50", "52"]),
        ("Art. 50 e 52", ["50", "52"]),
        ("Art. 663a und 663b", ["663a", "663b"]),
        ("Art. 697l et 697m", ["697l", "697m"]),
        ("Art. 135 à 149", ["135-149"]),  # a range stays one id
        ("Art. 28d a 28f", ["28d-28f"]),
        ("Art. 135–149", ["135-149"]),
        ("Art. 50 und 51 bis 55", ["50", "51-55"]),  # conjunction and range together
        ("Art. 257 e", ["257e"]),  # "e" before nothing is a suffix
        ("Art. 12 bis", ["12bis"]),
        ("Art. 257 d", ["257d"]),
        ("Art. 1", ["1"]),
        ("", []),
    ],
)
def test_normalize_article_numbers_splits_conjunctions(raw: str, expected: list[str]) -> None:
    assert normalize_article_numbers(raw) == expected


def test_conjunction_is_never_a_range() -> None:
    assert "-" not in normalize_article_number("Art. 50 und 52")


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
    assert len(parsed) == 20


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


def test_bodyless_article_with_non_repeal_note_is_repealed() -> None:
    # IT VVG Art. 48-49: no body, and the footnote only says "Introdotto dal ...".
    xml = wrap(
        '<article eId="art_48_49"><num><b>Art. 48 </b>e<b> 49</b>'
        f"{note('Introdotto dal n. I della LF del 19 giu. 2020.')}</num></article>"
    )
    (a,) = parse_akn(xml)
    assert a.number == "48-49"
    assert a.is_repealed
    assert a.text == ""


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


def test_unarticled_transitional_units_are_reported_and_chunked() -> None:
    units = unarticled_units(full_act())
    assert [(eid, n > 0) for eid, _, n in units] == [("disp_u4", True)]
    assert units[0][1].startswith("Uebergangsbestimmung")
    block = {a.number: a for a in parse_akn(full_act())}["u4"]
    assert block.kind == "transitional" and block.text == "Gilt sinngemaess."
    assert block.block_heading == block.title and not block.is_repealed


def test_parse_accepts_str() -> None:
    xml = wrap(ART_10_REPEALED + ART_PARTIAL_REPEAL).decode("utf-8")
    assert [a.number for a in parse_akn(xml)] == ["10", "56"]


def _doc_with_entity(entity_decl: str, body_ref: str) -> bytes:
    """An act whose single article contains ``body_ref`` and whose DOCTYPE declares an entity."""
    art = (
        '<article eId="art_1"><num>Art. 1</num><paragraph eId="art_1/para">'
        f"<content><p>Vor {body_ref} nach</p></content></paragraph></article>"
    )
    doc = wrap(art).decode("utf-8")
    return f"<!DOCTYPE akomaNtoso [{entity_decl}]>{doc}".encode()


def test_external_system_entity_is_not_expanded(tmp_path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET-CONTENT", encoding="utf-8")
    decl = f'<!ENTITY xxe SYSTEM "{secret.as_uri()}">'
    xml = _doc_with_entity(decl, "&xxe;")
    for source in (xml, xml.decode("utf-8")):  # bytes and str
        (a,) = parse_akn(source)
        assert "TOP-SECRET-CONTENT" not in a.text
        assert "TOP-SECRET-CONTENT" not in a.footnotes
        assert a.text == "Vor nach"  # parsing still works


def test_internal_entity_is_not_expanded_either() -> None:
    xml = _doc_with_entity('<!ENTITY inner "EXPANDED-VALUE">', "&inner;")
    (a,) = parse_akn(xml)
    assert "EXPANDED-VALUE" not in a.text


def test_validate_akn_does_not_resolve_external_entities(tmp_path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET-CONTENT", encoding="utf-8")
    xml = _doc_with_entity(f'<!ENTITY xxe SYSTEM "{secret.as_uri()}">', "&xxe;")
    ingest.validate_akn(xml)  # parses without touching the file or the network


def _fallback_article(eid: str, num: str) -> str:
    return (
        f'<article eId="{eid}"><num>{num}</num>'
        '<paragraph><content><p>Aufgehoben.</p></content></paragraph>'
        '<paragraph><content><p>Text bleibt.</p></content></paragraph></article>'
    )


def test_conjunction_citation_without_art_eid_becomes_separate_chunks() -> None:
    # No "art_" eId to take the number from: the printed "Art. 50 und 52" is the only source.
    arts = parse_akn(wrap(_fallback_article("lvl_x", "Art. 50 und 52")))
    assert [a.number for a in arts] == ["50", "52"]  # not one range "50-52"
    assert arts[0].text == arts[1].text and arts[0].text != ""
    assert [a.displayed_number for a in arts] == ["50,52", "50,52"]
    assert len({a.eid for a in arts}) == 1


def test_conjunction_citation_inside_transitional_block_keeps_block_prefix() -> None:
    xml = (
        '<transitional eId="disp_u3"><heading>Uebergang</heading>'
        + _fallback_article("disp_u3/x_1", "Art. 31 et 32")
        + "</transitional>"
    )
    assert [a.number for a in parse_akn(wrap(xml))] == ["u3/31", "u3/32"]


def test_eid_based_number_is_unchanged_by_conjunction_in_num() -> None:
    # The eId is Fedlex's structural id (identical in DE/FR/IT while the printed number varies:
    # DE "Art. 463–464", FR "Art. 463 et 464"), so it stays the id and is not split.
    (a,) = parse_akn(wrap(_fallback_article("art_50_51", "Art. 50 und 51")))
    assert a.number == "50-51"
    assert a.displayed_number == "50,51"


def test_transitional_only_document() -> None:
    (a, b) = parse_akn(wrap(TRANSITIONAL))
    assert (a.number, b.number) == ("u1/1", "u1/12cbis")


# ---------------------------------------------------------------- footnotes column


def _one(xml_inner: str):
    (a,) = parse_akn(wrap(xml_inner))
    return a


def test_footnote_in_paragraph_goes_to_footnotes_not_text(parsed: dict) -> None:
    a = parsed["art_1"]
    assert a.footnotes == "Fassung gemaess X."
    assert "Fassung gemaess" not in a.text


def test_footnote_in_num_is_kept_on_an_article_in_force() -> None:
    a = _one(
        f'<article eId="art_5"><num><b>Art. 5</b>{note("Eingefügt durch Ziff. I des BG.")}</num>'
        '<paragraph eId="art_5/para"><content><p>Text fuenf.</p></content></paragraph></article>'
    )
    assert not a.is_repealed
    assert a.text == "Text fuenf."
    assert a.footnotes == "Eingefügt durch Ziff. I des BG."


def test_multiple_notes_are_in_document_order_one_per_line() -> None:
    a = _one(
        f'<article eId="art_6"><num><b>Art. 6</b>{note("Note num.")}</num>'
        f"<heading>Titel{note('Note heading.')}</heading>"
        '<paragraph eId="art_6/para_1"><num>1</num><content>'
        f"<p>Eins{note('Note eins.')} und mehr.</p></content></paragraph>"
        '<paragraph eId="art_6/para_2"><num>2</num><content>'
        f"<p>Zwei.{note('Note zwei.')}</p></content></paragraph></article>"
    )
    assert a.footnotes.split("\n") == ["Note num.", "Note heading.", "Note eins.", "Note zwei."]
    assert a.text == "1 Eins und mehr.\n2 Zwei."


def test_note_with_several_paragraphs_and_inline_markup_is_one_line() -> None:
    a = _one(
        '<article eId="art_7"><num><b>Art. 7</b></num>'
        '<paragraph eId="art_7/para"><content><p>Text.'
        "<authorialNote><p> SR <i>220</i>; BBl 2020 1</p><p>\n  Zweiter Absatz.</p>"
        "</authorialNote></p></content></paragraph></article>"
    )
    assert a.footnotes == "SR 220; BBl 2020 1 Zweiter Absatz."
    assert "\n" not in a.footnotes and a.text == "Text."


def test_article_without_notes_has_empty_footnotes(parsed: dict) -> None:
    assert parsed["art_257_d"].footnotes == ""
    assert parsed["art_56"].footnotes == ""


def test_repealed_article_keeps_notes_but_empty_text(parsed: dict) -> None:
    a = parsed["art_10"]
    assert a.is_repealed and a.text == ""
    assert a.footnotes.startswith("Aufgehoben durch Anhang 1")
    assert a.footnotes == a.repeal_note  # single note: same cleaned text
    two = parsed["art_40_g"]
    assert two.footnotes == two.repeal_note and "Eingefügt" in two.footnotes


def test_repealed_article_with_several_notes_keeps_all() -> None:
    a = _one(
        f'<article eId="art_8"><num><b>Art. 8</b>{note("Eingefuegt durch A.")}'
        f"{note('Aufgehoben durch B.')}</num></article>"
    )
    assert a.is_repealed and a.text == ""
    assert a.footnotes == "Eingefuegt durch A.\nAufgehoben durch B."
    assert a.repeal_note == "Aufgehoben durch B."


def test_it_split_number_note_in_heading_lands_in_footnotes(parsed: dict) -> None:
    a = parsed["art_697_l_697_m"]
    assert a.is_repealed and a.text == "" and a.title == ""
    assert a.footnotes == "Abrogati dall’all. n. 2 della L del 26 set. 2025."


def test_transitional_article_and_unarticled_block_footnotes() -> None:
    art = _one(
        '<proviso eId="disp_u1"><heading>Schlusstitel</heading>'
        f'<article eId="disp_u1/art_3"><num><b>Art. 3</b>{note("Num-Note.")}</num>'
        '<paragraph eId="disp_u1/art_3/para"><content><p>Uebergang.</p></content></paragraph>'
        "</article></proviso>"
    )
    assert art.kind == "transitional" and art.text == "Uebergang." and art.footnotes == "Num-Note."
    block = _one(
        f'<transitional eId="disp_u4"><heading>Uebergang{note("Heading-Note.")}</heading>'
        '<paragraph eId="disp_u4/para"><content>'
        f"<p>Gilt sinngemaess.{note('Absatz-Note.')}</p></content></paragraph></transitional>"
    )
    assert block.number == "u4" and block.text == "Gilt sinngemaess."
    assert block.title == "Uebergang"
    assert block.footnotes == "Heading-Note.\nAbsatz-Note."


def test_repealed_unarticled_block_keeps_notes() -> None:
    block = _one(
        '<transitional eId="disp_u7"><heading>Alt</heading>'
        '<paragraph eId="disp_u7/para"><content>'
        f"<p>Aufgehoben{note('Aufgehoben durch Ziff. I.')}</p></content></paragraph></transitional>"
    )
    assert block.is_repealed and block.text == ""
    assert block.footnotes == "Aufgehoben durch Ziff. I."


def test_footnotes_are_deterministic_and_text_unchanged_by_notes() -> None:
    with_notes = parse_akn(full_act())
    assert [a.footnotes for a in with_notes] == [a.footnotes for a in parse_akn(full_act())]
    plain = _one(
        '<article eId="art_9"><num><b>Art. 9</b></num><paragraph eId="art_9/para"><content>'
        "<p>Text.</p></content></paragraph></article>"
    )
    noted = _one(
        f'<article eId="art_9"><num><b>Art. 9</b>{note("N.")}</num><paragraph eId="art_9/para">'
        f"<content><p>Text.{note('M.')}</p></content></paragraph></article>"
    )
    assert plain.text == noted.text == "Text." and plain.footnotes == ""


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
