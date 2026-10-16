"""Same-act article references ("Artikel N" / "article N" / "articolo N") found by code."""

from __future__ import annotations

import pytest

from fedlex_bench.benchmark.refs import (
    mask_article_references,
    mentions_article_number,
    mentions_sr,
    same_act_references,
)


@pytest.mark.parametrize(
    ("lang", "text", "expected"),
    [
        ("de", "Die Frist richtet sich nach Artikel 5 Absatz 2.", ["5"]),
        ("de", "Es gilt Art. 257d Absatz 1 sinngemäss.", ["257d"]),
        ("de", "Artikel 12bis gilt sinngemäss", ["12bis"]),
        ("de", "nach Artikel 4 dieses Gesetzes", ["4"]),
        (
            "de",
            "nach Artikel 9 und nach Artikel 11 anfechten",
            ["9", "11"],
        ),  # "und nach": not a list
        ("de", "nach Artikel 9 Absatz 1 und nach Artikel 9 Absatz 2", ["9"]),  # de-duplicated
        ("de", "Artikel 333, ausgenommen dessen Absatz 3, und 333a sinngemäss.", ["333"]),
        ("fr", "selon l'article 12, al. 2, pour les contrats", ["12"]),
        ("fr", "conformément à l'article 4 de la présente loi", ["4"]),
        ("it", "secondo l'articolo 7 capoverso 1", ["7"]),
        ("it", "ai sensi dell'articolo 4 della presente legge", ["4"]),
    ],
)
def test_references_to_the_same_act_are_found(lang, text, expected):
    assert same_act_references(text, lang) == expected


@pytest.mark.parametrize(
    ("lang", "text"),
    [
        ("de", "Artikel 50 des Militärstrafgesetzes vom 13. Juni 1927 gilt"),
        ("de", "Rechtsöffnung (Art. 80–84 SchKG), Arrest"),
        ("de", "(Art. 506 OR); c. Gesellschaftsrecht"),
        ("de", "Art. 166 Abs. 2 Ziff. 1 ZGB); c. die Ermächtigung"),
        ("de", "Artikel 16 und 17 des Datenschutzgesetzes vom 25. September 2020"),
        ("de", "nach Artikel 3, 4, 5 oder 6 begeht"),  # a list, not one article
        ("de", "(Art. 1023–1027); die Zahlung"),  # a range
        ("de", "Artikel 50 bis 52 gelten"),  # a range
        ("de", "nach Artikel 1165 OR."),
        ("de", "Artikel 9 und 10 gelten"),  # a list: no single secondary article
        ("fr", "selon l'article 4 du code civil suisse"),
        ("fr", "à l'article 8 CO"),
        ("fr", "l'article 30 de la loi fédérale du 18 décembre 1987"),
        ("fr", "aux articles 4 et 5"),
        ("it", "ai sensi dell'articolo 4 del Codice civile"),
        ("it", "secondo l'articolo 12 della legge federale del 4 dicembre 1947"),
        ("it", "l'articolo 7 LEF"),
        ("it", "degli articoli 4 e 5"),
    ],
)
def test_other_acts_ranges_and_lists_are_not_references(lang, text):
    assert same_act_references(text, lang) == []


def test_unknown_language_is_an_error():
    with pytest.raises(ValueError):
        same_act_references("Artikel 5", "en")


def test_no_reference_in_plain_text():
    assert same_act_references("Der Verkäufer haftet für Mängel.", "de") == []


# ------------------------------------------------------------------ lists with paragraph numbers


@pytest.mark.parametrize(
    ("lang", "text"),
    [
        ("de", "nach Artikel 9 Absatz 1, 10 Absatz 2 gilt"),
        ("de", "Artikel 8 Abs. 2, 3 und 4"),
        ("fr", "selon l'article 9, al. 1, 10, al. 2, de"),
        ("fr", "aux articles 9 al. 1 et 10 al. 2"),
        ("it", "secondo l'articolo 9 capoverso 1, 10 capoverso 2"),
    ],
)
def test_article_lists_with_paragraph_numbers_are_not_single_references(lang, text):
    assert same_act_references(text, lang) == []


def test_a_paragraph_number_alone_is_not_a_list():
    assert same_act_references("nach Artikel 5 Absatz 2, sofern der Vertrag gilt", "de") == ["5"]
    assert same_act_references("selon l'article 12, al. 2, pour les contrats", "fr") == ["12"]


# ------------------------------------------------------------------ masking


def test_references_become_placeholders():
    text = "Es gilt Artikel 5 Absatz 2 sinngemäss; vgl. auch Art. 7."
    assert mask_article_references(text, "de") == (
        "Es gilt [another article] Absatz 2 sinngemäss; vgl. auch [another article]."
    )


def test_the_other_displayed_article_is_named_by_its_label():
    text = "Es gilt Artikel 1170 Absatz 2 und Artikel 7."
    assert mask_article_references(text, "de", {"1170": "B"}) == (
        "Es gilt [Article B] Absatz 2 und [another article]."
    )


@pytest.mark.parametrize(
    ("lang", "text", "expected"),
    [
        ("de", "Artikel 3, 4 und 5 gelten", "[another article] gelten"),
        ("de", "Artikel 3, 4 und 5 gelten", "[Article B] gelten"),
        ("de", "(Art. 80–84 SchKG)", "([another article] SchKG)"),
        ("de", "Artikel 333, ausgenommen Absatz 3", "[another article], ausgenommen Absatz 3"),
        ("de", "nach Artikeln 5 und 6", "nach [another article]"),
        ("fr", "selon l'article 12, al. 2", "selon l'[another article], al. 2"),
        ("fr", "aux articles 4 et 5", "aux [another article]"),
        ("it", "dell'articolo 4 del Codice", "dell'[another article] del Codice"),
        ("it", "degli articoli 4 e 5", "degli [another article]"),
        ("de", "ARTIKEL 5 und art. 6", "[another article] und [another article]"),
        ("de", "Art.5 gilt", "[another article] gilt"),
        ("de", "Artikel 257d gilt", "[another article] gilt"),
    ],
)
def test_mask_variants(lang, text, expected):
    labels = {"4": "B"} if expected.startswith("[Article B]") else {}
    assert mask_article_references(text, lang, labels) == expected


def test_a_list_that_contains_the_other_article_is_named_by_it():
    assert mask_article_references("Artikel 3 und 4 gelten", "de", {"4": "B"}) == (
        "[Article B] gelten"
    )


def test_sr_numbers_are_masked_and_plain_text_is_unchanged():
    assert mask_article_references("(SR 281.1) und SR 220 gelten", "de") == (
        "([SR reference]) und [SR reference] gelten"
    )
    plain = "Der Verkäufer haftet für Mängel nach dem Vertrag. 2 Absatz 3 gilt."
    assert mask_article_references(plain, "de") == plain


def test_masked_text_contains_no_article_number():
    texts = {
        "de": "Artikel 5 und Art. 6a, Artikeln 9 und 10, Art.12 SR 220",
        "fr": "article 5 et art. 6a, articles 9 et 10, art.12 SR 220",
        "it": "articolo 5 e art. 6a, articoli 9 e 10, art.12 SR 220",
    }
    for lang, text in texts.items():
        masked = mask_article_references(text, lang)
        assert not mentions_article_number(masked), (lang, masked)
        assert not mentions_sr(masked), (lang, masked)
        assert masked.count("[another article]") >= 3


def test_mask_unknown_language():
    with pytest.raises(ValueError):
        mask_article_references("Artikel 5", "en")


# ------------------------------------------------------------------ what a writer must not type


@pytest.mark.parametrize(
    "text",
    [
        "Was sagt Artikel 5 dazu?",
        "Was sagt art. 12?",
        "What does ARTICLE 3 say?",
        "Que dit l'article 4 ?",
        "les articles 4 et 5",
        "Cosa dice l'articolo 7?",
        "gli articoli 4 e 5",
        "Und Art.5?",
        "Art 5 OR",
        "Artikeln 5 und 6",
    ],
)
def test_article_numbers_are_detected_in_all_languages(text):
    assert mentions_article_number(text)


@pytest.mark.parametrize(
    "text",
    [
        "Der Artikel ist kurz.",
        "Which article applies to a sale?",
        "Quand l'article est-il applicable ?",
        "Un start 5 minuti",
        "Der Absatz 5 gilt",
        "Die Partikel 9 ist neu",
        "Wie lange dauert die Frist?",
    ],
)
def test_ordinary_words_are_not_article_numbers(text):
    assert not mentions_article_number(text)


def test_sr_numbers_are_detected():
    for text in ("Siehe SR 220", "SR220 sagt", "sr 281.1 regelt", "(SR 235.1)"):
        assert mentions_sr(text), text
    for text in ("USR 5", "Das SR ist neu", "Messrs 5"):
        assert not mentions_sr(text), text
