"""Same-act article references ("Artikel N" / "article N" / "articolo N") found by code."""

from __future__ import annotations

import pytest

from fedlex_bench.benchmark.refs import same_act_references


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
