"""Find, mask and detect article-number references in texts.

**Finding (``same_act_references``).** Used to build ``cross_ref`` slots: the gold article
explicitly refers to a second article of the same act ("Artikel N", "article N", "articolo N") and
the full answer needs both. The matcher is deliberately conservative; a missed reference only
means one candidate pair less, a wrong one would put a question about a different act into the
benchmark. It therefore skips:

- references followed by a marker of another act within the same clause: a statute, ordinance or
  code name ("des Militärstrafgesetzes", "du code civil", "della legge"), or an abbreviation with
  two capital letters ("OR", "ZGB", "SchKG", "LEF"). "dieses Gesetzes", "la présente loi" and
  "della presente legge" refer to the act itself and are fine;
- ranges ("Art. 80-84", "50 bis 52") and lists ("Artikel 3, 4, 5 oder 6", "articles 4 et 5"),
  where no single second article is meant. The list test looks past sub-designations: in
  "Artikel 9 Absatz 1, 10 Absatz 2" the "10" is another article, so the reference is skipped.

References are not resolved here; ``plan`` resolves them with ``eligibility.resolve_gold``.

**Masking (``mask_article_references``).** The briefs must not show article numbers a writer could
copy into a question. A reference (keyword plus number, a range or a list of numbers) becomes
"[Article B]" when it names the other displayed article of a cross_ref slot, else
"[another article]"; "SR 220" becomes "[SR reference]".

**Detecting (``mentions_article_number``, ``mentions_sr``).** The draft parser rejects questions
and facts that contain an article number in any of the four languages.
"""

from __future__ import annotations

import re

from fedlex_bench.ingest import LATIN_ORDINAL

_ORDINAL = "|".join(sorted(LATIN_ORDINAL, key=len, reverse=True))
_NUMBER = rf"\d+(?:[a-z]?(?:{_ORDINAL})|[a-z])?"
_KEYWORD = {
    "de": r"(?:Artikeln?\s+|Art\.\s*)",
    "fr": r"(?:[Aa]rticles?\s+|[Aa]rt\.\s*)",
    "it": r"(?:[Aa]rticol[oi]\s+|[Aa]rt\.\s*)",
}
_REFERENCE = {lang: re.compile(rf"(?<!\w){kw}({_NUMBER})(?!\w)") for lang, kw in _KEYWORD.items()}
# What may sit between an article number and the next one in a list or range.
_SEPARATOR = r"\s*(?:[–—\-]|,|bis\b|und\b|oder\b|sowie\b|et\b|ou\b|à\b|e\b|o\b|a\b)\s*"
_GROUP = {
    lang: re.compile(
        rf"(?<!\w){kw}(?P<numbers>{_NUMBER}(?:{_SEPARATOR}{_NUMBER})*)(?!\w)", re.IGNORECASE
    )
    for lang, kw in _KEYWORD.items()
}
_BARE_NUMBER = re.compile(_NUMBER)
# Sub-designations that may follow an article number: "Absatz 2", "al. 3", "lettera b", ...
_SUB = (
    r"(?:Absätze|Absatz|Abs\.|Ziffer|Ziff\.|Buchstabe|Bst\.|lit\.|alinéas?|al\.|chiffre|ch\.|"
    r"lettre|let\.|capoversi|capoverso|cpv\.|cifra|lettera)"
)
# Connectors that turn "Artikel 4" into a range or a list. A digit must follow the connector,
# otherwise "und nach Artikel 11" would count. Sub-designations in between are looked past.
_LIST_OR_RANGE = re.compile(
    rf"(?:\s*,?\s*{_SUB}\s*\d+[a-z]?)*"
    r"\s*(?:[–—\-]|bis\b|und\b|oder\b|,|et\b|ou\b|à\b|e\b|o\b|a\b)\s*\d",
    re.IGNORECASE,
)
_SELF = re.compile(
    r"dieses\s+Gesetz\w*|dieser\s+Verordnung|dieses\s+Gesetzbuch\w*"
    r"|(?:la\s+)?pr[ée]sente\s+(?:loi|ordonnance)|pr[ée]sent\s+code"
    r"|(?:della\s+)?presente\s+(?:legge|ordinanza|codice)|di\s+questa\s+legge",
    re.IGNORECASE,
)
_OTHER_ACT = re.compile(
    r"[Gg]esetz|[Vv]erordnung|[Oo]rdnung|[Oo]bligationenrecht|[Kk]onkordat|[Aa]bkommen"
    r"|[Uu]ebereinkommen|Übereinkommen"
    r"|\b[Cc]ode\b|\b[Ll]oi\b|[Oo]rdonnance|[Cc]onvention|[Aa]ccord\b"
    r"|[Cc]odice|\b[Ll]egge\b|[Oo]rdinanza|[Cc]onvenzione|[Aa]ccordo"
    r"|\b(?=\w*[A-Z]\w*[A-Z])\w{2,8}\b"  # abbreviation: OR, ZGB, SchKG, CO, LEF
)
_ABBREVIATION_DOT = re.compile(r"\b(Abs|Ziff|lit|Bst|al|let|cpv|cifra|ch|Art|art|Nr|S|ss|ff)\.")
_CLAUSE_END = re.compile(r"[.;:)]\s")
_WINDOW = 70

# Any "article <number>" in German, French, Italian or English, any case.
_ARTICLE_MENTION = re.compile(r"(?<![a-z])(?:art\.?|artikeln?|articles?|articol[oi])\s*\d", re.I)
_SR_MENTION = re.compile(r"(?<![a-z])sr\s*\d", re.I)
_SR_REFERENCE = re.compile(r"(?<![A-Za-z])SR\s*\d[\d.]*")


def same_act_references(text: str, lang: str) -> list[str]:
    """Article numbers referred to in ``text`` (``"5"``, ``"257d"``, ``"12bis"``), in order of
    appearance without duplicates, after the skips described in the module docstring."""
    if lang not in _REFERENCE:
        raise ValueError(f"unknown language {lang!r}; expected one of {sorted(_REFERENCE)}")
    found: list[str] = []
    for m in _REFERENCE[lang].finditer(text):
        tail = text[m.end() : m.end() + _WINDOW]
        if _LIST_OR_RANGE.match(tail):
            continue
        clause = _CLAUSE_END.split(_ABBREVIATION_DOT.sub(r"\1", tail))[0]
        if _OTHER_ACT.search(_SELF.sub(" ", clause)):
            continue
        if m.group(1) not in found:
            found.append(m.group(1))
    return found


def mask_article_references(text: str, lang: str, labels: dict[str, str] | None = None) -> str:
    """Replace every article-number reference of ``text`` by a placeholder.

    ``labels`` maps an article number (``"1170"``, ``"257d"``) to the label of an article shown
    in the same brief (``"A"``, ``"B"``): a reference group that contains such a number becomes
    ``[Article B]``. Every other group becomes ``[another article]``. ``SR <number>`` becomes
    ``[SR reference]``. Only the keyword and the numbers are replaced; the words around them
    (and sub-designations such as "Absatz 2") stay."""
    if lang not in _GROUP:
        raise ValueError(f"unknown language {lang!r}; expected one of {sorted(_GROUP)}")
    labels = labels or {}

    def replace(match: re.Match) -> str:
        for number in _BARE_NUMBER.findall(match.group("numbers")):
            if number in labels:
                return f"[Article {labels[number]}]"
        return "[another article]"

    masked = _GROUP[lang].sub(replace, text)
    return _SR_REFERENCE.sub("[SR reference]", masked)


def mentions_article_number(text: str) -> bool:
    """True if ``text`` contains "Art./Artikel/article(s)/articolo(i)" followed by a digit."""
    return bool(_ARTICLE_MENTION.search(text))


def mentions_sr(text: str) -> bool:
    """True if ``text`` contains "SR" followed by a number."""
    return bool(_SR_MENTION.search(text))
