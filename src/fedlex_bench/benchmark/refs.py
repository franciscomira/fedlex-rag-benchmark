"""Find references to other articles of the same act in an article text.

Used to build ``cross_ref`` slots: the gold article explicitly refers to a second article of the
same act ("Artikel N", "article N", "articolo N") and the full answer needs both. The matcher is
deliberately conservative; a missed reference only means one candidate pair less, a wrong one
would put a question about a different act into the benchmark. It therefore skips:

- references followed by a marker of another act within the same clause: a statute, ordinance or
  code name ("des Militärstrafgesetzes", "du code civil", "della legge"), or an abbreviation with
  two capital letters ("OR", "ZGB", "SchKG", "LEF"). "dieses Gesetzes", "la présente loi" and
  "della presente legge" refer to the act itself and are fine;
- ranges ("Art. 80-84", "50 bis 52") and lists ("Artikel 3, 4, 5 oder 6", "articles 4 et 5"),
  where no single second article is meant.

References are not resolved here; ``plan`` resolves them with ``eligibility.resolve_gold``.
"""

from __future__ import annotations

import re

from fedlex_bench.ingest import LATIN_ORDINAL

_ORDINAL = "|".join(sorted(LATIN_ORDINAL, key=len, reverse=True))
_NUMBER = rf"\d+(?:[a-z]?(?:{_ORDINAL})|[a-z])?"
_KEYWORD = {
    "de": r"Artikel|Art\.",
    "fr": r"[Aa]rticles?|[Aa]rt\.",
    "it": r"[Aa]rticol[oi]|[Aa]rt\.",
}
_REFERENCE = {
    lang: re.compile(rf"(?<!\w)(?:{kw})\s+({_NUMBER})(?!\w)") for lang, kw in _KEYWORD.items()
}
# Connectors that turn "Artikel 4" into a range or a list. A digit must follow the connector,
# otherwise "und nach Artikel 11" would count.
_LIST_OR_RANGE = re.compile(r"\s*(?:[–—\-]|bis\b|und\b|oder\b|,|et\b|ou\b|à\b|e\b|o\b|a\b)\s*\d")
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
