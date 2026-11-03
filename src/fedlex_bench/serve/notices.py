"""Notices that must appear wherever statute text or answers are shown (demo, MCP, results page).

One definition, so no surface can drift from the others. `tests/test_serve_notices.py` fails if a
surface does not carry them.
"""

from __future__ import annotations

NOT_OFFICIAL_EN = (
    "This is not an official publication. "
    "Only the publication by the Federal Chancellery is authoritative."
)
NOT_OFFICIAL_DE = (
    "Dies ist keine amtliche Veröffentlichung. "
    "Massgebend ist allein die Veröffentlichung durch die Bundeskanzlei."
)
NOT_LEGAL_ADVICE = "Not legal advice."

# Every surface shows these two sentences verbatim.
REQUIRED = (NOT_OFFICIAL_EN, NOT_LEGAL_ADVICE)

SOURCE_NOTE = (
    "Statute texts come from Fedlex (Swiss Federal Chancellery open data) and are shown unaltered; "
    "answers, rankings and comments are produced by this project and are not part of the statute."
)


def notice_text() -> str:
    """The notice block as plain text (MCP responses, CLI output)."""
    return f"{NOT_OFFICIAL_EN} ({NOT_OFFICIAL_DE}) {NOT_LEGAL_ADVICE}"


def has_required(text: str) -> bool:
    return all(sentence in text for sentence in REQUIRED)
