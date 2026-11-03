"""``--check-numbers``: flag numbers typed by hand into the template.

The check reads the template with every placeholder blanked out, so a number produced by
a placeholder is never flagged. Any other number-looking token (digits, optionally with a decimal
point or comma groups, a trailing ``%`` or letters, as in ``0.81``, ``4,073``, ``81%``, ``257d``)
must be listed in the allowlist file, or it is reported with its line.

Ignored by rule (no allowlist entry needed):

* fenced code blocks, inline code spans and link targets;
* years (1900 to 2099) and dates (``2026-11-03``, ``3.11.2026``, ``3 November``, ``November 3``);
* section and structure numbers: ``Phase 4``, ``Section 2``, ``Figure 1``, ``Table 2``, ``Step 3``,
  numbers leading a heading (``## 3. Results``) and numbered-list markers (``1.``);
* words that contain digits: names and versions such as ``T1``, ``P2``, ``e5-small``, ``int8``,
  ``gpt-6.1-sol``, ``claude-haiku-5-5``, and metric names such as ``recall@10``.

Allowlist file (docs/writeup/numbers_allowlist.txt): one entry per line. ``#`` starts a comment.
``220`` allows the token ``220`` anywhere; ``15 @ traps`` allows ``15`` only on lines that contain
the text ``traps``. Tokens are compared as written (``81%`` and ``81`` are different tokens).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from fedlex_bench.writeup.placeholders import CODE_SPAN, FENCE, TOKEN

MONTHS = ("January|February|March|April|May|June|July|August|September|October|November|"
          "December")
NUMBER = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)*(?:%|[A-Za-z]+)?")
RULES = (
    re.compile(r"\]\([^)]*\)"),  # link targets
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b"),
    re.compile(rf"\b\d{{1,2}}\s+(?:{MONTHS})\b(?:\s+(?:19|20)\d{{2}})?"),
    re.compile(rf"\b(?:{MONTHS})\s+\d{{1,2}}\b(?:,?\s+(?:19|20)\d{{2}})?"),
    re.compile(r"\b(?:Phase|Section|Sections|Figure|Fig\.|Table|Step|Part|Chapter)s?\s+"
               r"\d+(?:\.\d+)*", re.I),
    re.compile(r"§\s*\d+(?:\.\d+)*"),
    re.compile(r"\b(?:19|20)\d{2}\b"),
    re.compile(r"\b\w+@\d+\b"),  # recall@10
)
WORD_WITH_DIGIT = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:[-.][A-Za-z0-9_]+)*")
HEADING_NUMBER = re.compile(r"^(\s{0,3}#{1,6}\s+)\d+(?:\.\d+)*\.?(?=\s)")
LIST_MARKER = re.compile(r"^(\s*)\d+[.)](?=\s)")


@dataclass(frozen=True)
class Offence:
    line: int
    token: str
    text: str


def load_allowlist(path: Path | None) -> list[tuple[str, str | None]]:
    """Entries as (token, required line text or None)."""
    if path is None or not path.exists():
        return []
    entries = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        token, _, context = line.partition("@ ")
        entries.append((token.strip(), context.strip() or None))
    return entries


def _words_with_digits(line: str) -> str:
    return WORD_WITH_DIGIT.sub(lambda m: " " if re.search(r"\d", m.group(0)) else m.group(0),
                               line)


def numbers_in(line: str) -> list[str]:
    """Number-looking tokens of one tokenized template line after the ignore rules."""
    line = CODE_SPAN.sub(" ", line)
    line = TOKEN.sub(" ", line)
    line = HEADING_NUMBER.sub(r"\1", line)
    line = LIST_MARKER.sub(r"\1", line)
    for rule in RULES:
        line = rule.sub(" ", line)
    line = _words_with_digits(line)
    return [m.group(0) for m in NUMBER.finditer(line)]


def check_numbers(lines: list[str], allowlist: list[tuple[str, str | None]]) -> list[Offence]:
    """Offences in the tokenized template lines (line numbers are those of the template)."""
    out: list[Offence] = []
    in_fence: str | None = None
    for number, line in enumerate(lines, 1):
        fence = FENCE.match(line)
        if in_fence is not None:
            if fence and fence.group(1)[0] == in_fence[0]:
                in_fence = None
            continue
        if fence:
            in_fence = fence.group(1)
            continue
        for token in numbers_in(line):
            allowed = any(
                token == entry and (context is None or context in line)
                for entry, context in allowlist
            )
            if not allowed:
                out.append(Offence(number, token, TOKEN.sub("{{...}}", line).strip()))
    return out
