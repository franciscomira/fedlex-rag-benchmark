"""Chunking variants. Every chunk keeps the ``article_id`` it came from (the scoring key).

- ``article`` (default): one chunk per article, ``chunk_id == article_id``.
- ``paragraph``: one chunk per numbered paragraph (``1 ...``, ``2bis ...``); list items such as
  ``1. ...`` / ``a. ...`` stay with their paragraph. An article without paragraph numbers is one
  chunk. ``chunk_id = "<article_id>#p<ordinal>"``.
- ``window``: fixed windows of 200 words with 50 words overlap (stride 150) over the article body,
  the last window ends at the last word. ``chunk_id = "<article_id>#w<ordinal>"``.

The article title is prefixed to the text of every chunk of every variant (``"<title>\\n<body>"``),
so a paragraph still carries the topic of its article. The index and the returned
``Candidate.text`` use this same text. Only retrieval-eligible articles are chunked
(:func:`fedlex_bench.eligibility.retrieval_eligible`).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from fedlex_bench.eligibility import retrieval_eligible

CHUNKINGS: tuple[str, ...] = ("article", "paragraph", "window")
CORPUS_LANGS: tuple[str, ...] = ("de", "fr", "it")
WINDOW_WORDS = 200
WINDOW_OVERLAP = 50

# A paragraph starts at a line "<digits><optional letters><space>": "1 ", "2bis ", "3quater ".
# List items ("1. x", "a. x", "1) x") have punctuation right after the marker and do not match.
_PARA_START = re.compile(r"^\d+[a-z]*\s")


@dataclass(frozen=True)
class Chunk:
    article_id: str
    lang: str
    chunk_id: str  # article id plus "#p<n>" / "#w<n>" suffix; equal to article_id for "article"
    text: str  # indexed text, title prefixed

    @property
    def key(self) -> tuple[str, str]:
        """Unique within an index that mixes languages."""
        return (self.chunk_id, self.lang)


def split_paragraphs(text: str) -> list[str]:
    parts: list[list[str]] = []
    for line in text.split("\n"):
        if _PARA_START.match(line) or not parts:
            parts.append([line])
        else:
            parts[-1].append(line)
    return ["\n".join(p).strip() for p in parts if "".join(p).strip()]


def split_windows(text: str, size: int = WINDOW_WORDS, overlap: int = WINDOW_OVERLAP) -> list[str]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError(f"need 0 <= overlap < size, got size={size}, overlap={overlap}")
    words = text.split()
    if not words:
        return []
    stride = size - overlap
    out: list[str] = []
    start = 0
    while True:
        out.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            return out
        start += stride


def _with_title(title: str, body: str) -> str:
    title = (title or "").strip()
    return f"{title}\n{body}" if title else body


def chunk_article(
    article_id: str,
    lang: str,
    title: str,
    text: str,
    chunking: str = "article",
    *,
    window_words: int = WINDOW_WORDS,
    window_overlap: int = WINDOW_OVERLAP,
) -> list[Chunk]:
    if chunking == "article":
        return [Chunk(article_id, lang, article_id, _with_title(title, text))]
    if chunking == "paragraph":
        bodies, tag = split_paragraphs(text), "p"
    elif chunking == "window":
        bodies, tag = split_windows(text, window_words, window_overlap), "w"
    else:
        raise ValueError(f"unknown chunking {chunking!r}; expected one of {CHUNKINGS}")
    return [
        Chunk(article_id, lang, f"{article_id}#{tag}{i}", _with_title(title, body))
        for i, body in enumerate(bodies, 1)
    ]


def build_chunks(
    articles: pd.DataFrame,
    chunking: str = "article",
    langs: Iterable[str] = CORPUS_LANGS,
) -> list[Chunk]:
    """Chunks of all retrieval-eligible articles of ``langs`` in deterministic order (act, article
    in natural order, language, chunk ordinal)."""
    if chunking not in CHUNKINGS:
        raise ValueError(f"unknown chunking {chunking!r}; expected one of {CHUNKINGS}")
    wanted = set(langs)
    live = retrieval_eligible(articles)  # already sorted: sr, kind, natural article order, lang
    live = live[live["lang"].isin(wanted)]
    out: list[Chunk] = []
    for r in live.itertuples(index=False):
        out.extend(chunk_article(r.article_id, r.lang, r.title, r.text, chunking))
    return out
