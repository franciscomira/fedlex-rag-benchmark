"""One chunk per article: parsed Fedlex acts -> ``data/processed/articles.parquet``.

Columns: sr, article_id, lang, title, text, source_url, consolidation_date, kind, is_repealed,
is_historical, block_heading, footnotes.

- ``article_id`` = ``"<SR>:<article number>"`` (e.g. ``"220:257d"``); identical across languages,
  so it is the alignment key. Transitional/final provisions (Schlusstitel, Uebergangsbestimmungen)
  live in their own namespace ``"<SR>:u<N>/<number>"`` (an article of block ``N``) or
  ``"<SR>:u<N>"`` (a block without articles, chunked whole) and have ``kind == "transitional"``.
  ``u<N>`` is positional per snapshot (the N-th ``disp_`` block of that consolidation) and can
  shift in a later consolidation; ``block_heading`` carries the stable human-readable identity.
- ``is_repealed``: no text in force (repealed, obsolete, or elided). Such chunks are kept with an
  empty ``text``.
- ``is_historical``: the chunk quotes superseded wording (ZGB "Wortlaut der frueheren
  Bestimmungen"). Not law in force: exclude it from retrieval together with repealed chunks.
- ``block_heading``: heading of the enclosing ``disp_`` block for transitional chunks, else "".
- ``footnotes``: all footnotes (``<authorialNote>``) of the chunk in document order, one cleaned
  note per line (joined with a newline), "" when there are none. ``text`` never contains footnote
  text; the notes are kept verbatim in their own column because the Fedlex reuse terms forbid
  altering content. Filled for repealed chunks too (their ``text`` is empty).
- Output is sorted by (sr, kind, natural article order, lang) and written deterministically.

Run: ``uv run python -m fedlex_bench.chunk`` (downloads missing raw files first, idempotent).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
from pathlib import Path

import pandas as pd

from fedlex_bench.ingest import (
    DEFAULT_ACTS,
    DEFAULT_AS_OF,
    LANGS,
    LATIN_ORDINAL,
    PROJECT_ROOT,
    RAW_DIR,
    ParsedArticle,
    ensure_raw,
    file_path,
    parse_akn,
)

log = logging.getLogger("fedlex_bench.chunk")

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTICLES_PARQUET = PROCESSED_DIR / "articles.parquet"
COLUMNS = [
    "sr",
    "article_id",
    "lang",
    "title",
    "text",
    "source_url",
    "consolidation_date",
    "kind",
    "is_repealed",
    "is_historical",
    "block_heading",
    "footnotes",
]
_KIND_RANK = {"article": 0, "transitional": 1}
_LATIN_RE = re.compile("|".join(sorted(LATIN_ORDINAL, key=len, reverse=True)))


def make_article_id(sr: str, number: str) -> str:
    return f"{sr}:{number}"


def _alpha_key(run: str) -> tuple:
    """Key for a non-digit run: single letters first (a < b), then Latin multiplicatives in
    ordinal order (bis < ter < quater < quinquies < ... < decies), then anything else."""
    key: list[tuple] = []
    pos = 0
    while pos < len(run):
        m = _LATIN_RE.match(run, pos)
        if m:
            key.append((1, LATIN_ORDINAL[m.group(0)], ""))
            pos = m.end()
        else:
            ch = run[pos]
            key.append((0, 0, ch) if ch.isalpha() else (2, 0, ch))
            pos += 1
    return tuple(key)


def natural_key(text: str) -> tuple:
    """Natural sort key over digit / non-digit runs:
    ``6 < 6a < 6bis < 6ter < 6quater < 10 < 257 < 257a < 257d``; Latin suffixes by ordinal."""
    parts = re.split(r"(\d+)", text)
    return tuple((0, int(p), ()) if p.isdigit() else (1, 0, _alpha_key(p)) for p in parts if p)


def articles_to_rows(
    parsed: list[ParsedArticle],
    sr: str,
    lang: str,
    source_url: str,
    consolidation_date: str,
) -> list[dict]:
    """Turn parsed articles into chunk rows. A duplicate id within one (sr, lang) would make the
    alignment key ambiguous, so it is an error (nothing is renamed or dropped silently)."""
    rows: list[dict] = []
    seen: dict[str, str] = {}
    for a in parsed:
        aid = make_article_id(sr, a.number)
        if aid in seen:
            raise ValueError(
                f"SR {sr}/{lang}: duplicate article id {aid} (eIds {seen[aid]!r} and {a.eid!r})"
            )
        seen[aid] = a.eid
        rows.append(
            {
                "sr": sr,
                "article_id": aid,
                "lang": lang,
                "title": a.title,
                "text": a.text,
                "source_url": f"{source_url}#{a.eid}" if a.eid else source_url,
                "consolidation_date": consolidation_date,
                "kind": a.kind,
                "is_repealed": a.is_repealed,
                "is_historical": a.is_historical,
                "block_heading": a.block_heading,
                "footnotes": a.footnotes,
            }
        )
    return rows


def sort_articles(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic order: sr, kind, natural article order, lang."""

    def key(i: int) -> tuple:
        r = df.iloc[i]
        number = r["article_id"].partition(":")[2]
        return (
            natural_key(r["sr"]),
            _KIND_RANK.get(r["kind"], 9),
            natural_key(number),
            r["article_id"],
            r["lang"],
        )

    order = sorted(range(len(df)), key=key)
    return df.iloc[order].reset_index(drop=True)


def build_articles(
    manifest: dict,
    raw_dir: Path = RAW_DIR,
    acts: tuple[str, ...] = DEFAULT_ACTS,
    langs: tuple[str, ...] = LANGS,
) -> pd.DataFrame:
    rows: list[dict] = []
    for sr in sorted(acts):
        entry = manifest[sr]
        for lang in sorted(langs):
            f = entry["files"][lang]
            path = file_path(entry, lang, raw_dir)
            if f["format"] != "xml":
                raise NotImplementedError(f"SR {sr}/{lang}: only Akoma Ntoso XML is parsed")
            parsed = parse_akn(path)
            rows.extend(articles_to_rows(parsed, sr, lang, f["url"], entry["consolidation_date"]))
            log.info(
                "SR %s/%s: %d chunks (%d repealed/elided, %d historical, %d transitional, "
                "%d with footnotes)",
                sr,
                lang,
                len(parsed),
                sum(a.is_repealed for a in parsed),
                sum(a.is_historical for a in parsed),
                sum(a.kind == "transitional" for a in parsed),
                sum(bool(a.footnotes) for a in parsed),
            )
    df = pd.DataFrame(rows, columns=COLUMNS)
    for col in ("is_repealed", "is_historical"):
        df[col] = df[col].astype(bool)
    return sort_articles(df)


def write_parquet(df: pd.DataFrame, path: Path = ARTICLES_PARQUET) -> str:
    """Write deterministically; returns the sha256 of the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow", compression="zstd")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def in_force(df: pd.DataFrame) -> pd.DataFrame:
    """Chunks with text in force: neither repealed/elided nor historical wording."""
    return df[~df["is_repealed"] & ~df["is_historical"]]


def log_counts(df: pd.DataFrame) -> None:
    log.info("acts: %d (%s)", df["sr"].nunique(), ", ".join(sorted(df["sr"].unique())))
    for sr, g in df.groupby("sr", sort=True):
        per_lang = {lang: int((g["lang"] == lang).sum()) for lang in sorted(g["lang"].unique())}
        ids = {lang: set(g.loc[g["lang"] == lang, "article_id"]) for lang in per_lang}
        union = set().union(*ids.values())
        common = set.intersection(*ids.values())
        live = in_force(g)
        log.info(
            "SR %s: per-language %s | union %d | in all languages %d | mismatches %d | "
            "in force (not repealed, not historical) %s",
            sr,
            per_lang,
            len(union),
            len(common),
            len(union) - len(common),
            {lang: int((live["lang"] == lang).sum()) for lang in sorted(per_lang)},
        )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--acts", nargs="*", default=list(DEFAULT_ACTS), help="SR numbers")
    ap.add_argument("--as-of", default=DEFAULT_AS_OF)
    ap.add_argument("--out", type=Path, default=ARTICLES_PARQUET)
    ap.add_argument("--refresh", action="store_true", help="redo SPARQL discovery")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    acts = tuple(sorted(args.acts))
    manifest = ensure_raw(acts, args.as_of, refresh=args.refresh)
    df = build_articles(manifest, acts=acts)
    digest = write_parquet(df, args.out)
    log_counts(df)
    log.info("wrote %d rows to %s (sha256 %s)", len(df), args.out, digest)


if __name__ == "__main__":
    main()
