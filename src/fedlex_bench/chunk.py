"""One chunk per article: parsed Fedlex acts -> ``data/processed/articles.parquet``.

Columns: sr, article_id, lang, title, text, source_url, consolidation_date, kind, is_repealed.

- ``article_id`` = ``"<SR>:<article number>"`` (e.g. ``"220:257d"``); identical across languages,
  so it is the alignment key. Transitional/final provisions (Schlusstitel, Übergangsbestimmungen)
  live in their own namespace ``"<SR>:u<N>/<number>"`` and have ``kind == "transitional"``.
- Repealed articles are kept with an empty ``text`` and ``is_repealed == True``.
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
    PROJECT_ROOT,
    RAW_DIR,
    ParsedArticle,
    ensure_raw,
    parse_akn,
    raw_path,
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
]
_KIND_RANK = {"article": 0, "transitional": 1}


def make_article_id(sr: str, number: str) -> str:
    return f"{sr}:{number}"


def natural_key(text: str) -> tuple:
    """Natural sort key over digit / non-digit runs: 6 < 6a < 6bis < 10 < 257 < 257a < u2/1."""
    parts = re.split(r"(\d+)", text)
    return tuple((0, int(p), "") if p.isdigit() else (1, 0, p) for p in parts if p != "")


def articles_to_rows(
    parsed: list[ParsedArticle],
    sr: str,
    lang: str,
    source_url: str,
    consolidation_date: str,
) -> list[dict]:
    """Turn parsed articles into chunk rows. Duplicate ids within one (sr, lang) are made unique
    deterministically (``#2``, ``#3`` ...) and logged, never dropped."""
    rows: list[dict] = []
    seen: dict[str, int] = {}
    for a in parsed:
        aid = make_article_id(sr, a.number)
        seen[aid] = seen.get(aid, 0) + 1
        if seen[aid] > 1:
            log.warning("SR %s/%s: duplicate article id %s (eId %s)", sr, lang, aid, a.eid)
            aid = f"{aid}#{seen[aid]}"
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
            path = raw_path(sr, lang, entry["consolidation_date"], f["format"], raw_dir)
            if f["format"] != "xml":
                raise NotImplementedError(f"SR {sr}/{lang}: only Akoma Ntoso XML is parsed")
            parsed = parse_akn(path)
            rows.extend(articles_to_rows(parsed, sr, lang, f["url"], entry["consolidation_date"]))
            log.info(
                "SR %s/%s: %d articles (%d repealed/elided, %d transitional)",
                sr,
                lang,
                len(parsed),
                sum(a.is_repealed for a in parsed),
                sum(a.kind == "transitional" for a in parsed),
            )
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["is_repealed"] = df["is_repealed"].astype(bool)
    return sort_articles(df)


def write_parquet(df: pd.DataFrame, path: Path = ARTICLES_PARQUET) -> str:
    """Write deterministically; returns the sha256 of the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow", compression="zstd")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def log_counts(df: pd.DataFrame) -> None:
    log.info("acts: %d (%s)", df["sr"].nunique(), ", ".join(sorted(df["sr"].unique())))
    for sr, g in df.groupby("sr", sort=True):
        per_lang = {lang: int((g["lang"] == lang).sum()) for lang in sorted(g["lang"].unique())}
        ids = {lang: set(g.loc[g["lang"] == lang, "article_id"]) for lang in per_lang}
        union = set().union(*ids.values())
        common = set.intersection(*ids.values())
        log.info(
            "SR %s: per-language %s | union %d | in all languages %d | mismatches %d",
            sr,
            per_lang,
            len(union),
            len(common),
            len(union) - len(common),
        )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--acts", nargs="*", default=list(DEFAULT_ACTS), help="SR numbers")
    ap.add_argument("--as-of", default=DEFAULT_AS_OF)
    ap.add_argument("--out", type=Path, default=ARTICLES_PARQUET)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    acts = tuple(sorted(args.acts))
    manifest = ensure_raw(acts, args.as_of)
    df = build_articles(manifest, acts=acts)
    digest = write_parquet(df, args.out)
    log_counts(df)
    log.info("wrote %d rows to %s (sha256 %s)", len(df), args.out, digest[:16])


if __name__ == "__main__":
    main()
