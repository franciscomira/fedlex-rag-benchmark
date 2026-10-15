"""DE/FR/IT alignment check over ``articles.parquet``; writes ``docs/phase0_alignment.txt``.

Alignment key: ``article_id`` (``"<SR>:<number>"``). An id is *aligned* when it exists in all
three languages. Reported: counts, alignment rate (aligned ids / union of ids), missing ids per
language with a reason, repealed-flag disagreements, and a seeded random side-by-side sample.

Run: ``uv run python -m fedlex_bench.alignment``.
"""

from __future__ import annotations

import argparse
import logging
import random
import textwrap
from pathlib import Path

import pandas as pd

from fedlex_bench.chunk import ARTICLES_PARQUET, natural_key
from fedlex_bench.ingest import (
    LANGS,
    PROJECT_ROOT,
    RAW_DIR,
    load_manifest,
    raw_path,
    unarticled_units,
)

log = logging.getLogger("fedlex_bench.alignment")

REPORT_PATH = PROJECT_ROOT / "docs" / "phase0_alignment.txt"
SEED = 42
MIN_RATE = 0.95


def ids_by_lang(df: pd.DataFrame, sr: str | None = None) -> dict[str, set[str]]:
    sub = df if sr is None else df[df["sr"] == sr]
    return {lang: set(sub.loc[sub["lang"] == lang, "article_id"]) for lang in LANGS}


def alignment_rate(df: pd.DataFrame, sr: str | None = None) -> tuple[int, int, float]:
    """(ids in all languages, union of ids, rate)."""
    ids = ids_by_lang(df, sr)
    union = set().union(*ids.values())
    common = set.intersection(*ids.values())
    rate = len(common) / len(union) if union else 1.0
    return len(common), len(union), rate


def missing_ids(df: pd.DataFrame) -> list[dict]:
    """Every (sr, lang, article_id) absent in a language but present in another, with a reason."""
    out: list[dict] = []
    flags = {(r.article_id, r.lang): r.is_repealed for r in df.itertuples(index=False)}
    for sr in sorted(df["sr"].unique()):
        ids = ids_by_lang(df, sr)
        union = set().union(*ids.values())
        for lang in LANGS:
            for aid in sorted(union - ids[lang], key=natural_key):
                present = [x for x in LANGS if aid in ids[x]]
                if all(flags[(aid, x)] for x in present):
                    reason = f"repealed/elided in {','.join(present)}; no counterpart in {lang}"
                else:
                    reason = (
                        f"present with text in {','.join(present)}; "
                        f"numbering difference or parsing issue in {lang}"
                    )
                out.append({"sr": sr, "lang": lang, "article_id": aid, "reason": reason})
    return out


def flag_disagreements(df: pd.DataFrame) -> pd.DataFrame:
    """Aligned ids whose ``is_repealed`` flag is not the same in all three languages."""
    p = df.pivot(index="article_id", columns="lang", values="is_repealed").dropna()
    return p[p.nunique(axis=1) > 1]


def length_ratio_outliers(df: pd.DataFrame, ratio: float = 2.5, min_len: int = 200) -> list[str]:
    """Aligned in-force ids whose longest/shortest text-length ratio across DE/FR/IT exceeds
    ``ratio``: a cheap signal for ids that point at different content per language."""
    live = df[~df["is_repealed"]].copy()
    live["n"] = live["text"].str.len()
    p = live.pivot(index="article_id", columns="lang", values="n").dropna()
    r = p.max(axis=1) / p.min(axis=1).clip(lower=1)
    return sorted(r[(r > ratio) & (p.max(axis=1) > min_len)].index, key=natural_key)


def sample_aligned(df: pd.DataFrame, n: int = 5, seed: int = SEED) -> list[str]:
    """Seeded random sample of ids that are in force (not repealed) and present in all langs."""
    live = df[(df["kind"] == "article") & (~df["is_repealed"])]
    ids = ids_by_lang(live)
    candidates = sorted(set.intersection(*ids.values()), key=lambda a: natural_key(a))
    return random.Random(seed).sample(candidates, min(n, len(candidates)))


def _wrap(text: str, width: int, max_chars: int = 700) -> list[str]:
    text = text if len(text) <= max_chars else text[:max_chars].rstrip() + " [...]"
    lines: list[str] = []
    for para in text.split("\n"):
        lines.extend(textwrap.wrap(para, width=width) or [""])
    return lines


def side_by_side(df: pd.DataFrame, article_id: str, width: int = 100) -> str:
    out = [f"=== {article_id} ==="]
    for lang in LANGS:
        r = df[(df["article_id"] == article_id) & (df["lang"] == lang)].iloc[0]
        out.append(f"--- {lang.upper()} | title: {r['title'] or '(none)'}")
        out.extend("    " + ln for ln in _wrap(r["text"], width))
    return "\n".join(out)


def build_report(df: pd.DataFrame, manifest: dict, raw_dir: Path = RAW_DIR) -> str:
    L: list[str] = []
    add = L.append
    add("PHASE 0 ALIGNMENT REPORT (DE/FR/IT)")
    add("=" * 72)
    add("Source: Fedlex SPARQL endpoint + filestore, Akoma Ntoso XML; one chunk per <article>.")
    for sr in sorted(manifest):
        e = manifest[sr]
        add(
            f"SR {sr}: consolidation {e['consolidation_date']} "
            f"(applicable {e['consolidation_date']} .. {e.get('consolidation_end') or 'open'}; "
            f"as_of {e['as_of']})  {e['consolidation']}"
        )
    add("")
    add("1. COUNTS PER ACT AND LANGUAGE")
    add("-" * 72)
    head = ("SR", "lang", "chunks", "body", "transit.", "repealed", "in force")
    add(
        f"{head[0]:<8}{head[1]:<6}"
        + "".join(f"{h:>{w}}" for h, w in zip(head[2:], (8, 8, 10, 10, 10), strict=True))
    )
    for sr in sorted(df["sr"].unique()):
        for lang in LANGS:
            g = df[(df["sr"] == sr) & (df["lang"] == lang)]
            add(
                f"{sr:<8}{lang:<6}{len(g):>8}{int((g['kind'] == 'article').sum()):>8}"
                f"{int((g['kind'] == 'transitional').sum()):>10}{int(g['is_repealed'].sum()):>10}"
                f"{int((~g['is_repealed']).sum()):>10}"
            )
    add("(body = main-text articles; transit. = Schluss-/Übergangsbestimmungen articles;")
    add(" repealed = no text in force: aufgehoben/abrogé/abrogato, gegenstandslos, or '…')")
    add("")
    add("2. ALIGNMENT (article_id present in all 3 languages / union of ids)")
    add("-" * 72)
    add(f"{'SR':<8}{'aligned':>9}{'union':>8}{'rate':>9}")
    failing = []
    for sr in sorted(df["sr"].unique()):
        c, u, r = alignment_rate(df, sr)
        add(f"{sr:<8}{c:>9}{u:>8}{r:>9.2%}")
        if r < MIN_RATE:
            failing.append(sr)
    c, u, r = alignment_rate(df)
    add(f"{'ALL':<8}{c:>9}{u:>8}{r:>9.2%}")
    if r < MIN_RATE:
        failing.append("ALL")
    add(f"Threshold {MIN_RATE:.0%}: {'FAIL for ' + ', '.join(failing) if failing else 'PASS'}")
    live = df[~df["is_repealed"]]
    c2, u2, r2 = alignment_rate(live)
    add(f"Same rate restricted to in-force (non-repealed) chunks: {c2}/{u2} = {r2:.2%}")
    outliers = length_ratio_outliers(df)
    add(
        "Content sanity check: aligned in-force ids with DE/FR/IT text-length ratio > 2.5 "
        f"(longest text > 200 chars): {len(outliers)}"
        + (f" -> {', '.join(outliers[:20])}" if outliers else "")
    )
    add("")
    add("3. MISSING IDS PER LANGUAGE")
    add("-" * 72)
    miss = missing_ids(df)
    if not miss:
        add("None: every article_id exists in DE, FR and IT for every act.")
    for m in miss:
        add(f"SR {m['sr']} {m['lang']}: {m['article_id']}  <- {m['reason']}")
    add("")
    add("4. IDS ALIGNED BUT is_repealed DIFFERS BETWEEN LANGUAGES")
    add("-" * 72)
    dis = flag_disagreements(df)
    if dis.empty:
        add("None.")
    for aid, row in dis.iterrows():
        add(
            f"{aid}: "
            + ", ".join(f"{lang}={'repealed/elided' if row[lang] else 'text'}" for lang in LANGS)
        )
    add("")
    add("5. TRANSITIONAL / FINAL-PROVISION BLOCKS WITHOUT <article> (not chunked)")
    add("-" * 72)
    n_units = 0
    for sr in sorted(manifest):
        e = manifest[sr]
        for lang in LANGS:
            f = e["files"][lang]
            path = raw_path(sr, lang, e["consolidation_date"], f["format"], raw_dir)
            units = unarticled_units(path)
            n_units += len(units)
            for eid, heading in units:
                add(f"SR {sr} {lang}: {eid} | {heading[:90]}")
    if n_units == 0:
        add("None.")
    add("")
    add("6. ARTICLES WITHOUT MARGINAL TITLE (per language; informational)")
    add("-" * 72)
    for sr in sorted(df["sr"].unique()):
        cells = []
        for lang in LANGS:
            g = df[(df["sr"] == sr) & (df["lang"] == lang) & (~df["is_repealed"])]
            cells.append(f"{lang}={int((g['title'] == '').sum())}")
        add(f"SR {sr}: " + ", ".join(cells) + " (in-force chunks with empty title)")
    add("")
    add(f"7. FIVE RANDOM ALIGNED IN-FORCE ARTICLES (seed={SEED})")
    add("-" * 72)
    for aid in sample_aligned(df):
        add(side_by_side(df, aid))
        add("")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parquet", type=Path, default=ARTICLES_PARQUET)
    ap.add_argument("--out", type=Path, default=REPORT_PATH)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = pd.read_parquet(args.parquet)
    report = build_report(df, load_manifest())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8", newline="\n")
    log.info("wrote %s", args.out)
    _, _, rate = alignment_rate(df)
    bad = [sr for sr in sorted(df["sr"].unique()) if alignment_rate(df, sr)[2] < MIN_RATE]
    if rate < MIN_RATE or bad:
        log.error(
            "alignment below %.0f%%: overall %.2f%%, acts %s", MIN_RATE * 100, rate * 100, bad
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
