"""DE/FR/IT alignment check over ``articles.parquet``; writes ``docs/phase0_alignment.txt``.

Two different questions are answered, and they must not be confused:

1. **Id alignment** (``alignment_rate``): is every ``article_id`` (``"<SR>:<number>"``) present in
   all three languages? The id derives from the ``eId`` that Fedlex assigns from the *structure*
   of the act, so 100% id alignment shows that the three XML files have the same structure. It
   does not show that the texts behind the ids are translations of each other.
2. **Content-level checks** (sections 5a-5c of the report): eId vs printed article number,
   paragraph/item counts per article across languages, and DE/FR/IT text-length ratios.

Section 8 of the report is informational: how many chunks carry footnotes (``footnotes`` column)
per act and language.

The command exits non-zero when id alignment drops below 95% overall or for any act. It reports
on the acts present in the parquet only, and refuses to run (exit code 2) when the parquet does
not belong to the manifest: a different ``source_url`` or consolidation date, a missing raw file,
or a raw file whose sha256 differs from the manifest (checked before the raw XML is re-parsed).

Run: ``uv run python -m fedlex_bench.alignment``.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import random
import textwrap
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from fedlex_bench.chunk import ARTICLES_PARQUET, in_force, make_article_id, natural_key
from fedlex_bench.ingest import (
    LANGS,
    MANIFEST_PATH,
    PROJECT_ROOT,
    RAW_DIR,
    file_path,
    load_manifest,
    parse_akn,
    sha256_of,
    unarticled_units,
    unchunked_components,
)

log = logging.getLogger("fedlex_bench.alignment")

REPORT_PATH = PROJECT_ROOT / "docs" / "phase0_alignment.txt"
SEED = 42
MIN_RATE = 0.95
LENGTH_RATIO = 1.6
LENGTH_MIN_CHARS = 200
MAX_LISTED = 40  # cap on lines per listing in the report (totals are always exact)


# ------------------------------------------------------------------ inputs


class InputMismatchError(RuntimeError):
    """The parquet and the manifest/raw files do not describe the same snapshot."""


def acts_in(df: pd.DataFrame) -> list[str]:
    """SR numbers present in the parquet, in natural order."""
    return sorted(df["sr"].unique(), key=natural_key)


def verify_inputs(df: pd.DataFrame, manifest: dict, raw_dir: Path = RAW_DIR) -> None:
    """Check, for every act and language in ``df``, that the manifest describes the snapshot the
    parquet was built from, before any raw XML is parsed again:

    - the act and its three languages are in the manifest;
    - every ``source_url`` of the parquet is the manifest URL plus a ``#<eId>`` fragment, and the
      consolidation date agrees;
    - the raw file exists and its sha256 equals the one recorded in the manifest.

    Raises :class:`InputMismatchError` listing the first problem found."""
    for sr in acts_in(df):
        entry = manifest.get(sr)
        if entry is None:
            raise InputMismatchError(f"SR {sr} is in the parquet but not in the manifest")
        for lang in LANGS:
            what = f"SR {sr}/{lang}"
            f = entry.get("files", {}).get(lang)
            g = df[(df["sr"] == sr) & (df["lang"] == lang)]
            if f is None or g.empty:
                raise InputMismatchError(f"{what}: missing in the manifest or in the parquet")
            urls = {u.partition("#")[0] for u in g["source_url"]}
            if urls != {f["url"]}:
                raise InputMismatchError(
                    f"{what}: source_url in the parquet {sorted(urls)} differs from the manifest "
                    f"url {f['url']!r}; rebuild the parquet with fedlex_bench.chunk"
                )
            dates = set(g["consolidation_date"])
            if dates != {entry["consolidation_date"]}:
                raise InputMismatchError(
                    f"{what}: consolidation_date in the parquet {sorted(dates)} differs from the "
                    f"manifest {entry['consolidation_date']!r}"
                )
            path = file_path(entry, lang, raw_dir)
            if not path.exists():
                raise InputMismatchError(f"{what}: raw file {path} is missing")
            expected = f.get("sha256")
            if expected and sha256_of(path) != expected:
                raise InputMismatchError(
                    f"{what}: raw file sha256 {sha256_of(path)[:12]}... differs from the manifest "
                    f"sha256 {expected[:12]}... ({path.name})"
                )
            if not expected:
                log.warning("%s: no sha256 recorded in the manifest, raw file not verified", what)


# ------------------------------------------------------------------ id alignment


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


def flag_disagreements(df: pd.DataFrame, column: str = "is_repealed") -> pd.DataFrame:
    """Aligned ids whose boolean ``column`` is not the same in all three languages."""
    p = df.pivot(index="article_id", columns="lang", values=column).dropna()
    return p[p.nunique(axis=1) > 1]


# ------------------------------------------------------------------ content-level checks


def structure_table(
    manifest: dict, raw_dir: Path = RAW_DIR, acts: Iterable[str] | None = None
) -> pd.DataFrame:
    """Per (sr, lang, article_id): eId, printed number and paragraph/item counts, re-parsed from
    the raw XML (these fingerprints are not stored in the parquet). ``acts`` limits it to the
    acts of the parquet (default: every act of the manifest)."""
    rows: list[dict] = []
    for sr in sorted(manifest if acts is None else acts, key=natural_key):
        for lang in LANGS:
            for a in parse_akn(file_path(manifest[sr], lang, raw_dir)):
                rows.append(
                    {
                        "sr": sr,
                        "lang": lang,
                        "article_id": make_article_id(sr, a.number),
                        "eid": a.eid,
                        "number": a.number,
                        "displayed_number": a.displayed_number,
                        "n_paragraphs": a.n_paragraphs,
                        "n_items": a.n_items,
                    }
                )
    return pd.DataFrame(rows)


def displayed_number_mismatches(struct: pd.DataFrame) -> pd.DataFrame:
    """Articles whose printed number (normalised) differs from the eId-derived number. Block
    pseudo-articles (no printed number) are skipped. A printed conjunction ("Art. 50 und 51") is
    displayed as ``50,51`` and compares equal to the eId-derived ``50-51``: the eId cannot tell a
    conjunction from a range, and the three languages print the same eId differently."""
    s = struct[struct["displayed_number"] != ""]
    tail = s["number"].str.rsplit("/", n=1).str[-1]
    return s[s["displayed_number"].str.replace(",", "-", regex=False) != tail]


def structure_mismatches(struct: pd.DataFrame) -> pd.DataFrame:
    """Aligned ids whose (paragraph, item) counts are not identical in DE, FR and IT.
    Columns: one ``de/fr/it`` string ``"<paragraphs>p/<items>i"`` per language."""
    s = struct.assign(
        fp=struct["n_paragraphs"].astype(str) + "p/" + struct["n_items"].astype(str) + "i"
    )
    p = s.pivot(index="article_id", columns="lang", values="fp").dropna()
    return p[p.nunique(axis=1) > 1]


def length_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Aligned in-force ids with their text lengths per language and the longest/shortest ratio."""
    live = in_force(df).copy()
    live["n"] = live["text"].str.len()
    p = live.pivot(index="article_id", columns="lang", values="n").dropna()
    p["ratio"] = p[list(LANGS)].max(axis=1) / p[list(LANGS)].min(axis=1).clip(lower=1)
    p["longest"] = p[list(LANGS)].max(axis=1)
    return p


def length_ratio_outliers(
    df: pd.DataFrame, ratio: float = LENGTH_RATIO, min_len: int = LENGTH_MIN_CHARS
) -> list[str]:
    """Aligned in-force ids whose longest/shortest DE/FR/IT text-length ratio exceeds ``ratio``
    (longest text above ``min_len`` chars): a cheap signal for content that differs by language.
    Returned in natural article order."""
    p = length_ratios(df)
    return sorted(p[(p["ratio"] > ratio) & (p["longest"] > min_len)].index, key=natural_key)


def sample_aligned(df: pd.DataFrame, n: int = 5, seed: int = SEED) -> list[str]:
    """Seeded random sample of body-article ids in force (not repealed, not historical) and
    present in all languages."""
    live = in_force(df[df["kind"] == "article"])
    ids = ids_by_lang(live)
    candidates = sorted(set.intersection(*ids.values()), key=natural_key)
    return random.Random(seed).sample(candidates, min(n, len(candidates)))


# ------------------------------------------------------------------ report


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


def footnote_coverage_lines(df: pd.DataFrame) -> list[str]:
    """Per act and language: chunks with footnotes / chunks. ``differ`` counts aligned ids where
    having footnotes is not the same in DE, FR and IT (informational, not a failure)."""
    if "footnotes" not in df.columns:
        return ["(no `footnotes` column in this parquet: rebuild with fedlex_bench.chunk)"]
    has = df.assign(has_fn=df["footnotes"] != "")
    out = [f"{'SR':<8}" + "".join(f"{lang:>14}" for lang in LANGS) + f"{'differ':>9}"]
    for sr in [*sorted(df["sr"].unique()), None]:
        g = has if sr is None else has[has["sr"] == sr]
        cells = []
        for lang in LANGS:
            gl = g[g["lang"] == lang]
            cells.append(f"{int(gl['has_fn'].sum())}/{len(gl)}")
        p = g.pivot(index="article_id", columns="lang", values="has_fn").dropna()
        differ = int((p.nunique(axis=1) > 1).sum()) if len(p) else 0
        out.append(
            f"{'ALL' if sr is None else sr:<8}"
            + "".join(f"{c:>14}" for c in cells)
            + f"{differ:>9}"
        )
    return out


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report(
    df: pd.DataFrame,
    manifest: dict,
    raw_dir: Path = RAW_DIR,
    parquet_path: Path | None = None,
) -> str:
    verify_inputs(df, manifest, raw_dir)  # raises InputMismatchError before any raw XML is read
    acts = acts_in(df)
    L: list[str] = []
    add = L.append
    rule = "-" * 72
    add("PHASE 0 ALIGNMENT REPORT (DE/FR/IT)")
    add("=" * 72)
    add("Source: Fedlex SPARQL endpoint + filestore, Akoma Ntoso XML; one chunk per <article>.")
    add("")
    add("PROVENANCE")
    add(rule)
    for sr in acts:
        e = manifest[sr]
        add(
            f"SR {sr}: consolidation {e['consolidation_date']} "
            f"(applicable {e['consolidation_date']} .. {e.get('consolidation_end') or 'open'}; "
            f"as_of {e['as_of']})  {e['consolidation']}"
        )
        for lang in LANGS:
            f = e["files"][lang]
            sha, size = f.get("sha256", "(not recorded)"), f.get("bytes", "?")
            add(f"    {lang}: sha256 {sha}  {size} bytes")
    if parquet_path is not None and parquet_path.exists():
        add(f"articles.parquet: sha256 {_sha256_file(parquet_path)}  ({len(df)} rows)")
    add("")

    add("1. COUNTS PER ACT AND LANGUAGE")
    add(rule)
    head = ("SR", "lang", "chunks", "body", "transit.", "repealed", "historic.", "in force")
    widths = (8, 6, 8, 8, 10, 10, 10, 10)
    add(
        "".join(
            f"{h:<{w}}" if i < 2 else f"{h:>{w}}"
            for i, (h, w) in enumerate(zip(head, widths, strict=True))
        )
    )
    for sr in sorted(df["sr"].unique()):
        for lang in LANGS:
            g = df[(df["sr"] == sr) & (df["lang"] == lang)]
            cells = (
                sr,
                lang,
                len(g),
                int((g["kind"] == "article").sum()),
                int((g["kind"] == "transitional").sum()),
                int(g["is_repealed"].sum()),
                int(g["is_historical"].sum()),
                len(in_force(g)),
            )
            add(
                "".join(
                    f"{c:<{w}}" if i < 2 else f"{c:>{w}}"
                    for i, (c, w) in enumerate(zip(cells, widths, strict=True))
                )
            )
    add("body = main-text articles; transit. = Schluss-/Uebergangsbestimmungen chunks (articles of")
    add("a disp_ block, plus blocks without articles chunked whole). repealed = no text in force")
    add("(aufgehoben/abroge/abrogato, gegenstandslos, '...' elision, amendment-of-other-act stub).")
    add("historic. = block quoting superseded wording (ZGB 'Wortlaut der frueheren Bestimmungen').")
    add("in force = neither repealed nor historical.")
    add("")

    add("2. ID ALIGNMENT (article_id present in all 3 languages / union of ids)")
    add(rule)
    add("What this shows: the three XML files expose the same article structure (same eIds, which")
    add(
        "Fedlex derives from the structure of the act). What it does NOT show: that the text behind"
    )
    add("an id is the same provision in each language. See the content-level checks in section 5.")
    add("")
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
    verdict = "FAIL for " + ", ".join(failing) if failing else "PASS"
    add(f"Threshold {MIN_RATE:.0%} on id alignment: {verdict}")
    live = in_force(df)
    c2, u2, r2 = alignment_rate(live)
    add(f"Same rate restricted to in-force chunks: {c2}/{u2} = {r2:.2%}")
    add("")

    add("3. MISSING IDS PER LANGUAGE")
    add(rule)
    miss = missing_ids(df)
    if not miss:
        add("None: every article_id exists in DE, FR and IT for every act.")
    for m in miss:
        add(f"SR {m['sr']} {m['lang']}: {m['article_id']}  <- {m['reason']}")
    add("")

    add("4. IDS ALIGNED BUT FLAGS DIFFER BETWEEN LANGUAGES")
    add(rule)
    any_flag = False
    for col in ("is_repealed", "is_historical"):
        dis = flag_disagreements(df, col)
        for aid, row in dis.iterrows():
            any_flag = True
            add(f"{col} {aid}: " + ", ".join(f"{lang}={bool(row[lang])}" for lang in LANGS))
    if not any_flag:
        add("None: is_repealed and is_historical agree across DE/FR/IT for every aligned id.")
    add("")

    struct = structure_table(manifest, raw_dir, acts)
    add("5. CONTENT-LEVEL CHECKS (aligned ids only)")
    add(rule)
    add("5a. eId-derived number vs number printed in <num> (per language)")
    dn = displayed_number_mismatches(struct)
    add(f"    mismatches: {len(dn)} of {int((struct['displayed_number'] != '').sum())} articles")
    for r in dn.sort_values(["sr", "lang", "article_id"]).head(MAX_LISTED).itertuples():
        add(
            f"    SR {r.sr} {r.lang}: eId {r.eid} -> id number '{r.number}', "
            f"printed '{r.displayed_number}'"
        )
    if len(dn) > MAX_LISTED:
        add(f"    ... {len(dn) - MAX_LISTED} more")
    add("")
    add("5b. paragraph / item counts per article, DE vs FR vs IT")
    sm = structure_mismatches(struct)
    n_cmp = (
        struct.pivot(index="article_id", columns="lang", values="n_paragraphs").dropna().shape[0]
    )
    add(
        f"    mismatching articles: {len(sm)} of {n_cmp} aligned ids "
        "(format: <paragraphs>p/<items>i)"
    )
    for aid in sorted(sm.index, key=natural_key)[:MAX_LISTED]:
        add(f"    {aid}: " + "  ".join(f"{lang}={sm.loc[aid, lang]}" for lang in LANGS))
    if len(sm) > MAX_LISTED:
        add(f"    ... {len(sm) - MAX_LISTED} more")
    add("")
    add(
        f"5c. text-length ratio (longest/shortest of DE/FR/IT) > {LENGTH_RATIO} "
        f"(longest text > {LENGTH_MIN_CHARS} chars), in-force chunks"
    )
    lr = length_ratios(df)
    flagged = lr[(lr["ratio"] > LENGTH_RATIO) & (lr["longest"] > LENGTH_MIN_CHARS)]
    add(f"    flagged: {len(flagged)} of {len(lr)} aligned in-force ids")
    for aid in flagged.sort_values("ratio", ascending=False).head(MAX_LISTED).index:
        row = flagged.loc[aid]
        add(
            f"    {aid}: ratio {row['ratio']:.2f}  "
            + "  ".join(f"{lang}={int(row[lang])}" for lang in LANGS)
        )
    if len(flagged) > MAX_LISTED:
        add(f"    ... {len(flagged) - MAX_LISTED} more (worst {MAX_LISTED} shown)")
    add("")

    add("6. BLOCKS AND ANNEXES")
    add(rule)
    add("6a. transitional/final-provision blocks without <article>: chunked whole as '<SR>:u<N>'")
    n_units = 0
    for sr in acts:
        for lang in LANGS:
            for eid, heading, n_chars in unarticled_units(file_path(manifest[sr], lang, raw_dir)):
                n_units += 1
                state = f"chunked ({n_chars} chars)" if n_chars else "NOT chunked (no text)"
                add(f"    SR {sr} {lang}: {eid} | {heading[:60]} | {state}")
    if n_units == 0:
        add("    None.")
    add("6b. annexes (<components>) are NOT chunked; text left out:")
    n_comp = 0
    for sr in acts:
        for lang in LANGS:
            comps = unchunked_components(file_path(manifest[sr], lang, raw_dir))
            if comps:
                n_comp += 1
                add(f"    SR {sr} {lang}: " + ", ".join(f"{name}: {n} chars" for name, n in comps))
    if n_comp == 0:
        add("    None.")
    add("")

    add("7. ARTICLES WITHOUT MARGINAL TITLE (per language; informational)")
    add(rule)
    for sr in sorted(df["sr"].unique()):
        cells = []
        for lang in LANGS:
            g = in_force(df[(df["sr"] == sr) & (df["lang"] == lang)])
            cells.append(f"{lang}={int((g['title'] == '').sum())}")
        add(f"SR {sr}: " + ", ".join(cells) + " (in-force chunks with empty title)")
    add("")

    add("8. FOOTNOTE COVERAGE (chunks with a non-empty `footnotes` column / chunks)")
    add(rule)
    add("Footnotes are kept apart from `text` (Fedlex reuse terms forbid altering content).")
    L.extend(footnote_coverage_lines(df))
    add("")

    add(f"9. FIVE RANDOM ALIGNED IN-FORCE ARTICLES (seed={SEED})")
    add(rule)
    for aid in sample_aligned(df):
        add(side_by_side(df, aid))
        add("")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parquet", type=Path, default=ARTICLES_PARQUET)
    ap.add_argument("--out", type=Path, default=REPORT_PATH)
    ap.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = pd.read_parquet(args.parquet)
    try:
        report = build_report(df, load_manifest(args.manifest), parquet_path=args.parquet)
    except InputMismatchError as exc:
        log.error("inputs do not match: %s", exc)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8", newline="\n")
    log.info("wrote %s", args.out)
    _, _, rate = alignment_rate(df)
    bad = [sr for sr in sorted(df["sr"].unique()) if alignment_rate(df, sr)[2] < MIN_RATE]
    if rate < MIN_RATE or bad:
        log.error(
            "id alignment below %.0f%%: overall %.2f%%, acts %s", MIN_RATE * 100, rate * 100, bad
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
