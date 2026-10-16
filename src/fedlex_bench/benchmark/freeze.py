"""Freeze the reviewed selection into ``benchmark/v1`` (docs/benchmark_spec.md, section 12).

Refuses unless ``validate --freeze`` passes on the selected folder, and refuses while a rejected
item is still in it (a rejected item must be replaced from the reserve first; the message names
the short cells). Writes ``dev.jsonl`` and ``test.jsonl`` (items sorted by id),
``test_gold_articles.txt`` (sorted unique gold ids of the test split) and ``SHA256SUMS``.
Refuses to overwrite an existing ``benchmark/v1`` without ``--force``. Tagging the freeze commit
(``benchmark-v1``) is a manual step.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from fedlex_bench.benchmark.common import (
    SELECTED_DIR,
    SPLITS,
    V1_DIR,
    dumps,
    id_key,
    load_corpus,
)
from fedlex_bench.benchmark.select import cell_quota
from fedlex_bench.benchmark.validate import FROZEN_STATUSES, Report, load_dir, validate_dir
from fedlex_bench.chunk import ARTICLES_PARQUET

GOLD_FILE = "test_gold_articles.txt"
SUMS_FILE = "SHA256SUMS"
FILES = ("dev.jsonl", "test.jsonl", GOLD_FILE)


def cell_label(item: dict) -> str:
    return f"{item['split']}/{item['source_act'] or 'trap'}"


def short_cells(items: list[dict]) -> dict[str, dict]:
    """Cells that lost an item to a rejection: ``{"dev/220": {"rejected": [ids], "kept": n,
    "quota": q}}`` where ``kept`` counts the cell's items that are not rejected."""
    cells: dict[str, dict] = {}
    for it in items:
        cell = cells.setdefault(cell_label(it), {"rejected": [], "kept": 0})
        if it["review_status"] == "rejected":
            cell["rejected"].append(it["id"])
        else:
            cell["kept"] += 1
        cell["quota"] = cell_quota(it["split"], it["source_act"])
    return {k: v for k, v in sorted(cells.items()) if v["rejected"]}


def render(files: dict[str, list[dict]]) -> dict[str, bytes]:
    """The bytes of the four output files, built from the (validated) selection."""
    out: dict[str, bytes] = {}
    for split in SPLITS:
        kept = sorted(
            (i for i in files[split] if i["review_status"] in FROZEN_STATUSES),
            key=lambda i: i["id"],
        )
        out[f"{split}.jsonl"] = "".join(dumps(i) + "\n" for i in kept).encode("utf-8")
    gold = {g for it in files["test"] for g in it["gold_article_ids"]}
    out[GOLD_FILE] = "".join(g + "\n" for g in sorted(gold, key=id_key)).encode("utf-8")
    sums = "".join(f"{hashlib.sha256(out[n]).hexdigest()}  {n}\n" for n in sorted(FILES))
    out[SUMS_FILE] = sums.encode("utf-8")
    return out


def freeze(directory: Path, out_dir: Path, df: pd.DataFrame, force: bool = False) -> int:
    pre = Report()
    files = load_dir(directory, pre)
    if pre.errors:
        for v in pre.errors:
            print(f"ERROR   {v}")
        print("freeze refused: the selected folder cannot be read")
        return 1
    rejected = short_cells([i for split in SPLITS for i in files[split]])
    if rejected:
        for label, cell in rejected.items():
            print(
                f"SHORT {label}: {cell['kept']} of {cell['quota']} after rejecting "
                f"{', '.join(cell['rejected'])}"
            )
        print(
            "freeze refused: rejected items must be replaced by reserve backfill in the "
            "selected folder first (not automated yet); the cells above are short"
        )
        return 1
    report = validate_dir(directory, df, freeze=True, check_ranks=True)
    for v in report.warnings:
        print(f"WARNING {v}")
    if report.errors:
        for v in report.errors:
            print(f"ERROR   {v}")
        print(f"freeze refused: validate --freeze reports {len(report.errors)} error(s)")
        return 1
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        print(f"freeze refused: {out_dir} already exists (use --force to overwrite)")
        return 1
    content = render(files)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in sorted(content):
        (out_dir / name).write_bytes(content[name])
    n = {s: len(files[s]) for s in SPLITS}
    print(f"froze dev {n['dev']} and test {n['test']} items into {out_dir}")
    print(content[SUMS_FILE].decode("utf-8"), end="")
    print("next: commit benchmark/v1 and tag it benchmark-v1")
    return 0


def run(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="benchmark freeze", description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, default=SELECTED_DIR, help="reviewed selection")
    ap.add_argument("--out", type=Path, default=V1_DIR, help="folder to write (benchmark/v1)")
    ap.add_argument("--parquet", type=Path, default=ARTICLES_PARQUET)
    ap.add_argument("--force", action="store_true", help="overwrite an existing output folder")
    args = ap.parse_args(argv)
    return freeze(args.dir, args.out, load_corpus(args.parquet), force=args.force)
