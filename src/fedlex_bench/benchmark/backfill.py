"""``benchmark backfill``: replace selected items by reserve items of the same cell.

For each ``--ids`` item (say an expert flagged it as wrong) the item leaves ``--dir`` and the best
reserve item of the same *cell* (same split, source act and question type) takes its id, in the
same place in the file. The removed item is appended to ``<removed>/<split>.jsonl`` with
``removal_reason`` (``expert-flag`` unless ``--reason`` says otherwise).

Best reserve item, in this order: same query language as the removed item, then same source
language, then the order of the reserve file (its ``slot_id``, then file position). Reserve items
are skipped when
- a gold article is already used by a selected item (translations share their original's gold, so
  they are covered), or was the gold of an item removed in this run, or by a replacement chosen
  earlier in this run;
- a gold article is a twin article (``benchmark/plan/twins.jsonl``): twins are not allowed as gold;
- the item is expert-flagged or human-rejected.

The new item is ``unreviewed`` (reviewer fields and ``provenance.review_basis`` cleared) and carries
``provenance.backfill = {"replaces": <old id>}``. The used reserve item is removed from the
reserve file. If any id has no replacement, or has translations, nothing is written. An id whose
selected item is itself a backfill is skipped (so running twice changes nothing); ``--again``
backfills it anyway. The files are replaced atomically; if one changed on disk since it was read
nothing is written.
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

from fedlex_bench.benchmark.common import (
    BENCHMARK_DIR,
    RESERVE_DIR,
    SELECTED_DIR,
    SPLITS,
    TWINS_PATH,
    read_jsonl,
    read_jsonl_text,
)
from fedlex_bench.benchmark.translations import (
    TranslationError,
    read_split_files,
    replace_files,
    to_bytes,
)

REMOVED_DIR = BENCHMARK_DIR / "removed"
DEFAULT_REASON = "expert-flag"


class BackfillError(TranslationError):
    """Nothing was written."""


@dataclass(frozen=True)
class Replacement:
    old_id: str
    split: str
    old_question: str
    new_question: str
    new_slot_id: str | None
    tier: str  # "same query language and source language", ...


def _cell(item: Mapping) -> tuple:
    return (item["split"], item["source_act"], item["qtype"])


def _tier(old: Mapping, new: Mapping) -> int:
    same_q = new["query_lang"] == old["query_lang"]
    same_s = new["source_lang"] == old["source_lang"]
    return 0 if same_q and same_s else 1 if same_q else 2 if same_s else 3


TIER_TEXT = (
    "same query and source language",
    "same query language",
    "same source language",
    "same cell only",
)


def _reserve_key(old: Mapping, new: Mapping, position: int) -> tuple:
    prov = new.get("provenance") or {}
    return (
        new["query_lang"] != old["query_lang"],
        new["source_lang"] != old["source_lang"],
        str(prov.get("slot_id") or ""),
        position,
    )


def backfill_items(
    ids: list[str],
    selected: dict[str, list[dict]],
    reserve: dict[str, list[dict]],
    twin_articles: Collection[str] = (),
    reason: str = DEFAULT_REASON,
    again: bool = False,
) -> tuple[dict[str, list[dict]], dict[str, list[dict]], dict[str, list[dict]], list[Replacement]]:
    """Pure core. Returns ``(selected, reserve, removed, replacements)``: new lists per split
    (the inputs are not changed). ``removed`` holds the items taken out, per split."""
    sel = {s: copy.deepcopy(selected[s]) for s in SPLITS}
    res = {s: copy.deepcopy(reserve[s]) for s in SPLITS}
    removed: dict[str, list[dict]] = {s: [] for s in SPLITS}
    where = {it["id"]: (s, n) for s in SPLITS for n, it in enumerate(sel[s])}
    twins = set(twin_articles)

    wanted = sorted(dict.fromkeys(ids))
    todo: list[str] = []
    for item_id in wanted:
        if item_id not in where:
            raise BackfillError(f"{item_id}: not a selected item")
        split, n = where[item_id]
        item = sel[split][n]
        if item.get("translation_of"):
            raise BackfillError(f"{item_id}: a translation cannot be backfilled; remove it by hand")
        children = sorted(
            it["id"] for s in SPLITS for it in sel[s] if it.get("translation_of") == item_id
        )
        if children:
            raise BackfillError(f"{item_id}: has translations {children}; resolve them first")
        backfill = (item.get("provenance") or {}).get("backfill")
        if backfill and not again:
            print(f"{item_id}: already a backfill (replaces {backfill.get('replaces')}), skipped")
            continue
        todo.append(item_id)

    used_gold: set[str] = {g for s in SPLITS for it in sel[s] for g in it["gold_article_ids"]}
    for item_id in todo:  # the flagged item's gold stays blocked: the article may be the problem
        split, n = where[item_id]
        used_gold.update(sel[split][n]["gold_article_ids"])

    replacements: list[Replacement] = []
    problems: list[str] = []
    for item_id in todo:
        split, n = where[item_id]
        old = sel[split][n]
        options = [
            (pos, r)
            for pos, r in enumerate(res[split])
            if _cell(r) == _cell(old)
            and not set(r["gold_article_ids"]) & used_gold
            and not set(r["gold_article_ids"]) & twins
            and not (r.get("provenance") or {}).get("expert_flag")
            and r.get("review_status") != "rejected"
        ]
        if not options:
            problems.append(
                f"{item_id}: no usable reserve item in cell split={old['split']} "
                f"act={old['source_act']} qtype={old['qtype']}"
            )
            continue
        pos, best = min(options, key=lambda o: _reserve_key(old, o[1], o[0]))
        new = copy.deepcopy(best)
        new.update(
            id=item_id,
            review_status="unreviewed",
            reviewed_by=None,
            reviewed_at=None,
            review_note=None,
        )
        prov = {**new.get("provenance", {}), "backfill": {"replaces": item_id}}
        prov.pop("review_basis", None)  # describes a review of other text
        new["provenance"] = prov
        used_gold.update(new["gold_article_ids"])
        removed[split].append({**old, "removal_reason": reason})
        sel[split][n] = new
        del res[split][pos]
        replacements.append(
            Replacement(
                item_id,
                split,
                old["question"],
                new["question"],
                (best.get("provenance") or {}).get("slot_id"),
                TIER_TEXT[_tier(old, best)],
            )
        )
    if problems:
        raise BackfillError("cannot backfill, nothing written:\n  " + "\n  ".join(problems))
    return sel, res, removed, replacements


def run(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="benchmark backfill", description=__doc__.splitlines()[0])
    ap.add_argument("--ids", required=True, help="comma-separated selected ids, e.g. v1-0012")
    ap.add_argument("--dir", type=Path, default=SELECTED_DIR, help="folder with dev/test.jsonl")
    ap.add_argument("--reserve", type=Path, default=RESERVE_DIR, help="folder with the reserve")
    ap.add_argument("--removed", type=Path, default=REMOVED_DIR, help="folder for removed items")
    ap.add_argument("--twins", type=Path, default=TWINS_PATH, help="twins.jsonl (may be missing)")
    ap.add_argument("--reason", default=DEFAULT_REASON, help="why the items are removed")
    ap.add_argument("--again", action="store_true", help="also backfill an item that is a backfill")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args(argv)
    try:
        ids = [i.strip() for i in args.ids.split(",") if i.strip()]
        if not ids:
            raise BackfillError("--ids is empty")
        sel_raw, sel = read_split_files(args.dir)
        res_raw, res = read_split_files(args.reserve)
        removed_raw = {
            s: (args.removed / f"{s}.jsonl").read_bytes()
            if (args.removed / f"{s}.jsonl").exists()
            else None
            for s in SPLITS
        }
        old_removed = {
            s: read_jsonl_text(removed_raw[s].decode("utf-8"), f"removed/{s}.jsonl")
            if removed_raw[s] is not None
            else []
            for s in SPLITS
        }
        twins = {r["article_id"] for r in read_jsonl(args.twins)} if args.twins.exists() else set()
        new_sel, new_res, removed, replacements = backfill_items(
            ids, sel, res, twins, args.reason, args.again
        )
        if not args.dry_run and replacements:
            changes = {}
            for s in SPLITS:
                if any(r.split == s for r in replacements):
                    changes[args.dir / f"{s}.jsonl"] = (sel_raw[s], to_bytes(new_sel[s]))
                    changes[args.reserve / f"{s}.jsonl"] = (res_raw[s], to_bytes(new_res[s]))
                    changes[args.removed / f"{s}.jsonl"] = (
                        removed_raw[s],
                        to_bytes(old_removed[s] + removed[s]),
                    )
            replace_files(changes)
    except (TranslationError, ValueError, OSError) as exc:
        print(f"ERROR   {exc}")
        return 1
    for r in replacements:
        print(f"{r.old_id}\t{r.split}\treplaced by reserve slot {r.new_slot_id} ({r.tier})")
        print(f"    old: {r.old_question}")
        print(f"    new: {r.new_question}")
    verb = "would replace" if args.dry_run else "replaced"
    print(f"{verb} {len(replacements)} of {len(set(ids))} item(s); new items are unreviewed")
    return 0
