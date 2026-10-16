"""``benchmark apply-revisions``: swap revised candidates into the selected items.

``--candidates`` is an assembled candidates JSONL (one record per slot), ``--slots-file`` a JSON
list of slot ids to revise. For each slot the selected item with ``provenance.slot_id`` equal to it
gets the candidate's question, gold facts, difficulty, ranks, trap reason and candidate hash. The
candidate must agree with the item on gold article ids, query language, question type, split and
source act, otherwise nothing is written. The previous question and facts are kept once under
``provenance.pre_revision``; ``provenance.revision`` is set; the item goes back to ``unreviewed``
(reviewer fields cleared) because a human must re-review revised text. ``expert_flag`` is kept.

A slot whose item already carries the candidate's hash and a revision mark is skipped, so running
twice never wipes a human review of the revised text. Files are replaced atomically (sorted keys,
LF, same order); if a file changed on disk while the command ran (the review app writes there) the
command aborts instead of overwriting the change.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from fedlex_bench.benchmark.common import SELECTED_DIR, SPLITS, dumps, read_jsonl_text

REVISION = "machine-feedback-1"
MATCH_FIELDS = ("gold_article_ids", "query_lang", "qtype", "split", "source_act")
REPLACED = ("question", "gold_facts", "difficulty", "bm25_rank", "lexical_overlap", "trap_reason")


class RevisionError(Exception):
    """Candidates and selection do not fit; nothing was written."""


@dataclass(frozen=True)
class Change:
    item_id: str
    slot_id: str
    old_status: str


def _revise(item: dict, cand: dict) -> None:
    prov = item["provenance"]
    prov.setdefault(
        "pre_revision", {"question": item["question"], "gold_facts": item["gold_facts"]}
    )
    for key in REPLACED:
        item[key] = cand.get(key)
    item["gold_facts"] = [
        {"text": f["text"], "article_id": f["article_id"]} for f in cand["gold_facts"]
    ]
    prov["candidate_sha256"] = cand["candidate_sha256"]
    prov["revision"] = REVISION
    item["review_status"] = "unreviewed"
    item["reviewed_by"] = item["reviewed_at"] = item["review_note"] = None


def _replace_atomic(path: Path, data: bytes, expected: bytes) -> None:
    """Atomically replace ``path`` with ``data``, unless it no longer holds ``expected``."""
    if path.read_bytes() != expected:
        raise RevisionError(f"{path.name} changed on disk while revising; re-run")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def apply_revisions(
    candidates: list[dict], slot_ids: list[str], directory: Path, dry_run: bool = False
) -> list[Change]:
    """Revise the items of ``slot_ids`` in ``directory``; returns the changed items."""
    by_slot: dict[str, dict] = {}
    for cand in candidates:
        slot = cand.get("slot_id")
        if slot in by_slot:
            raise RevisionError(f"duplicate candidate for slot {slot}")
        by_slot[slot] = cand
    raw: dict[str, bytes] = {}
    rows: dict[str, list[dict]] = {}
    for split in SPLITS:
        path = directory / f"{split}.jsonl"
        raw[split] = path.read_bytes()
        rows[split] = read_jsonl_text(raw[split].decode("utf-8"), str(path))
    where = {
        it["provenance"]["slot_id"]: (split, it)
        for split in SPLITS
        for it in rows[split]
        if isinstance(it.get("provenance"), dict) and it["provenance"].get("slot_id")
    }
    changes: list[Change] = []
    touched: set[str] = set()
    for slot in sorted(dict.fromkeys(slot_ids)):
        if slot not in by_slot:
            raise RevisionError(f"slot {slot}: no candidate in the candidates file")
        if slot not in where:
            raise RevisionError(f"slot {slot}: no selected item with this slot id")
        split, item = where[slot]
        cand = by_slot[slot]
        for field in MATCH_FIELDS:
            if cand.get(field) != item.get(field):
                raise RevisionError(
                    f"slot {slot} ({item['id']}): {field} differs: item {item.get(field)!r}, "
                    f"candidate {cand.get(field)!r}"
                )
        prov = item["provenance"]
        if prov.get("revision") == REVISION and prov.get("candidate_sha256") == cand.get(
            "candidate_sha256"
        ):
            print(f"{item['id']}: already revised, skipped")
            continue
        changes.append(Change(item["id"], slot, item["review_status"]))
        _revise(item, cand)
        touched.add(split)
    if not dry_run:
        for split in sorted(touched):
            data = "".join(dumps(r) + "\n" for r in rows[split]).encode("utf-8")
            _replace_atomic(directory / f"{split}.jsonl", data, raw[split])
    return changes


def run(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="benchmark apply-revisions", description=__doc__.splitlines()[0]
    )
    ap.add_argument("--candidates", type=Path, required=True, help="assembled candidates JSONL")
    ap.add_argument("--slots-file", type=Path, required=True, help="JSON list of slot ids")
    ap.add_argument("--dir", type=Path, default=SELECTED_DIR, help="folder with dev/test.jsonl")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args(argv)
    try:
        candidates = read_jsonl_text(
            args.candidates.read_bytes().decode("utf-8"), str(args.candidates)
        )
        slots = json.loads(args.slots_file.read_text(encoding="utf-8"))
        if not isinstance(slots, list) or not all(isinstance(s, str) for s in slots):
            raise RevisionError(f"{args.slots_file}: expected a JSON list of slot id strings")
        changes = apply_revisions(candidates, slots, args.dir, dry_run=args.dry_run)
    except (RevisionError, ValueError, OSError) as exc:
        print(f"ERROR   {exc}")
        return 1
    for c in changes:
        print(f"{c.item_id}\t{c.slot_id}\told status: {c.old_status}")
    verb = "would revise" if args.dry_run else "revised"
    print(f"{verb} {len(changes)} of {len(set(slots))} item(s); all reset to unreviewed")
    return 0
