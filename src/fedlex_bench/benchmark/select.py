"""``benchmark select``: candidates + critic verdicts -> dev, test and reserve files.

For every cell (split x source act; traps: split) the critic-kept candidates are taken in slot
order: the first ``quota`` go to the selected files, the surplus to the reserve (backfill if a
human rejects an item of that cell). Dropped candidates, candidates that need a fix and candidates
without a verdict are in neither. Final ids ``v1-0001`` ... are assigned by code, in the order
(split, source act, slot id), counting on across dev and test.

Verdict files are JSONL ``{"slot_id", "verdict": keep|fix|drop, "notes"}`` (more keys, such as the
critic's ``reason_code``, are allowed and ignored). A slot may have several records when each
carries a distinct integer ``round``; the highest round decides and ``critic_rounds`` counts them.
``review_status`` is never touched: items leave this step as ``unreviewed``.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from fedlex_bench.benchmark.common import (
    CANDIDATES_PATH,
    CRITIC_DIR,
    RESERVE_DIR,
    SELECTED_DIR,
    SPLITS,
    TRAP_FINAL,
    act_key,
    read_jsonl,
    write_jsonl,
)
from fedlex_bench.benchmark.plan import final_quotas

VERDICTS = ("keep", "fix", "drop")
MAX_DROP_SHARE = 0.40
ANSWERABLE_TARGET = 150

Cell = tuple[str, "str | None"]


@dataclass
class SelectResult:
    dev: list[dict] = field(default_factory=list)
    test: list[dict] = field(default_factory=list)
    reserve: list[dict] = field(default_factory=list)
    cells: dict[Cell, dict] = field(default_factory=dict)
    shortfalls: list[dict] = field(default_factory=list)
    verdict_counts: Counter = field(default_factory=Counter)


def cell_quota(split: str, act: str | None) -> int:
    """Final items per cell: the act table split into dev/test, or 5 / 15 traps."""
    if act is None:
        return TRAP_FINAL[split]
    return final_quotas().get((act, split), 0)


# ------------------------------------------------------------------ verdict files


def load_verdicts(
    directory: Path, known_slots: set[str]
) -> tuple[dict[str, list[dict]], list[str]]:
    """``({slot_id: [records]}, problems)``. Problems are ``file:line: ...`` strings."""
    if not directory.is_dir():
        raise FileNotFoundError(f"critic directory {directory} does not exist")
    records: dict[str, list[dict]] = {}
    problems: list[str] = []
    for path in sorted(directory.glob("*.jsonl")):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            where = f"{path.name}:{n}"
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                problems.append(f"{where}: invalid JSON ({exc.msg})")
                continue
            slot_id = rec.get("slot_id") if isinstance(rec, dict) else None
            if not isinstance(slot_id, str):
                problems.append(f"{where}: missing field 'slot_id'")
                continue
            if slot_id not in known_slots:
                problems.append(f"{where}: unknown slot {slot_id}")
                continue
            if rec.get("verdict") not in VERDICTS:
                problems.append(f"{where}: {slot_id}: verdict must be one of {list(VERDICTS)}")
                continue
            if not isinstance(rec.get("notes", ""), str):
                problems.append(f"{where}: {slot_id}: notes must be a string")
                continue
            rnd = rec.get("round")
            if "round" in rec and (isinstance(rnd, bool) or not isinstance(rnd, int) or rnd < 1):
                problems.append(f"{where}: {slot_id}: round must be an integer >= 1")
                continue
            records.setdefault(slot_id, []).append({**rec, "_where": where})
    for slot_id, recs in sorted(records.items()):
        if len(recs) == 1:
            continue
        rounds = [r.get("round") for r in recs]
        if None in rounds or len(set(rounds)) != len(rounds):
            problems.append(
                f"{recs[1]['_where']}: duplicate verdict for {slot_id} (several records need "
                "distinct integer 'round' values)"
            )
            records[slot_id] = []
    return {k: v for k, v in records.items() if v}, problems


def _latest(records: list[dict]) -> dict:
    return max(records, key=lambda r: r.get("round", 0))


# ------------------------------------------------------------------ selection


def _finish(candidate: dict, records: list[dict], item_id: str | None) -> dict:
    item = copy.deepcopy(candidate)
    item.pop("slot_id", None)
    latest = _latest(records)
    item["id"] = item_id
    item["provenance"]["critic_verdict"] = latest["verdict"]
    item["provenance"]["critic_rounds"] = len(records)
    item["provenance"]["critic_notes"] = latest.get("notes", "")
    return item


def _order_key(item: dict) -> tuple:
    return (item["split"], act_key(item["source_act"]), item["provenance"]["slot_id"])


def select(candidates: list[dict], verdicts: dict[str, list[dict]]) -> SelectResult:
    result = SelectResult()
    by_cell: dict[Cell, list[dict]] = {}
    for c in sorted(candidates, key=lambda c: c["slot_id"]):
        by_cell.setdefault((c["split"], c["source_act"]), []).append(c)

    chosen: list[dict] = []
    surplus: list[dict] = []
    for cell in sorted(by_cell, key=lambda k: (SPLITS.index(k[0]), act_key(k[1]))):
        members = by_cell[cell]
        counts: Counter = Counter()
        kept: list[tuple[dict, list[dict]]] = []
        for c in members:
            records = verdicts.get(c["slot_id"])
            if not records:
                counts["no_verdict"] += 1
                continue
            verdict = _latest(records)["verdict"]
            counts[verdict] += 1
            result.verdict_counts[verdict] += 1
            if verdict == "keep":
                kept.append((c, records))
        quota = cell_quota(*cell)
        for c, records in kept[:quota]:
            chosen.append(_finish(c, records, "pending"))
        for c, records in kept[quota:]:
            surplus.append(_finish(c, records, None))
        n_selected = min(quota, len(kept))
        result.cells[cell] = {
            "quota": quota,
            "candidates": len(members),
            "keep": counts["keep"],
            "fix": counts["fix"],
            "drop": counts["drop"],
            "no_verdict": counts["no_verdict"],
            "selected": n_selected,
            "reserve": max(0, len(kept) - quota),
        }
        if n_selected < quota:
            result.shortfalls.append(
                {"cell": cell, "quota": quota, "selected": n_selected, "short": quota - n_selected}
            )

    chosen.sort(key=_order_key)
    for n, item in enumerate(chosen, 1):
        item["id"] = f"v1-{n:04d}"
    result.dev = [i for i in chosen if i["split"] == "dev"]
    result.test = [i for i in chosen if i["split"] == "test"]
    result.reserve = sorted(surplus, key=_order_key)
    return result


# ------------------------------------------------------------------ command


def _table(result: SelectResult) -> list[str]:
    lines = [
        f"{'split':<6}{'act':<11}{'quota':>6}{'cand':>6}{'keep':>6}{'fix':>5}{'drop':>6}"
        f"{'none':>6}{'sel':>5}{'res':>5}"
    ]
    for (split, act), c in result.cells.items():
        lines.append(
            f"{split:<6}{act or '(trap)':<11}{c['quota']:>6}{c['candidates']:>6}{c['keep']:>6}"
            f"{c['fix']:>5}{c['drop']:>6}{c['no_verdict']:>6}{c['selected']:>5}{c['reserve']:>5}"
        )
    return lines


def run(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="benchmark select", description=__doc__.splitlines()[0])
    ap.add_argument("--candidates", type=Path, default=CANDIDATES_PATH)
    ap.add_argument("--critic", type=Path, default=CRITIC_DIR, help="directory of verdict *.jsonl")
    ap.add_argument("--out-dir", type=Path, default=SELECTED_DIR)
    ap.add_argument("--reserve-dir", type=Path, default=RESERVE_DIR)
    args = ap.parse_args(argv)

    candidates = read_jsonl(args.candidates)
    verdicts, problems = load_verdicts(args.critic, {c["slot_id"] for c in candidates})
    if problems:
        print(f"{len(problems)} problem(s) in the critic files:")
        for p in problems:
            print(f"  {p}")
        print("nothing written")
        return 1
    result = select(candidates, verdicts)

    write_jsonl(args.out_dir / "dev.jsonl", result.dev)
    write_jsonl(args.out_dir / "test.jsonl", result.test)
    write_jsonl(args.reserve_dir / "dev.jsonl", [i for i in result.reserve if i["split"] == "dev"])
    write_jsonl(
        args.reserve_dir / "test.jsonl", [i for i in result.reserve if i["split"] == "test"]
    )

    print("\n".join(_table(result)))
    n_answerable = sum(1 for i in result.dev + result.test if not i["is_trap"])
    print(
        f"selected {len(result.dev) + len(result.test)} (dev {len(result.dev)}, test "
        f"{len(result.test)}; answerable {n_answerable}), reserve {len(result.reserve)}"
    )
    for s in result.shortfalls:
        split, act = s["cell"]
        label = f"{split}/{act or 'trap'}"
        print(f"SHORT {label}: {s['selected']} of {s['quota']} (short by {s['short']})")
    total_verdicts = sum(result.verdict_counts.values())
    if total_verdicts and result.verdict_counts["drop"] / total_verdicts > MAX_DROP_SHARE:
        share = result.verdict_counts["drop"] / total_verdicts
        print(
            f"STOP: the critic dropped {share:.0%} of the drafts (limit 40%): ask before going on"
        )
    if n_answerable < ANSWERABLE_TARGET:
        print(f"STOP: only {n_answerable} of {ANSWERABLE_TARGET} answerable items survive")
    return 0
