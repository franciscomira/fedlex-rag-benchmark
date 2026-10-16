"""``benchmark select``: candidates + critic verdicts -> dev, test and reserve files.

For every cell (split x source act; traps: split) the plan fixes a *final prefix*: the first
``quota`` slots in slot order. A prefix slot whose candidate the critic kept is selected as it is.
A prefix slot that was dropped, needs a fix, was skipped by the writer or has no verdict is
**backfilled** from the kept slots of the same cell outside the prefix (the surplus), the best
match first:

1. same query language, question type and source language;
2. same query language and question type;
3. same source language;
4. any kept surplus slot, in slot order.

Every replacement that is not a tier-1 match is printed, because it shifts the language or source
mix away from the plan. Kept surplus slots that are not needed go to the reserve (backfill if a
human rejects an item of that cell). Ids ``v1-0001`` ... are assigned by code, in the order
(split, source act, slot id), counting on across dev and test.

Verdict files are JSONL ``{"slot_id", "verdict": keep|fix|drop, "notes", "candidate_sha256"}``
(more keys, such as the critic's ``reason_code``, are allowed and ignored). ``candidate_sha256`` is
the hash ``assemble`` wrote into the candidate: a verdict about other text is rejected. A slot may
have several records when each carries a distinct integer ``round``; the highest round decides
(and must quote the current hash), and ``critic_rounds`` counts them.

If the critic dropped more than 40% of the drafts or fewer than 150 answerable items survive,
``select`` prints STOP, exits non-zero and writes nothing unless ``--force`` is given.
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
    SLOTS_PATH,
    SPLITS,
    TRAP_FINAL,
    act_key,
    numbered_lines,
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
    backfills: list[dict] = field(default_factory=list)
    verdict_counts: Counter = field(default_factory=Counter)


def cell_quota(split: str, act: str | None) -> int:
    """Final items per cell: the act table split into dev/test, or 5 / 15 traps."""
    if act is None:
        return TRAP_FINAL[split]
    return final_quotas().get((act, split), 0)


# ------------------------------------------------------------------ verdict files


def load_verdicts(
    directory: Path, candidate_hashes: dict[str, str]
) -> tuple[dict[str, list[dict]], list[str]]:
    """``({slot_id: [records]}, problems)``; ``candidate_hashes`` maps each candidate's slot id to
    its ``candidate_sha256``. Problems are ``file:line: ...`` strings."""
    if not directory.is_dir():
        raise FileNotFoundError(f"critic directory {directory} does not exist")
    records: dict[str, list[dict]] = {}
    problems: list[str] = []
    for path in sorted(directory.glob("*.jsonl")):
        for n, line in numbered_lines(path.read_bytes().decode("utf-8")):
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
            if slot_id not in candidate_hashes:
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
            if not isinstance(rec.get("candidate_sha256"), str):
                problems.append(
                    f"{where}: {slot_id}: missing field 'candidate_sha256' (a verdict must quote "
                    "the hash of the text it judged)"
                )
                continue
            records.setdefault(slot_id, []).append({**rec, "_where": where})
    for slot_id, recs in sorted(records.items()):
        if len(recs) > 1:
            rounds = [r.get("round") for r in recs]
            if None in rounds or len(set(rounds)) != len(rounds):
                problems.append(
                    f"{recs[1]['_where']}: duplicate verdict for {slot_id} (several records need "
                    "distinct integer 'round' values)"
                )
                records[slot_id] = []
                continue
        latest = _latest(recs)
        if latest["candidate_sha256"] != candidate_hashes[slot_id]:
            problems.append(
                f"{latest['_where']}: {slot_id}: the verdict was written for different text "
                f"(candidate_sha256 {latest['candidate_sha256'][:12]}..., the candidate has "
                f"{candidate_hashes[slot_id][:12]}...); the question or facts changed after the "
                "critic saw them, ask for a new verdict"
            )
            records[slot_id] = []
    return {k: v for k, v in records.items() if v}, problems


def _latest(records: list[dict]) -> dict:
    return max(records, key=lambda r: r.get("round", 0))


# ------------------------------------------------------------------ selection


def _finish(candidate: dict, records: list[dict], item_id: str | None) -> dict:
    item = copy.deepcopy(candidate)
    item.pop("slot_id", None)
    sha = item.pop("candidate_sha256", None)
    latest = _latest(records)
    item["id"] = item_id
    item["provenance"]["critic_verdict"] = latest["verdict"]
    item["provenance"]["critic_rounds"] = len(records)
    item["provenance"]["critic_notes"] = latest.get("notes", "")
    item["provenance"]["candidate_sha256"] = sha
    return item


def _order_key(item: dict) -> tuple:
    return (item["split"], act_key(item["source_act"]), item["provenance"]["slot_id"])


def _tier(missing: dict, option: dict) -> int:
    """How well a kept surplus slot replaces a missing prefix slot (1 = best)."""
    same_lang_type = (
        option["query_lang"] == missing["query_lang"] and option["qtype"] == missing["qtype"]
    )
    same_source = option["source_lang"] == missing["source_lang"]
    if same_lang_type and same_source:
        return 1
    if same_lang_type:
        return 2
    if same_source:
        return 3
    return 4


def select(
    candidates: list[dict], verdicts: dict[str, list[dict]], slots: list[dict]
) -> SelectResult:
    result = SelectResult()
    by_slot = {c["slot_id"]: c for c in candidates}
    plan_cells: dict[Cell, list[dict]] = {}
    for s in sorted(slots, key=lambda s: s["slot_id"]):
        plan_cells.setdefault((s["split"], s["act"]), []).append(s)

    chosen: list[dict] = []
    surplus: list[dict] = []
    for cell in sorted(plan_cells, key=lambda k: (SPLITS.index(k[0]), act_key(k[1]))):
        members = plan_cells[cell]
        quota = cell_quota(*cell)
        counts: Counter = Counter()
        kept: set[str] = set()
        for s in members:
            records = verdicts.get(s["slot_id"])
            if s["slot_id"] not in by_slot:
                counts["no_candidate"] += 1  # skipped by the writer or never drafted
            elif not records:
                counts["no_verdict"] += 1
            else:
                verdict = _latest(records)["verdict"]
                counts[verdict] += 1
                result.verdict_counts[verdict] += 1
                if verdict == "keep":
                    kept.add(s["slot_id"])
        prefix, rest = members[:quota], members[quota:]
        selected = [s for s in prefix if s["slot_id"] in kept]
        missing = [s for s in prefix if s["slot_id"] not in kept]
        spare = [s for s in rest if s["slot_id"] in kept]
        n_backfilled = 0
        for m in missing:
            if not spare:
                break
            best = min(spare, key=lambda o: (_tier(m, o), o["slot_id"]))
            tier = _tier(m, best)
            spare.remove(best)
            selected.append(best)
            n_backfilled += 1
            result.backfills.append(
                {
                    "cell": cell,
                    "dropped": m["slot_id"],
                    "replacement": best["slot_id"],
                    "tier": tier,
                    "dropped_attrs": (m["query_lang"], m["qtype"], m["source_lang"]),
                    "replacement_attrs": (best["query_lang"], best["qtype"], best["source_lang"]),
                }
            )
        for s in selected:
            chosen.append(_finish(by_slot[s["slot_id"]], verdicts[s["slot_id"]], "pending"))
        for s in spare:
            surplus.append(_finish(by_slot[s["slot_id"]], verdicts[s["slot_id"]], None))
        result.cells[cell] = {
            "quota": quota,
            "candidates": sum(1 for s in members if s["slot_id"] in by_slot),
            "keep": counts["keep"],
            "fix": counts["fix"],
            "drop": counts["drop"],
            "no_verdict": counts["no_verdict"],
            "selected": len(selected),
            "backfilled": n_backfilled,
            "reserve": len(spare),
        }
        if len(selected) < quota:
            result.shortfalls.append(
                {
                    "cell": cell,
                    "quota": quota,
                    "selected": len(selected),
                    "short": quota - len(selected),
                }
            )

    chosen.sort(key=_order_key)
    for n, item in enumerate(chosen, 1):
        item["id"] = f"v1-{n:04d}"
    result.dev = [i for i in chosen if i["split"] == "dev"]
    result.test = [i for i in chosen if i["split"] == "test"]
    result.reserve = sorted(surplus, key=_order_key)
    return result


def stop_conditions(result: SelectResult) -> list[str]:
    """Spec section 14: more than 40% of the drafts dropped, fewer than 150 answerable survive."""
    out: list[str] = []
    total = sum(result.verdict_counts.values())
    if total and result.verdict_counts["drop"] / total > MAX_DROP_SHARE:
        share = result.verdict_counts["drop"] / total
        out.append(f"the critic dropped {share:.0%} of the drafts (limit 40%): ask before going on")
    n_answerable = sum(1 for i in result.dev + result.test if not i["is_trap"])
    if n_answerable < ANSWERABLE_TARGET:
        out.append(f"only {n_answerable} of {ANSWERABLE_TARGET} answerable items survive")
    return out


# ------------------------------------------------------------------ command


def _table(result: SelectResult) -> list[str]:
    lines = [
        f"{'split':<6}{'act':<11}{'quota':>6}{'cand':>6}{'keep':>6}{'fix':>5}{'drop':>6}"
        f"{'none':>6}{'sel':>5}{'bf':>4}{'res':>5}"
    ]
    for (split, act), c in result.cells.items():
        lines.append(
            f"{split:<6}{act or '(trap)':<11}{c['quota']:>6}{c['candidates']:>6}{c['keep']:>6}"
            f"{c['fix']:>5}{c['drop']:>6}{c['no_verdict']:>6}{c['selected']:>5}"
            f"{c['backfilled']:>4}{c['reserve']:>5}"
        )
    return lines


def run(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="benchmark select", description=__doc__.splitlines()[0])
    ap.add_argument("--candidates", type=Path, default=CANDIDATES_PATH)
    ap.add_argument("--slots", type=Path, default=SLOTS_PATH, help="the plan (defines the prefix)")
    ap.add_argument("--critic", type=Path, default=CRITIC_DIR, help="directory of verdict *.jsonl")
    ap.add_argument("--out-dir", type=Path, default=SELECTED_DIR)
    ap.add_argument("--reserve-dir", type=Path, default=RESERVE_DIR)
    ap.add_argument("--force", action="store_true", help="write the files although STOP applies")
    args = ap.parse_args(argv)

    candidates = read_jsonl(args.candidates)
    slots = read_jsonl(args.slots)
    hashes = {c["slot_id"]: c["candidate_sha256"] for c in candidates}
    verdicts, problems = load_verdicts(args.critic, hashes)
    if problems:
        print(f"{len(problems)} problem(s) in the critic files:")
        for p in problems:
            print(f"  {p}")
        print("nothing written")
        return 1
    result = select(candidates, verdicts, slots)

    print("\n".join(_table(result)))
    n_answerable = sum(1 for i in result.dev + result.test if not i["is_trap"])
    print(
        f"selected {len(result.dev) + len(result.test)} (dev {len(result.dev)}, test "
        f"{len(result.test)}; answerable {n_answerable}), reserve {len(result.reserve)}"
    )
    for b in result.backfills:
        if b["tier"] > 1:
            split, act = b["cell"]
            print(
                f"BACKFILL {split}/{act or 'trap'}: {b['dropped']} {b['dropped_attrs']} replaced "
                f"by {b['replacement']} {b['replacement_attrs']} (match tier {b['tier']})"
            )
    for s in result.shortfalls:
        split, act = s["cell"]
        label = f"{split}/{act or 'trap'}"
        print(f"SHORT {label}: {s['selected']} of {s['quota']} (short by {s['short']})")
    stops = stop_conditions(result)
    for message in stops:
        print(f"STOP: {message}")
    if stops and not args.force:
        print("nothing written (use --force to write the files anyway)")
        return 1
    write_jsonl(args.out_dir / "dev.jsonl", result.dev)
    write_jsonl(args.out_dir / "test.jsonl", result.test)
    write_jsonl(args.reserve_dir / "dev.jsonl", [i for i in result.reserve if i["split"] == "dev"])
    write_jsonl(
        args.reserve_dir / "test.jsonl", [i for i in result.reserve if i["split"] == "test"]
    )
    if stops:
        print("written although STOP applies (--force)")
    return 0
