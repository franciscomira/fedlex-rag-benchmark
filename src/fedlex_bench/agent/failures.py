"""Rule-based failure classification of agent traces (docs/eval_spec.md section 10).

The nine categories are fixed. No LLM is involved: everything is derived from the trace and the
gold ids. Rules (an item can match several; `primary` is the first by precedence):

  malformed_output            no final step, or the final answer failed to parse
  answered_trap               trap item, but status == answered or citations non-empty
  wrong_language_retriever    answerable, no gold retrieved, and at least one of de/fr/it was
                              never searched (a retriever that could hold the gold was not tried)
  stopped_too_early           answerable, no gold retrieved, all languages tried, but the agent
                              stopped by its own decision (planner done / nothing new / checker
                              accepted) with budget left
  no_gold_retrieved           answerable, no gold retrieved, budget exhausted
  refused_answerable          answerable item with status == refused
  gold_retrieved_not_cited    answered, some gold retrieved, no gold id cited
  invalid_citation            the answerer produced an id outside the candidate list
  looped_without_new_evidence another failure, and the last round (of 2+) added no new article

Secondary `flags` (not categories): step_limit_hit, checker_rejected (unsupported claim),
missed_reference (gold not retrieved but named in a retrieved article, never followed),
tool_error, partial_gold (only some gold ids cited).

CLI: python -m fedlex_bench.agent.failures --traces <traces.jsonl> --items <dev.jsonl> [--out f]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CATEGORIES: tuple[str, ...] = (
    "malformed_output",
    "answered_trap",
    "wrong_language_retriever",
    "stopped_too_early",
    "no_gold_retrieved",
    "refused_answerable",
    "gold_retrieved_not_cited",
    "invalid_citation",
    "looped_without_new_evidence",
)  # precedence order
OK = "ok"
ALL_LANGS = {"de", "fr", "it"}
SELF_STOPS = {"planner_done", "no_new_actions", "verified"}


@dataclass(frozen=True)
class FailureReport:
    primary: str  # a CATEGORIES entry or "ok"
    categories: list[str]
    flags: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


def _tool_steps(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [s for s in trace if s.get("kind") == "tool"]


def classify_trace(
    trace: list[dict[str, Any]], gold_ids: list[str], is_trap: bool = False
) -> FailureReport:
    """Classify one run. `gold_ids` is empty for traps."""
    gold = set(gold_ids)
    finals = [s for s in trace if s.get("kind") == "final"]
    tools = _tool_steps(trace)
    retrieved = {i for s in tools if not s.get("error") for i in (s.get("result_ids") or [])}
    refs_seen = {r for s in tools for r in (s.get("refs_seen") or [])}
    langs_tried = {(s.get("args") or {}).get("lang") for s in tools}
    followed = {
        (s.get("args") or {}).get("id") for s in tools if s.get("tool") == "follow_reference"
    }
    cats: list[str] = []
    flags: list[str] = []

    if not finals or finals[-1].get("parse_failed") or finals[-1].get("error"):
        cats.append("malformed_output")
    final = finals[-1] if finals else {}
    status = final.get("status")
    citations = set(final.get("citations") or [])

    if final.get("step_limit_hit"):
        flags.append("step_limit_hit")
    if final.get("rejected"):
        flags.append("checker_rejected")
    if any(s.get("error") for s in tools):
        flags.append("tool_error")

    if "malformed_output" not in cats:
        if is_trap:
            if status == "answered" or citations:
                cats.append("answered_trap")
        else:
            found = gold & retrieved
            if not found:
                if gold & refs_seen and not followed:
                    flags.append("missed_reference")
                if ALL_LANGS - langs_tried:
                    cats.append("wrong_language_retriever")
                elif final.get("stop_reason") in SELF_STOPS:
                    cats.append("stopped_too_early")
                else:
                    cats.append("no_gold_retrieved")
            if status == "refused":
                cats.append("refused_answerable")
            elif found and not (gold & citations):
                cats.append("gold_retrieved_not_cited")
            elif gold & citations and gold - citations:
                flags.append("partial_gold")
        if final.get("invalid_citations"):
            cats.append("invalid_citation")
        gains = [s.get("new_ids_count", 0) for s in trace if s.get("kind") == "round_end"]
        if cats and len(gains) >= 2 and gains[-1] == 0:
            cats.append("looped_without_new_evidence")

    cats = [c for c in CATEGORIES if c in cats]  # fixed precedence, deduplicated
    details = {
        "n_retrieved": len(retrieved),
        "gold_retrieved": sorted(gold & retrieved),
        "gold_cited": sorted(gold & citations),
        "rounds": final.get("rounds"),
        "stop_reason": final.get("stop_reason"),
        "langs_tried": sorted(x for x in langs_tried if x),
    }
    return FailureReport(cats[0] if cats else OK, cats, flags, details)


# -- CLI ------------------------------------------------------------------------------------
def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def classify_files(traces_path: Path, items_path: Path) -> list[dict[str, Any]]:
    items = _read_jsonl(items_path)
    by_id = {it["id"]: it for it in items if "id" in it}
    by_text = {(it.get("query_lang"), it.get("question")): it for it in items}
    rows: list[dict[str, Any]] = []
    for rec in _read_jsonl(traces_path):
        item = by_id.get(rec.get("item_id") or rec.get("id")) or by_text.get(
            (rec.get("query_lang"), rec.get("query"))
        )
        if item is None:
            rows.append({"id": None, "query": rec.get("query"), "primary": "unmatched_item"})
            continue
        rep = classify_trace(
            rec.get("trace") or [], item.get("gold_article_ids") or [], bool(item.get("is_trap"))
        )
        rows.append({
            "id": item["id"], "is_trap": bool(item.get("is_trap")), "primary": rep.primary,
            "categories": rep.categories, "flags": rep.flags, "details": rep.details,
        })  # fmt: skip
    return sorted(rows, key=lambda r: (str(r.get("id")), r["primary"]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Classify agent failures from trace JSONL.")
    ap.add_argument("--traces", type=Path, required=True)
    ap.add_argument("--items", type=Path, required=True, help="benchmark split JSONL (dev)")
    ap.add_argument("--out", type=Path, help="write per-item classification JSONL here")
    args = ap.parse_args(argv)
    rows = classify_files(args.traces, args.items)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8", newline="\n") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    counts = Counter(r["primary"] for r in rows)
    flag_counts = Counter(f for r in rows for f in r.get("flags", []))
    print(f"items classified: {len(rows)}", file=sys.stdout)
    for name in (OK, *CATEGORIES, "unmatched_item"):
        if counts.get(name):
            print(f"  {name:30s} {counts[name]}")
    for name, n in sorted(flag_counts.items()):
        print(f"  flag {name:25s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
