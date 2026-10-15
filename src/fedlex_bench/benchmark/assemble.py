"""``benchmark assemble``: writer drafts + plan -> candidate items (docs/benchmark_spec.md).

Everything that must be exact is copied from the plan or computed here: ids, act, languages,
split, BM25 ranks and lexical overlap. The writer's text supplies only the question, the gold
facts, the difficulty and, for traps, the reason. ``review_status`` is always ``unreviewed``.
A candidate is an item without an id plus ``slot_id`` (the critic answers by slot id); ``select``
turns the kept candidates into the dev and test files.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from fedlex_bench.benchmark import drafts as drafts_mod
from fedlex_bench.benchmark.bm25 import Bm25Index, build_indexes, compute_ranks
from fedlex_bench.benchmark.brief import read_index
from fedlex_bench.benchmark.common import (
    CANDIDATES_PATH,
    DRAFTS_DIR,
    SLOTS_PATH,
    load_corpus,
    read_jsonl,
    sha256_file,
    write_jsonl,
)
from fedlex_bench.chunk import ARTICLES_PARQUET

_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


@dataclass
class AssembleResult:
    candidates: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    missing_slots: list[str] = field(default_factory=list)  # in the plan, no draft at all


def _item(
    slot: dict,
    rec: dict,
    indexes: dict[str, Bm25Index],
    provenance: dict,
) -> dict:
    is_trap = slot["qtype"] == "trap"
    gold = list(slot["gold_article_ids"])
    facts = [
        {"text": f["text"].strip(), "article_id": gold["AB".index(f["article"])]}
        for f in rec.get("gold_facts") or []
    ]
    question = rec["question"].strip()
    if is_trap:
        ranks = overlap = None
    else:
        ranks, overlap = compute_ranks(question, gold, slot["query_lang"], indexes)
    return {
        "id": None,
        "slot_id": slot["slot_id"],
        "split": slot["split"],
        "qtype": slot["qtype"],
        "is_trap": is_trap,
        "query_lang": slot["query_lang"],
        "question": question,
        "source_lang": slot["source_lang"],
        "gold_article_ids": gold,
        "gold_facts": facts,
        "source_act": slot["act"],
        "difficulty": None if is_trap else rec["difficulty"],
        "lexical_overlap": overlap,
        "bm25_rank": ranks,
        "trap_type": slot["trap_type"] if is_trap else None,
        "trap_reason": rec["trap_reason"].strip() if is_trap else None,
        "translation_of": None,
        "provenance": {
            **provenance,
            "slot_id": slot["slot_id"],
            "writer_note": rec.get("note") or "",
            "competing_candidates": list(rec.get("competing_candidates") or []),
        },
        "review_status": "unreviewed",
        "reviewed_by": None,
        "reviewed_at": None,
        "review_note": None,
    }


def assemble(
    slots: list[dict],
    records: dict[str, dict],
    df: pd.DataFrame,
    *,
    drafted_on: str,
    indexes: dict[str, Bm25Index] | None = None,
    agent_file: Path | None = None,
    plan_sha256: str | None = None,
    drafter: str = "question-writer",
    model: str = "opus",
    brief_index: dict[str, dict] | None = None,
) -> AssembleResult:
    """Merge valid draft records (see ``drafts.load_drafts``) with the plan."""
    if not _DATE.fullmatch(drafted_on):
        raise ValueError(f"drafted_on must be YYYY-MM-DD, got {drafted_on!r}")
    indexes = indexes if indexes is not None else build_indexes(df)
    agent_sha = sha256_file(agent_file) if agent_file is not None else None
    base = {
        "drafter": drafter,
        "model": model,
        "agent_file_sha256": agent_sha,
        "plan_sha256": plan_sha256,
        "drafted_on": drafted_on,
        "critic_verdict": None,
        "critic_rounds": 0,
        "critic_notes": None,
    }
    result = AssembleResult()
    for slot in sorted(slots, key=lambda s: s["slot_id"]):
        rec = records.get(slot["slot_id"])
        if rec is None:
            result.missing_slots.append(slot["slot_id"])
            continue
        if rec["skip"]:
            result.skipped.append({"slot_id": slot["slot_id"], "skip_reason": rec["skip_reason"]})
            continue
        brief_sha = (brief_index or {}).get(slot["slot_id"], {}).get("sha256")
        provenance = {**base, "brief_sha256": brief_sha}
        result.candidates.append(_item(slot, rec, indexes, provenance))
    return result


def run(argv: list[str] | None = None, corpus: pd.DataFrame | None = None) -> int:
    ap = argparse.ArgumentParser(prog="benchmark assemble", description=__doc__.splitlines()[0])
    ap.add_argument("--drafts", type=Path, default=DRAFTS_DIR, help="directory of writer *.jsonl")
    ap.add_argument("--slots", type=Path, default=SLOTS_PATH)
    ap.add_argument("--out", type=Path, default=CANDIDATES_PATH)
    ap.add_argument(
        "--agent-file", type=Path, help="role file of the writer (its sha256 is stored)"
    )
    ap.add_argument("--drafted-on", required=True, help="date of drafting, YYYY-MM-DD")
    ap.add_argument("--drafter", default="question-writer")
    ap.add_argument("--model", default="opus", help="model alias as given to the harness")
    ap.add_argument("--brief-index", type=Path, help="briefs/index.json (stores the brief sha256)")
    ap.add_argument("--parquet", type=Path, default=ARTICLES_PARQUET)
    ap.add_argument("--allow-partial", action="store_true", help="write the valid records anyway")
    args = ap.parse_args(argv)

    slots = read_jsonl(args.slots)
    loaded = drafts_mod.load_drafts(args.drafts, slots)
    print(f"read {len(loaded.records)} valid draft records from {len(loaded.files)} file(s)")
    if loaded.problems:
        print(f"{len(loaded.problems)} problem(s) in the drafts:")
        for p in loaded.problems:
            print(f"  {p}")
        if not args.allow_partial:
            print("nothing written (use --allow-partial to write the valid records)")
            return 1
    df = corpus if corpus is not None else load_corpus(args.parquet)
    result = assemble(
        slots,
        loaded.records,
        df,
        drafted_on=args.drafted_on,
        agent_file=args.agent_file,
        plan_sha256=sha256_file(args.slots),
        drafter=args.drafter,
        model=args.model,
        brief_index=read_index(args.brief_index) if args.brief_index else None,
    )
    sha = write_jsonl(args.out, result.candidates)
    if result.skipped:
        write_jsonl(args.out.with_name("skipped.jsonl"), result.skipped)
    print(
        f"{len(result.candidates)} candidates, {len(result.skipped)} skipped, "
        f"{len(result.missing_slots)} slots without a draft"
    )
    for s in result.skipped:
        print(f"  skipped {s['slot_id']}: {s['skip_reason']}")
    print(f"wrote {args.out} (sha256 {sha})")
    return 1 if loaded.problems else 0
