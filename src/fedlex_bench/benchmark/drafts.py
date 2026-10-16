"""Strict parsing of writer draft files (docs/benchmark_spec.md, section 5, "Draft record").

Every problem is collected with ``file:line`` and returned as a list; nothing raises on bad
input, so one broken line never hides the others. A writer never types ids: a gold fact points
to a label (``A``, ``B``), and anything that looks like an article id inside the question or a
fact is a problem.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from fedlex_bench.benchmark.common import DIFFICULTIES, numbered_lines
from fedlex_bench.benchmark.refs import mentions_article_number, mentions_sr

COMMON_KEYS = {"slot_id", "skip", "skip_reason", "note", "competing_candidates"}
# trap_reason is part of every record of the spec format: null on answerable slots
ANSWERABLE_KEYS = COMMON_KEYS | {"question", "gold_facts", "difficulty", "trap_reason"}
TRAP_KEYS = COMMON_KEYS | {"question", "trap_reason", "gold_facts", "difficulty"}
ANSWERABLE_REQUIRED = {"slot_id", "skip", "question", "gold_facts", "difficulty"}
TRAP_REQUIRED = {"slot_id", "skip", "question", "trap_reason"}
FACT_KEYS = {"text", "article"}
_ID_LIKE = re.compile(r"(?<![\w.])\d{1,3}(?:\.\d+){0,2}:\d+[a-z]*(?:-\d+[a-z]*)?(?!\w)")


@dataclass
class DraftLoad:
    records: dict[str, dict] = field(default_factory=dict)
    where: dict[str, str] = field(default_factory=dict)  # slot_id -> "file:line"
    problems: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)


def _typed_a_reference(text: str) -> bool:
    """An article id ("220:257d"), "Art./Artikel/article/articolo" + number, or "SR" + number."""
    return bool(_ID_LIKE.search(text) or mentions_article_number(text) or mentions_sr(text))


def _check_record(rec: dict, slot: dict, where: str) -> list[str]:
    """Problems of one record against its slot (``[]`` when fine)."""
    out: list[str] = []
    slot_id = slot["slot_id"]
    is_trap = slot["qtype"] == "trap"
    skip = rec.get("skip")
    if not isinstance(skip, bool):
        return [f"{where}: {slot_id}: skip must be true or false"]
    if skip:
        reason = rec.get("skip_reason")
        if not isinstance(reason, str) or not reason.strip():
            out.append(f"{where}: {slot_id}: skip is true but skip_reason is empty")
        unknown = set(rec) - TRAP_KEYS - ANSWERABLE_KEYS  # a skip record may carry null fields
        if unknown:
            out.append(f"{where}: {slot_id}: unknown key(s) {sorted(unknown)}")
        for key in ("question", "gold_facts", "difficulty", "trap_reason", "competing_candidates"):
            if rec.get(key) not in (None, "", [], {}):
                out.append(f"{where}: {slot_id}: skip record has a non-empty {key!r}")
        return out

    allowed = TRAP_KEYS if is_trap else ANSWERABLE_KEYS
    required = TRAP_REQUIRED if is_trap else ANSWERABLE_REQUIRED
    for key in sorted(required - set(rec)):
        out.append(f"{where}: {slot_id}: missing field {key!r}")
    for key in sorted(set(rec) - allowed):
        out.append(f"{where}: {slot_id}: unknown key {key!r}")
    if "skip_reason" in rec and rec["skip_reason"] not in (None, ""):
        out.append(f"{where}: {slot_id}: skip_reason is set but skip is false")

    question = rec.get("question")
    if "question" in rec:
        if not isinstance(question, str) or not question.strip():
            out.append(f"{where}: {slot_id}: question must be a non-empty string")
        elif _typed_a_reference(question):
            out.append(
                f"{where}: {slot_id}: question contains an article id, article number or SR "
                "number; writers never type ids or numbers"
            )
    cc = rec.get("competing_candidates") or []
    if not isinstance(cc, list) or not all(isinstance(x, str) for x in cc):
        out.append(f"{where}: {slot_id}: competing_candidates must be a list of strings")
    if rec.get("note") is not None and not isinstance(rec["note"], str):
        out.append(f"{where}: {slot_id}: note must be a string")

    if not is_trap and rec.get("trap_reason") is not None:
        out.append(f"{where}: {slot_id}: trap_reason must be null on a non-trap slot")
    if is_trap:
        reason = rec.get("trap_reason")
        if "trap_reason" in rec and (not isinstance(reason, str) or not reason.strip()):
            out.append(f"{where}: {slot_id}: trap_reason must be a non-empty string")
        if rec.get("gold_facts") not in (None, []):
            out.append(f"{where}: {slot_id}: a trap has no gold_facts")
        if rec.get("difficulty") is not None:
            out.append(f"{where}: {slot_id}: a trap has no difficulty")
        return out

    if "difficulty" in rec and rec["difficulty"] not in DIFFICULTIES:
        out.append(f"{where}: {slot_id}: difficulty must be one of {list(DIFFICULTIES)}")
    if "gold_facts" in rec:
        out.extend(_check_facts(rec["gold_facts"], slot, where))
    return out


def _check_facts(facts: object, slot: dict, where: str) -> list[str]:
    slot_id = slot["slot_id"]
    cross = slot["qtype"] == "cross_ref"
    labels = "AB" if cross else "A"
    limit = 4 if cross else 3
    if not isinstance(facts, list) or not 1 <= len(facts) <= limit:
        return [f"{where}: {slot_id}: gold_facts must be a list of 1 to {limit} facts"]
    out: list[str] = []
    for i, fact in enumerate(facts):
        if not isinstance(fact, dict):
            out.append(f"{where}: {slot_id}: gold_facts[{i}] must be an object")
            continue
        for key in sorted(set(fact) - FACT_KEYS):
            out.append(f"{where}: {slot_id}: gold_facts[{i}] has unknown key {key!r}")
        for key in sorted(FACT_KEYS - set(fact)):
            out.append(f"{where}: {slot_id}: gold_facts[{i}] is missing {key!r}")
        text = fact.get("text")
        if "text" in fact and (not isinstance(text, str) or not text.strip()):
            out.append(f"{where}: {slot_id}: gold_facts[{i}] text must be a non-empty string")
        elif isinstance(text, str) and _typed_a_reference(text):
            out.append(
                f"{where}: {slot_id}: gold_facts[{i}] text contains an article id, article "
                "number or SR number"
            )
        if "article" in fact and fact["article"] not in tuple(labels):
            out.append(
                f"{where}: {slot_id}: gold_facts[{i}] label {fact['article']!r} is not allowed "
                f"for a {slot['qtype']} slot (use {' or '.join(labels)})"
            )
    if cross and not out:
        used = {f["article"] for f in facts}
        if used != {"A", "B"}:
            out.append(f"{where}: {slot_id}: a cross_ref slot needs facts from both A and B")
    return out


def load_drafts(directory: Path, slots: list[dict]) -> DraftLoad:
    """Read every ``*.jsonl`` in ``directory`` (sorted) and validate each record against the plan.
    Invalid records are not added to ``records``; each problem is a ``file:line: ...`` string."""
    if not directory.is_dir():
        raise FileNotFoundError(f"drafts directory {directory} does not exist")
    by_slot = {s["slot_id"]: s for s in slots}
    load = DraftLoad()
    for path in sorted(directory.glob("*.jsonl")):
        load.files.append(path.name)
        for n, line in numbered_lines(path.read_bytes().decode("utf-8")):
            where = f"{path.name}:{n}"
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                load.problems.append(f"{where}: invalid JSON ({exc.msg})")
                continue
            if not isinstance(rec, dict):
                load.problems.append(f"{where}: a record must be a JSON object")
                continue
            slot_id = rec.get("slot_id")
            if not isinstance(slot_id, str):
                load.problems.append(f"{where}: missing field 'slot_id'")
                continue
            if slot_id not in by_slot:
                load.problems.append(f"{where}: unknown slot {slot_id}")
                continue
            if slot_id in load.where:
                load.problems.append(
                    f"{where}: duplicate draft for {slot_id} (first at {load.where[slot_id]})"
                )
                load.records.pop(slot_id, None)  # neither copy is trusted
                continue
            load.where[slot_id] = where
            found = _check_record(rec, by_slot[slot_id], where)
            if found:
                load.problems.extend(found)
            else:
                load.records[slot_id] = rec
    return load
