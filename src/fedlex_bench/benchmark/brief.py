"""``benchmark brief``: the text the question writers read.

A brief shows, per slot, the slot id, the question type, the language the question must be
written in, the language the article text is shown in, and the article text with its title and
block heading, labelled ``A`` (and ``B`` for cross_ref). It never shows an article id or a
number the writer could copy as an id: gold ids stay in ``slots.jsonl`` and are joined by code in
``assemble``. Briefs contain Fedlex text, so ``benchmark/plan/briefs/`` is git-ignored.

Files: one per group (SR 220 in files of about 25 slots, the other acts together in files of
about 25, the traps together) or a single ``pilot.md``, plus ``index.json`` / ``index-pilot.json``
mapping every slot to its brief file and sha256 (stored later in ``provenance``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from fedlex_bench.benchmark.common import (
    BRIEFS_DIR,
    LANG_NAMES,
    SLOTS_PATH,
    dumps,
    load_corpus,
    read_jsonl,
)
from fedlex_bench.chunk import ARTICLES_PARQUET
from fedlex_bench.eligibility import retrieval_eligible

TARGET_GROUP_SIZE = 25

RULES = """\
You write one draft record per slot below, as JSON Lines: one object per line, in slot order,
written only to the file you were assigned. Rules:

- Never write or guess an article number or id. Refer to the articles only by their labels (A, B).
- Write the question in `query_lang`, in plain words, with a concrete situation, the way a
  non-lawyer would ask. Do not copy the distinctive wording or numbers of the article.
- `gold_facts`: 1 to 3 short atomic facts in English (up to 4 for cross_ref), each a sentence the
  labelled article supports: `{"text": "...", "article": "A"}`.
- `difficulty`: easy = one sentence of the article states the answer; medium = needs a condition
  or several paragraphs; hard = needs an exception, a cross-reference or a numeric rule.
- `competing_candidates`: advisory text only, naming any other topic where you suspect another
  article also answers the question.
- If the article cannot support a good question, return a skip record instead.

Answerable record:
`{"slot_id": "s-0001", "skip": false, "skip_reason": null, "question": "...", "gold_facts":
[{"text": "...", "article": "A"}], "difficulty": "medium", "competing_candidates": [], "note": ""}`
Skip record: `{"slot_id": "s-0001", "skip": true, "skip_reason": "..."}`
Trap record (no facts): `{"slot_id": "s-0001", "skip": false, "skip_reason": null,
"question": "...", "trap_reason": "one sentence on why the 8 acts do not answer it"}`
"""

TRAP_INSTRUCTIONS = {
    "out_of_area": (
        "Write a question about a legal area that none of the eight acts regulates (for example "
        "tax, criminal law or social insurance). Not an area the Code of Obligations covers, such "
        "as tenancy or employment. It must sound like something a statute could answer."
    ),
    "repealed": (
        "Write a question about the content of a provision that is repealed in the current text, "
        "so that no in-force article answers it. Name the topic, not an article number."
    ),
    "unspecified_detail": (
        "Write a question that sounds answerable from a covered act but asks for a figure or a "
        "procedure the acts do not state (for example an amount, a form or a deadline that is "
        "left to another text)."
    ),
    "non_statute": (
        "Write a question that needs case law, cantonal rules, fees or administrative practice "
        "rather than the federal statute."
    ),
}
CROSS_REF_INSTRUCTION = (
    "Article A refers to Article B. Write one question whose full answer needs both articles: "
    "gold facts from A and from B, each labelled."
)
SINGLE_INSTRUCTION = "Write one question that article A alone answers."


def chunk_evenly(items: list, target: int = TARGET_GROUP_SIZE) -> list[list]:
    """Consecutive chunks of about ``target`` items, sizes differing by at most one."""
    if not items:
        return []
    n_chunks = max(1, int(len(items) / target + 0.5))
    base, extra = divmod(len(items), n_chunks)
    out, start = [], 0
    for i in range(n_chunks):
        size = base + (1 if i < extra else 0)
        out.append(items[start : start + size])
        start += size
    return out


def group_slots(slots: list[dict], pilot: bool = False) -> dict[str, list[dict]]:
    """Brief groups in file order. Pilot: ``{"pilot": [...]}``. Full run: ``220-01`` ..., then
    ``other-01`` ..., then ``traps-01`` ...; slots stay in slot order inside a group."""
    ordered = sorted(slots, key=lambda s: s["slot_id"])
    if pilot:
        return {"pilot": [s for s in ordered if s["pilot"]]}
    parts = {
        "220": [s for s in ordered if s["act"] == "220"],
        "other": [s for s in ordered if s["act"] not in (None, "220")],
        "traps": [s for s in ordered if s["act"] is None],
    }
    groups: dict[str, list[dict]] = {}
    for prefix, members in parts.items():
        for i, chunk in enumerate(chunk_evenly(members), 1):
            groups[f"{prefix}-{i:02d}"] = chunk
    return groups


def _article_lookup(
    df: pd.DataFrame, wanted: set[str]
) -> dict[tuple[str, str], tuple[str, str, str]]:
    live = retrieval_eligible(df[df["article_id"].isin(wanted)])  # filter first: it is faster
    return {
        (r.article_id, r.lang): (r.title, r.block_heading, r.text)
        for r in live.itertuples(index=False)
    }


def _render_slot(slot: dict, lookup: dict) -> str:
    lines = [f"## Slot {slot['slot_id']}", "", f"- qtype: {slot['qtype']}"]
    qlang = slot["query_lang"]
    lines.append(f"- query_lang: {qlang} (write the question in {LANG_NAMES[qlang]})")
    if slot["qtype"] == "trap":
        lines.append(f"- trap_type: {slot['trap_type']}")
        lines += ["", f"Instruction: {TRAP_INSTRUCTIONS[slot['trap_type']]}"]
        lines += ["Return `question` and `trap_reason`; no facts, no article."]
        return "\n".join(lines) + "\n"
    source = slot["source_lang"]
    lines.append(f"- source_lang: {source} (the article text below is in {LANG_NAMES[source]})")
    instruction = CROSS_REF_INSTRUCTION if slot["qtype"] == "cross_ref" else SINGLE_INSTRUCTION
    lines += ["", f"Instruction: {instruction}"]
    for label, article_id in zip("AB", slot["gold_article_ids"], strict=False):
        try:
            title, heading, text = lookup[(article_id, source)]
        except KeyError:
            raise KeyError(f"{article_id} ({source}) has no retrieval-eligible text") from None
        lines += [
            "",
            f"### Article {label}",
            f"Title: {title or '(none)'}",
            f"Block heading: {heading or '(none)'}",
            "",
            text,
        ]
    return "\n".join(lines) + "\n"


def render_group(name: str, slots: list[dict], df: pd.DataFrame) -> str:
    wanted = {a for s in slots for a in s["gold_article_ids"]}
    lookup = _article_lookup(df, wanted)
    parts = [f"# Brief {name}", "", RULES, f"Slots in this brief: {len(slots)}.", ""]
    parts += [_render_slot(s, lookup) for s in slots]
    return "\n".join(parts).rstrip("\n") + "\n"


def write_briefs(
    slots: list[dict], df: pd.DataFrame, out_dir: Path, pilot: bool = False
) -> dict[str, str]:
    """Write the brief files and the index; returns ``{file name: sha256}``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sums: dict[str, str] = {}
    index: dict[str, dict] = {}
    for name, members in group_slots(slots, pilot).items():
        data = render_group(name, members, df).encode("utf-8")
        (out_dir / f"{name}.md").write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        sums[f"{name}.md"] = sha
        for s in members:
            index[s["slot_id"]] = {"file": f"{name}.md", "sha256": sha}
    index_name = "index-pilot.json" if pilot else "index.json"
    (out_dir / index_name).write_bytes((dumps({"slots": index}) + "\n").encode("utf-8"))
    return sums


def read_index(path: Path) -> dict[str, dict]:
    return json.loads(path.read_text(encoding="utf-8"))["slots"]


def run(argv: list[str] | None = None, corpus: pd.DataFrame | None = None) -> int:
    ap = argparse.ArgumentParser(prog="benchmark brief", description="Write the writer briefs.")
    ap.add_argument("--slots", type=Path, default=SLOTS_PATH)
    ap.add_argument("--pilot", action="store_true", help="only the pilot slots, in pilot.md")
    ap.add_argument("--out-dir", type=Path, default=BRIEFS_DIR)
    ap.add_argument("--parquet", type=Path, default=ARTICLES_PARQUET)
    args = ap.parse_args(argv)
    slots = read_jsonl(args.slots)
    df = corpus if corpus is not None else load_corpus(args.parquet)
    sums = write_briefs(slots, df, args.out_dir, pilot=args.pilot)
    for name, sha in sums.items():
        print(f"{name} sha256 {sha}")
    print(f"wrote {len(sums)} brief file(s) to {args.out_dir}")
    return 0
