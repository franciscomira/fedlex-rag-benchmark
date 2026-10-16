"""``benchmark brief``: the text the question writers read.

A brief shows, per slot, the slot id, the question type, the language the question must be
written in, the language the article text is shown in, and the article text with its act name,
its position in the act (division / title / chapter / section headings, in the language of the
text), its title and block heading, labelled ``A`` (and ``B`` for cross_ref).

It never shows an article id and nothing a writer could copy as one: every article-number
reference inside the displayed text is replaced (a reference to the other displayed article by
``[Article B]``, any other by ``[another article]``, ``SR 220`` by ``[SR reference]``), see
``refs.mask_article_references``. Gold ids stay in ``slots.jsonl`` and are joined by code in
``assemble``. Briefs contain Fedlex text, so ``benchmark/plan/briefs/`` is git-ignored.

Files: one per group (SR 220 in files of about 25 slots, the other acts together in files of
about 25, the traps together; pilot slots are not in them) or a single ``pilot.md``. Next to the
plan, ``index.json`` / ``index-pilot.json`` (committable, no Fedlex text) record the sha256 of the
plan the briefs were made from and, per slot, its brief file, the brief sha256 and a hash of the
slot record including its gold ids. ``assemble`` refuses drafts whose slot no longer matches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from fedlex_bench.benchmark.common import (
    BRIEFS_DIR,
    INDEX_PATH,
    INDEX_PILOT_PATH,
    LANG_NAMES,
    SLOTS_PATH,
    dumps,
    load_corpus,
    read_jsonl,
    sha256_file,
    slot_hash,
    slots_sha256,
)
from fedlex_bench.benchmark.refs import mask_article_references
from fedlex_bench.chunk import ARTICLES_PARQUET
from fedlex_bench.eligibility import retrieval_eligible

TARGET_GROUP_SIZE = 25

RULES = """\
You write one draft record per slot below, as JSON Lines: one object per line, in slot order,
written only to the file you were assigned. Rules:

- Never write or guess an article number or id. Refer to the articles only by their labels (A, B).
  Article numbers inside the text are replaced by `[Article B]` or `[another article]` on purpose.
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
NOT_AVAILABLE = "not available"


class StructureProvider(Protocol):
    def act_name(self, sr: str, lang: str) -> str | None: ...

    def path(self, article_id: str, lang: str) -> list[str] | None: ...


class NoStructure:
    """Provider for tests and for runs without the raw files: everything is "not available"."""

    def act_name(self, sr: str, lang: str) -> str | None:
        return None

    def path(self, article_id: str, lang: str) -> list[str] | None:
        return None


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
    ``other-01`` ..., then ``traps-01`` ... without the pilot slots, which are briefed in
    ``pilot.md``; slots stay in slot order inside a group."""
    ordered = sorted(slots, key=lambda s: s["slot_id"])
    if pilot:
        return {"pilot": [s for s in ordered if s["pilot"]]}
    ordered = [s for s in ordered if not s["pilot"]]
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


def _render_slot(slot: dict, lookup: dict, structure: StructureProvider) -> str:
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
    shown = list(zip("AB", slot["gold_article_ids"], strict=False))
    for label, article_id in shown:
        try:
            title, heading, text = lookup[(article_id, source)]
        except KeyError:
            raise KeyError(f"{article_id} ({source}) has no retrieval-eligible text") from None
        # references to the other displayed article by label, all others by a placeholder
        labels = {a.partition(":")[2]: other for other, a in shown if other != label}

        def mask(value: str, labels: dict[str, str] = labels) -> str:
            return mask_article_references(value, source, labels)

        sr = article_id.partition(":")[0]
        act_name = structure.act_name(sr, source)
        path = structure.path(article_id, source)
        lines += [
            "",
            f"### Article {label}",
            f"Act: {act_name or NOT_AVAILABLE}",
            f"Structure: {' > '.join(mask(p) for p in path) if path else NOT_AVAILABLE}",
            f"Title: {mask(title) if title else '(none)'}",
            f"Block heading: {mask(heading) if heading else '(none)'}",
            "",
            mask(text),
        ]
    return "\n".join(lines) + "\n"


def render_group(
    name: str, slots: list[dict], df: pd.DataFrame, structure: StructureProvider | None = None
) -> str:
    structure = structure or NoStructure()
    wanted = {a for s in slots for a in s["gold_article_ids"]}
    lookup = _article_lookup(df, wanted)
    parts = [f"# Brief {name}", "", RULES, f"Slots in this brief: {len(slots)}.", ""]
    parts += [_render_slot(s, lookup, structure) for s in slots]
    return "\n".join(parts).rstrip("\n") + "\n"


def write_briefs(
    slots: list[dict],
    df: pd.DataFrame,
    out_dir: Path,
    pilot: bool = False,
    structure: StructureProvider | None = None,
    plan_sha256: str | None = None,
    index_dir: Path | None = None,
) -> dict[str, str]:
    """Write the brief files to ``out_dir`` and the index to ``index_dir`` (default ``out_dir``);
    returns ``{file name: sha256}``. ``plan_sha256`` defaults to the hash of ``slots`` as a file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    index_dir = index_dir or out_dir
    index_dir.mkdir(parents=True, exist_ok=True)
    sums: dict[str, str] = {}
    entries: dict[str, dict] = {}
    for name, members in group_slots(slots, pilot).items():
        data = render_group(name, members, df, structure).encode("utf-8")
        (out_dir / f"{name}.md").write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        sums[f"{name}.md"] = sha
        for s in members:
            entries[s["slot_id"]] = {
                "file": f"{name}.md",
                "sha256": sha,
                "slot_sha256": slot_hash(s),
            }
    index = {"plan_sha256": plan_sha256 or slots_sha256(slots), "slots": entries}
    index_name = INDEX_PILOT_PATH.name if pilot else INDEX_PATH.name
    (index_dir / index_name).write_bytes((dumps(index) + "\n").encode("utf-8"))
    return sums


@dataclass
class BriefIndex:
    plan_sha256: str | None
    slots: dict[str, dict]


def read_index(path: Path) -> BriefIndex:
    data = json.loads(path.read_bytes().decode("utf-8"))
    return BriefIndex(plan_sha256=data.get("plan_sha256"), slots=data["slots"])


def merge_indexes(indexes: list[BriefIndex]) -> tuple[BriefIndex, list[str]]:
    """Merge several indexes (full run and pilot) into one; problems are listed, not raised."""
    problems: list[str] = []
    shas = {i.plan_sha256 for i in indexes}
    if len(shas) > 1:
        problems.append(f"the brief indexes come from different plans: {sorted(map(str, shas))}")
    merged: dict[str, dict] = {}
    for index in indexes:
        for slot_id, entry in index.slots.items():
            if slot_id in merged:
                problems.append(f"slot {slot_id} is in two brief indexes")
            merged[slot_id] = entry
    sha = indexes[0].plan_sha256 if indexes else None
    return BriefIndex(plan_sha256=sha, slots=merged), problems


def run(
    argv: list[str] | None = None,
    corpus: pd.DataFrame | None = None,
    structure: StructureProvider | None = None,
) -> int:
    ap = argparse.ArgumentParser(prog="benchmark brief", description="Write the writer briefs.")
    ap.add_argument("--slots", type=Path, default=SLOTS_PATH)
    ap.add_argument("--pilot", action="store_true", help="only the pilot slots, in pilot.md")
    ap.add_argument("--out-dir", type=Path, default=BRIEFS_DIR)
    ap.add_argument("--index-dir", type=Path, default=None, help="default: the folder of --slots")
    ap.add_argument("--parquet", type=Path, default=ARTICLES_PARQUET)
    ap.add_argument(
        "--no-structure",
        action="store_true",
        help="do not read the raw XML: act name and structure are shown as not available",
    )
    args = ap.parse_args(argv)
    slots = read_jsonl(args.slots)
    df = corpus if corpus is not None else load_corpus(args.parquet)
    if structure is None and not args.no_structure:
        from fedlex_bench.benchmark.structure import RawStructure

        structure = RawStructure()
    sums = write_briefs(
        slots,
        df,
        args.out_dir,
        pilot=args.pilot,
        structure=structure,
        plan_sha256=sha256_file(args.slots),
        index_dir=args.index_dir or args.slots.parent,
    )
    for name, sha in sums.items():
        print(f"{name} sha256 {sha}")
    print(f"wrote {len(sums)} brief file(s) to {args.out_dir}")
    return 0
