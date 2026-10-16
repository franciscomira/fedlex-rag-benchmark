"""``benchmark translations-plan`` and ``translations-apply``: the parallel translated subset
(docs/benchmark_spec.md section 9).

**Plan.** 24 parents (``--n``) are chosen from the selected items, 6 per query language, 6 in dev
and 18 in test overall. Parent rule, in this order:

1. only ``single`` items that are not traps, not translations, not expert-flagged and not rejected
   (by the human or by a machine judge);
2. per query language, dev gets ``dev_total // 4`` parents, plus one more for each of the first
   ``dev_total % 4`` languages in the order en, de, fr, it (6 dev parents: en 2, de 2, fr 1, it 1);
   test gets the rest of the language's share (en 4, de 4, fr 5, it 5);
3. within a (language, split) cell the order of preference is: SR 220 first, then human-accepted or
   -edited over unreviewed, then a fixed shuffle (``random.Random(20261017)`` over the sorted ids).
   Nothing is required to be reviewed.

Every parent gets one row per other query language (3 rows), so 24 parents give 72 rows. The plan
``benchmark/translations/plan.jsonl`` is sorted; a different existing plan is never overwritten
without ``--force``. The translator input ``in-<k>.jsonl`` (chunks of 12 rows) holds only
``id``, ``source_lang``, ``target_lang`` and ``question``; no gold ids.

**Apply.** ``--drafts DIR`` holds the translators' JSONL (``{parent_id, target_lang, question}``;
``id`` may stand in for the pair, ``id = "<parent_id>-<target_lang>"``). Every plan row must be
answered exactly once, nothing else may be present. Each translation becomes an item with id
``<parent_id>-<target_lang>`` (e.g. ``v1-0012-fr``), copied from its parent (question type, split,
gold, facts, source act and language, difficulty), with the question replaced, ``translation_of``
set, the BM25 ranks recomputed for the new question and status ``unreviewed``. Translations go
after the originals of the parent's split file, sorted by id. Running it twice changes nothing;
a translation already in the file with a different question is an error. The files are replaced
atomically, and if one changed on disk while the command ran nothing is written.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
import random
import tempfile
from collections.abc import Collection, Iterable, Mapping
from datetime import date
from pathlib import Path

from fedlex_bench.benchmark.bm25 import Bm25Index, build_indexes, compute_ranks
from fedlex_bench.benchmark.common import (
    BENCHMARK_DIR,
    QUERY_LANGS,
    SELECTED_DIR,
    SPLITS,
    dumps,
    load_corpus,
    read_jsonl_text,
    write_jsonl,
)
from fedlex_bench.benchmark.machine_review import MACHINE_DIR, load_verdicts
from fedlex_bench.chunk import ARTICLES_PARQUET

TRANSLATIONS_DIR = BENCHMARK_DIR / "translations"
PLAN_FILE = "plan.jsonl"
SEED = 20261017
DEFAULT_N = 24
CHUNK = 12
MAX_QUESTION_CHARS = 400
HUMAN_STATUSES = ("accepted", "edited")
DRAFT_KEYS = {"id", "parent_id", "source_lang", "target_lang", "question"}


class TranslationError(Exception):
    """The plan or the drafts do not fit; nothing was written."""


# ------------------------------------------------------------------ shared file helpers


def translation_id(parent_id: str, target_lang: str) -> str:
    return f"{parent_id}-{target_lang}"


def to_bytes(rows: Iterable[Mapping]) -> bytes:
    return "".join(dumps(r) + "\n" for r in rows).encode("utf-8")


def replace_files(changes: Mapping[Path, tuple[bytes | None, bytes]]) -> None:
    """Replace several files: ``{path: (expected current bytes or None if absent, new bytes)}``.

    Every file is compared with what the caller read first; if any differs, nothing is written
    (the review app writes in the same folder). Each replacement is atomic (temp file, then
    ``os.replace``)."""
    for path, (expected, _) in sorted(changes.items()):
        current = path.read_bytes() if path.exists() else None
        if current != expected:
            raise TranslationError(f"{path.name} changed on disk since it was read; re-run")
    for path, (_, data) in sorted(changes.items()):
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise


def read_split_files(directory: Path) -> tuple[dict[str, bytes], dict[str, list[dict]]]:
    raw: dict[str, bytes] = {}
    rows: dict[str, list[dict]] = {}
    for split in SPLITS:
        path = directory / f"{split}.jsonl"
        raw[split] = path.read_bytes()
        rows[split] = read_jsonl_text(raw[split].decode("utf-8"), str(path))
    return raw, rows


# ------------------------------------------------------------------ plan


def machine_rejected_ids(directory: Path) -> set[str]:
    """Ids that any machine judge rejected (invalid verdict files are ignored here: ``validate``
    and ``machine-status`` report them)."""
    verdicts, _ = load_verdicts(directory)
    return {
        item_id
        for item_id, by_judge in verdicts.items()
        if any(v["verdict"] == "reject" for v in by_judge.values())
    }


def parent_block_reason(parent: Mapping) -> str | None:
    """Why a translation of ``parent`` must not be made or frozen: the parent is expert-flagged
    or human-rejected (``None`` if neither)."""
    if (parent.get("provenance") or {}).get("expert_flag"):
        return "expert-flagged"
    if parent.get("review_status") == "rejected":
        return "human-rejected"
    return None


def is_parent_candidate(item: Mapping, machine_rejected: Collection[str]) -> bool:
    prov = item.get("provenance") or {}
    return (
        item["qtype"] == "single"
        and not item["is_trap"]
        and not item.get("translation_of")
        and not prov.get("expert_flag")
        and item["review_status"] != "rejected"
        and item["id"] not in machine_rejected
    )


def per_language_quota(n: int) -> dict[str, dict[str, int]]:
    """``{lang: {"dev": k, "test": m}}``: ``n // 4`` parents per language, ``n // 4`` of the ``n``
    in dev (the first languages in en, de, fr, it order take the remainder)."""
    if n <= 0 or n % len(QUERY_LANGS):
        raise TranslationError(f"--n must be a positive multiple of {len(QUERY_LANGS)}, got {n}")
    per_lang = n // len(QUERY_LANGS)
    dev_total = n // 4
    base, extra = divmod(dev_total, len(QUERY_LANGS))
    out = {}
    for i, lang in enumerate(QUERY_LANGS):
        dev = base + (1 if i < extra else 0)
        out[lang] = {"dev": dev, "test": per_lang - dev}
    return out


def choose_parents(
    items: list[dict], n: int = DEFAULT_N, machine_rejected: Collection[str] = ()
) -> list[dict]:
    """The parent items, sorted by id (see the module docstring for the rule)."""
    quota = per_language_quota(n)
    pool = sorted(
        (i for i in items if is_parent_candidate(i, machine_rejected)), key=lambda i: i["id"]
    )
    shuffled = list(pool)
    random.Random(SEED).shuffle(shuffled)
    chosen: list[dict] = []
    for lang in QUERY_LANGS:
        for split in SPLITS:
            cell = [i for i in shuffled if i["query_lang"] == lang and i["split"] == split]
            cell.sort(
                key=lambda i: (
                    i["source_act"] != "220",
                    i["review_status"] not in HUMAN_STATUSES,
                )
            )  # stable: ties keep the seeded shuffle order
            want = quota[lang][split]
            if len(cell) < want:
                raise TranslationError(
                    f"not enough parents for query_lang={lang} split={split}: "
                    f"{len(cell)} eligible, {want} needed"
                )
            chosen.extend(cell[:want])
    return sorted(chosen, key=lambda i: i["id"])


def plan_rows(parents: list[dict]) -> list[dict]:
    rows = [
        {
            "parent_id": p["id"],
            "target_lang": lang,
            "parent_question": p["question"],
            "parent_query_lang": p["query_lang"],
        }
        for p in parents
        for lang in QUERY_LANGS
        if lang != p["query_lang"]
    ]
    return sorted(rows, key=lambda r: (r["parent_id"], r["target_lang"]))


def input_chunks(rows: list[dict], size: int = CHUNK) -> list[list[dict]]:
    """Translator inputs: id, source language, target language and question; no gold ids."""
    flat = [
        {
            "id": translation_id(r["parent_id"], r["target_lang"]),
            "source_lang": r["parent_query_lang"],
            "target_lang": r["target_lang"],
            "question": r["parent_question"],
        }
        for r in rows
    ]
    return [flat[i : i + size] for i in range(0, len(flat), size)]


def write_plan(
    rows: list[dict], out_dir: Path, force: bool = False, dry_run: bool = False
) -> list[Path]:
    """Write plan.jsonl and the in-<k>.jsonl chunks; returns the written paths."""
    plan_path = out_dir / PLAN_FILE
    data = to_bytes(rows)
    if plan_path.exists() and plan_path.read_bytes() != data and not force:
        raise TranslationError(
            f"{plan_path} exists and differs from the new plan; pass --force to overwrite it"
        )
    chunks = input_chunks(rows)
    names = [f"in-{k}.jsonl" for k in range(1, len(chunks) + 1)]
    written = [plan_path] + [out_dir / name for name in names]
    if dry_run:
        return written
    write_jsonl(plan_path, rows)
    for name, chunk in zip(names, chunks, strict=True):
        write_jsonl(out_dir / name, chunk)
    for stale in sorted(out_dir.glob("in-*.jsonl")):
        if stale.name not in names:
            stale.unlink()
    return written


# ------------------------------------------------------------------ drafts


def load_plan(path: Path) -> tuple[list[dict], str]:
    data = path.read_bytes()
    rows = read_jsonl_text(data.decode("utf-8"), str(path))
    keys = {"parent_id", "target_lang", "parent_question", "parent_query_lang"}
    seen = set()
    for n, r in enumerate(rows, 1):
        if not isinstance(r, dict) or set(r) != keys:
            raise TranslationError(f"{path}:{n}: a plan row has exactly the keys {sorted(keys)}")
        key = (r["parent_id"], r["target_lang"])
        if key in seen:
            raise TranslationError(f"{path}:{n}: duplicate plan row {key}")
        seen.add(key)
    return rows, hashlib.sha256(data).hexdigest()


def load_drafts(directory: Path, plan: list[dict]) -> dict[tuple[str, str], str]:
    """``{(parent_id, target_lang): question}`` from every ``*.jsonl`` in ``directory``.

    Strict: bad JSON, unknown keys, a pair that is not in the plan, a pair answered twice, an
    empty or too long question, or a plan row left unanswered are all errors (all listed)."""
    if not directory.is_dir():
        raise TranslationError(f"drafts folder {directory} does not exist")
    wanted = {(r["parent_id"], r["target_lang"]) for r in plan}
    drafts: dict[tuple[str, str], str] = {}
    problems: list[str] = []
    for path in sorted(directory.glob("*.jsonl")):
        text = path.read_bytes().decode("utf-8")
        try:
            records = read_jsonl_text(text, path.name)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        for n, rec in enumerate(records, 1):
            where = f"{path.name}:{n}"
            if not isinstance(rec, dict) or not set(rec) <= DRAFT_KEYS or "question" not in rec:
                problems.append(f"{where}: keys must be parent_id, target_lang, question")
                continue
            key = _draft_key(rec)
            if key is None:
                problems.append(f"{where}: needs parent_id and target_lang (or a matching id)")
                continue
            question = rec["question"]
            if not isinstance(question, str) or not question.strip():
                problems.append(f"{where}: {key}: question must be a non-empty string")
                continue
            if len(question) > MAX_QUESTION_CHARS:
                problems.append(
                    f"{where}: {key}: {len(question)} characters (limit {MAX_QUESTION_CHARS})"
                )
                continue
            if key not in wanted:
                problems.append(f"{where}: {key} is not in the plan")
                continue
            if key in drafts:
                problems.append(f"{where}: {key} answered more than once")
                continue
            drafts[key] = question
    for key in sorted(wanted - set(drafts)):
        problems.append(f"missing translation for {key}")
    if problems:
        raise TranslationError("drafts rejected:\n  " + "\n  ".join(problems))
    return drafts


def _draft_key(rec: Mapping) -> tuple[str, str] | None:
    parent, lang, rid = rec.get("parent_id"), rec.get("target_lang"), rec.get("id")
    if isinstance(parent, str) and isinstance(lang, str):
        if rid is not None and rid != translation_id(parent, lang):
            return None
        return parent, lang
    if isinstance(rid, str) and "-" in rid:
        parent, _, lang = rid.rpartition("-")
        if lang in QUERY_LANGS and (rec.get("target_lang") in (None, lang)):
            return parent, lang
    return None


# ------------------------------------------------------------------ apply


def build_translation(
    parent: Mapping,
    target_lang: str,
    question: str,
    indexes: Mapping[str, Bm25Index],
    *,
    plan_sha256: str,
    translated_on: str,
    model_note: str,
    agent_file_sha256: str | None = None,
) -> dict:
    item = copy.deepcopy(dict(parent))
    ranks, overlap = compute_ranks(question, parent["gold_article_ids"], target_lang, indexes)
    prov = parent.get("provenance") or {}
    item.update(
        id=translation_id(parent["id"], target_lang),
        query_lang=target_lang,
        question=question,
        translation_of=parent["id"],
        bm25_rank=ranks,
        lexical_overlap=overlap,
        review_status="unreviewed",
        reviewed_by=None,
        reviewed_at=None,
        review_note=None,
        provenance={
            "drafter": "translator",
            "model": model_note,
            "agent_file_sha256": agent_file_sha256,
            "plan_sha256": plan_sha256,
            "drafted_on": translated_on,
            "translated_on": translated_on,
            "critic_verdict": None,
            "critic_rounds": 0,
            "critic_notes": None,
            "candidate_sha256": prov.get("candidate_sha256"),
        },
    )
    return item


def apply_translations(
    plan: list[dict],
    drafts: Mapping[tuple[str, str], str],
    directory: Path,
    indexes: Mapping[str, Bm25Index],
    *,
    plan_sha256: str,
    translated_on: str,
    model_note: str,
    agent_file_sha256: str | None = None,
    dry_run: bool = False,
) -> dict[str, list[str]]:
    """Add the translations to ``directory``; returns ``{"added": ids, "unchanged": ids,
    "skipped": ids}``. A plan row whose parent is expert-flagged or human-rejected is skipped
    (not added, an existing translation is left as it is) and listed under ``skipped``."""
    raw, rows = read_split_files(directory)
    by_id = {it["id"]: it for split in SPLITS for it in rows[split]}
    new: dict[str, list[dict]] = {split: [] for split in SPLITS}
    added: list[str] = []
    unchanged: list[str] = []
    skipped: list[str] = []
    for row in plan:
        parent_id, lang = row["parent_id"], row["target_lang"]
        parent = by_id.get(parent_id)
        if parent is None:
            raise TranslationError(f"{parent_id}: parent is not in {directory}")
        if parent_block_reason(parent):
            skipped.append(translation_id(parent_id, lang))
            continue
        if parent.get("translation_of") or parent["qtype"] != "single" or parent["is_trap"]:
            raise TranslationError(f"{parent_id}: parent must be a single, non-trap original")
        if parent["question"] != row["parent_question"]:
            raise TranslationError(
                f"{parent_id}: the parent question changed since the plan was made; re-plan"
            )
        if parent["query_lang"] == lang:
            raise TranslationError(f"{parent_id}: target language equals the parent's")
        question = drafts[(parent_id, lang)]
        item_id = translation_id(parent_id, lang)
        existing = by_id.get(item_id)
        if existing is not None:
            if existing.get("translation_of") != parent_id or existing["question"] != question:
                raise TranslationError(
                    f"{item_id} already exists with a different question; resolve by hand"
                )
            unchanged.append(item_id)
            continue
        item = build_translation(
            parent,
            lang,
            question,
            indexes,
            plan_sha256=plan_sha256,
            translated_on=translated_on,
            model_note=model_note,
            agent_file_sha256=agent_file_sha256,
        )
        new[parent["split"]].append(item)
        added.append(item_id)
    if not dry_run and added:
        changes = {}
        for split in SPLITS:
            if not new[split]:
                continue
            originals = [it for it in rows[split] if not it.get("translation_of")]
            existing_tr = [it for it in rows[split] if it.get("translation_of")]
            ordered = originals + sorted(existing_tr + new[split], key=lambda it: it["id"])
            changes[directory / f"{split}.jsonl"] = (raw[split], to_bytes(ordered))
        replace_files(changes)
    return {"added": sorted(added), "unchanged": sorted(unchanged), "skipped": sorted(skipped)}


# ------------------------------------------------------------------ commands


def run_plan(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="benchmark translations-plan", description=__doc__.splitlines()[0]
    )
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="parent items (default 24)")
    ap.add_argument("--dir", type=Path, default=SELECTED_DIR, help="folder with dev/test.jsonl")
    ap.add_argument("--out-dir", type=Path, default=TRANSLATIONS_DIR)
    ap.add_argument("--machine-dir", type=Path, default=MACHINE_DIR)
    ap.add_argument("--force", action="store_true", help="overwrite a different plan")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args(argv)
    try:
        _, rows = read_split_files(args.dir)
        items = [it for split in SPLITS for it in rows[split]]
        parents = choose_parents(items, args.n, machine_rejected_ids(args.machine_dir))
        plan = plan_rows(parents)
        written = write_plan(plan, args.out_dir, force=args.force, dry_run=args.dry_run)
    except (TranslationError, ValueError, OSError) as exc:
        print(f"ERROR   {exc}")
        return 1
    for p in parents:
        print(
            f"{p['id']}\t{p['split']}\t{p['query_lang']}\tSR {p['source_act']}\t"
            f"{p['review_status']}"
        )
    by_cell = {
        (lang, split): sum(1 for p in parents if p["query_lang"] == lang and p["split"] == split)
        for lang in QUERY_LANGS
        for split in SPLITS
    }
    print(
        "parents per (language, split): "
        + ", ".join(f"{lang}/{split}={n}" for (lang, split), n in sorted(by_cell.items()))
    )
    verb = "would write" if args.dry_run else "wrote"
    print(f"{verb} {len(plan)} plan rows from {len(parents)} parents; files: ")
    for p in written:
        print(f"  {p}")
    return 0


def run_apply(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="benchmark translations-apply", description=__doc__.splitlines()[0]
    )
    ap.add_argument("--plan", type=Path, default=TRANSLATIONS_DIR / PLAN_FILE)
    ap.add_argument("--drafts", type=Path, required=True, help="folder with translator JSONL")
    ap.add_argument("--dir", type=Path, default=SELECTED_DIR, help="folder with dev/test.jsonl")
    ap.add_argument("--parquet", type=Path, default=ARTICLES_PARQUET)
    ap.add_argument("--model-note", default="llm translator (model alias not recorded)")
    ap.add_argument("--agent-file", type=Path, default=None, help="translator agent file (hashed)")
    ap.add_argument("--date", default=None, help="translated_on, YYYY-MM-DD (default: today)")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args(argv)
    try:
        plan, plan_sha = load_plan(args.plan)
        drafts = load_drafts(args.drafts, plan)
        agent_sha = (
            hashlib.sha256(args.agent_file.read_bytes()).hexdigest() if args.agent_file else None
        )
        indexes = build_indexes(load_corpus(args.parquet))
        result = apply_translations(
            plan,
            drafts,
            args.dir,
            indexes,
            plan_sha256=plan_sha,
            translated_on=args.date or date.today().isoformat(),
            model_note=args.model_note,
            agent_file_sha256=agent_sha,
            dry_run=args.dry_run,
        )
    except (TranslationError, ValueError, OSError) as exc:
        print(f"ERROR   {exc}")
        return 1
    for item_id in result["added"]:
        print(f"{item_id}\tadded")
    for item_id in result["skipped"]:
        print(f"{item_id}\tskipped: the parent is expert-flagged or human-rejected")
    verb = "would add" if args.dry_run else "added"
    print(
        f"{verb} {len(result['added'])} translation(s), {len(result['unchanged'])} already "
        f"present, of {len(plan)} plan rows; all unreviewed"
    )
    return 0
