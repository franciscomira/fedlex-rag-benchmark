"""``benchmark validate``: the checks of docs/benchmark_spec.md, section 10.

Every violation is reported with the item id and a short check code; errors make the exit code
non-zero, warnings do not. Checks that need the corpus (gold ids exist, are eligible, ranks) use
``data/processed/articles.parquet``.

Errors: ``json``, ``files``, ``schema``, ``is_trap``, ``id_format``, ``id_order``,
``id_duplicate``, ``file_split``, ``gold_count``, ``gold_missing``, ``gold_ineligible``,
``source_act``, ``split_rule``, ``gold_in_both_splits``, ``gold_reused``, ``translation``,
``fact_article``, ``fact_count``, ``trap_fields``, ``answerable_fields``, ``question_length``,
``question_duplicate``, ``question_near_duplicate``, ``lang_balance``, ``act_quota``,
``provenance``, ``review_fields``, ``rank_mismatch`` and, with ``--freeze``, ``freeze_status`` and
``freeze_reviewer``.

Warnings: ``warn_same_lang_rank1`` (more than 40% of the non-English items have BM25 rank 1 in
their own language), ``warn_pair_cell`` (fewer than 20 items in a query/source language cell),
``warn_agent_file`` (provenance without an agent file hash).

The language balance and the act quotas apply to a complete benchmark. ``--partial`` skips them
for a set that is still being filled (the selected files of an unfinished run).
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from fedlex_bench.benchmark.bm25 import Bm25Index, build_indexes, compute_ranks
from fedlex_bench.benchmark.common import (
    ACT_QUOTA,
    CORPUS_LANGS,
    DIFFICULTIES,
    QTYPES,
    QUERY_LANGS,
    SELECTED_DIR,
    SPLITS,
    TRAP_TYPES,
    id_key,
    load_corpus,
    read_jsonl,
    split_of,
)
from fedlex_bench.chunk import ARTICLES_PARQUET
from fedlex_bench.eligibility import benchmark_eligible

ITEM_KEYS = {
    "id",
    "split",
    "qtype",
    "is_trap",
    "query_lang",
    "question",
    "source_lang",
    "gold_article_ids",
    "gold_facts",
    "source_act",
    "difficulty",
    "lexical_overlap",
    "bm25_rank",
    "trap_type",
    "trap_reason",
    "translation_of",
    "provenance",
    "review_status",
    "reviewed_by",
    "reviewed_at",
    "review_note",
}
PROVENANCE_REQUIRED = {
    "drafter",
    "model",
    "agent_file_sha256",
    "plan_sha256",
    "drafted_on",
    "critic_verdict",
    "critic_rounds",
    "critic_notes",
}
REVIEW_STATUSES = ("unreviewed", "accepted", "edited", "rejected")
FROZEN_STATUSES = ("accepted", "edited")
MAX_QUESTION_CHARS = 400
LANG_SHARE = (0.22, 0.28)
QUOTA_TOLERANCE = 1
NEAR_DUPLICATE_JACCARD = 0.8
SAME_LANG_RANK1_SHARE = 0.40
MIN_PAIR_CELL = 20
_ID_RE = re.compile(r"v1-\d{4}")
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_AGENT_RE = re.compile(
    r"agent|claude|opus|sonnet|haiku|gpt|llm|\bai\b|\bbot\b|assistant|critic|writer|model|gemini"
    r"|copilot",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Violation:
    check: str
    item_id: str
    message: str

    def __str__(self) -> str:
        return f"{self.item_id}: [{self.check}] {self.message}"


@dataclass
class Report:
    errors: list[Violation] = field(default_factory=list)
    warnings: list[Violation] = field(default_factory=list)

    def error(self, check: str, item_id: str, message: str) -> None:
        self.errors.append(Violation(check, item_id, message))

    def warn(self, check: str, item_id: str, message: str) -> None:
        self.warnings.append(Violation(check, item_id, message))

    @property
    def codes(self) -> set[str]:
        return {v.check for v in self.errors}

    @property
    def warning_codes(self) -> set[str]:
        return {v.check for v in self.warnings}


# ------------------------------------------------------------------ schema


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_str_or_none(value: object) -> bool:
    return value is None or isinstance(value, str)


def schema_problems(item: dict) -> list[str]:
    """Everything wrong with the shape and the types of one item (``[]`` if fine)."""
    out: list[str] = []
    missing, extra = ITEM_KEYS - set(item), set(item) - ITEM_KEYS
    if missing:
        out.append(f"missing key(s) {sorted(missing)}")
    if extra:
        out.append(f"unknown key(s) {sorted(extra)}")
    if missing:
        return out

    def check(ok: bool, message: str) -> None:
        if not ok:
            out.append(message)

    check(isinstance(item["id"], str), "id must be a string")
    check(item["split"] in SPLITS, f"split must be one of {list(SPLITS)}")
    check(item["qtype"] in QTYPES, f"qtype must be one of {list(QTYPES)}")
    check(isinstance(item["is_trap"], bool), "is_trap must be a boolean")
    check(item["query_lang"] in QUERY_LANGS, f"query_lang must be one of {list(QUERY_LANGS)}")
    check(
        isinstance(item["question"], str) and item["question"].strip() != "",
        "question must be a non-empty string",
    )
    check(
        item["source_lang"] is None or item["source_lang"] in CORPUS_LANGS,
        "source_lang must be de, fr, it or null",
    )
    gold = item["gold_article_ids"]
    check(
        isinstance(gold, list) and all(isinstance(g, str) for g in gold),
        "gold_article_ids must be a list of strings",
    )
    facts = item["gold_facts"]
    facts_ok = isinstance(facts, list) and all(
        isinstance(f, dict)
        and set(f) == {"text", "article_id"}
        and isinstance(f["text"], str)
        and isinstance(f["article_id"], str)
        for f in facts
    )
    check(facts_ok, "gold_facts must be a list of {text, article_id} objects")
    check(_is_str_or_none(item["source_act"]), "source_act must be a string or null")
    check(
        item["difficulty"] is None or item["difficulty"] in DIFFICULTIES,
        f"difficulty must be one of {list(DIFFICULTIES)} or null",
    )
    check(
        item["lexical_overlap"] is None or _is_int(item["lexical_overlap"]),
        "lexical_overlap must be an integer or null",
    )
    ranks = item["bm25_rank"]
    check(
        ranks is None
        or (
            isinstance(ranks, dict)
            and set(ranks) == set(CORPUS_LANGS)
            and all(_is_int(v) and v >= 1 for v in ranks.values())
        ),
        "bm25_rank must be null or {de, fr, it} with integers >= 1",
    )
    check(
        item["trap_type"] is None or item["trap_type"] in TRAP_TYPES,
        f"trap_type must be one of {list(TRAP_TYPES)} or null",
    )
    check(_is_str_or_none(item["trap_reason"]), "trap_reason must be a string or null")
    check(_is_str_or_none(item["translation_of"]), "translation_of must be a string or null")
    check(isinstance(item["provenance"], dict), "provenance must be an object")
    check(
        item["review_status"] in REVIEW_STATUSES,
        f"review_status must be one of {list(REVIEW_STATUSES)}",
    )
    for key in ("reviewed_by", "reviewed_at", "review_note"):
        check(_is_str_or_none(item[key]), f"{key} must be a string or null")
    return out


def provenance_problems(prov: dict) -> list[str]:
    out = [f"missing provenance key {k!r}" for k in sorted(PROVENANCE_REQUIRED - set(prov))]
    if out:
        return out
    for key in ("drafter", "model"):
        if not isinstance(prov[key], str) or not prov[key]:
            out.append(f"provenance.{key} must be a non-empty string")
    for key in ("agent_file_sha256", "plan_sha256"):
        if prov[key] is not None and not (
            isinstance(prov[key], str) and _SHA_RE.fullmatch(prov[key])
        ):
            out.append(f"provenance.{key} must be a sha256 hex string or null")
    if not (
        isinstance(prov["drafted_on"], str)
        and re.fullmatch(r"\d{4}-\d{2}-\d{2}", prov["drafted_on"])
    ):
        out.append("provenance.drafted_on must be YYYY-MM-DD")
    if prov["critic_verdict"] not in (None, "keep", "fix", "drop"):
        out.append("provenance.critic_verdict must be keep, fix, drop or null")
    if not _is_int(prov["critic_rounds"]) or prov["critic_rounds"] < 0:
        out.append("provenance.critic_rounds must be an integer >= 0")
    if not _is_str_or_none(prov["critic_notes"]):
        out.append("provenance.critic_notes must be a string or null")
    return out


# ------------------------------------------------------------------ text helpers


def normalise_question(text: str) -> str:
    """Lowercase, letters and digits only, single spaces: the form compared for duplicates."""
    return " ".join(re.sub(r"[\W_]+", " ", text.lower()).split())


def char_ngrams(text: str, n: int = 5) -> set[str]:
    return {text[i : i + n] for i in range(max(0, len(text) - n + 1))} or {text}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


# ------------------------------------------------------------------ the checks


class CorpusFacts(NamedTuple):
    """What the checks need to know about the corpus: the benchmark pool and the ids present in
    all three languages. Computing it takes a moment; reuse it for several validations."""

    pool: frozenset[str]
    in_all_langs: frozenset[str]


def corpus_facts(df: pd.DataFrame) -> CorpusFacts:
    per_lang = {lang: set(df.loc[df["lang"] == lang, "article_id"]) for lang in CORPUS_LANGS}
    return CorpusFacts(
        frozenset(benchmark_eligible(df)["article_id"]),
        frozenset(set.intersection(*per_lang.values())),
    )


def validate_items(
    files: dict[str, list[dict]],
    df: pd.DataFrame,
    *,
    freeze: bool = False,
    partial: bool = False,
    check_ranks: bool = True,
    indexes: dict[str, Bm25Index] | None = None,
    facts: CorpusFacts | None = None,
) -> Report:
    """Validate ``{"dev": [...], "test": [...]}`` against the corpus ``df``."""
    report = Report()
    pool, in_all_langs = facts if facts is not None else corpus_facts(df)

    # ---- ids, schema, file placement
    items: list[dict] = []
    seen_ids: Counter = Counter()
    for fname in SPLITS:
        previous = None
        for item in files.get(fname, []):
            label = str(item.get("id", "?")) if isinstance(item, dict) else "?"
            problems = (
                schema_problems(item) if isinstance(item, dict) else ["item must be an object"]
            )
            for p in problems:
                report.error("schema", label, p)
            if problems:
                continue
            if not _ID_RE.fullmatch(item["id"]):
                report.error("id_format", label, "id must look like v1-0001")
            if previous is not None and item["id"] <= previous:
                report.error("id_order", label, f"ids must be sorted ascending (after {previous})")
            previous = item["id"]
            seen_ids[item["id"]] += 1
            if item["split"] != fname:
                report.error("file_split", label, f"split {item['split']!r} in {fname}.jsonl")
            items.append(item)
    for item_id, n in sorted(seen_ids.items()):
        if n > 1:
            report.error("id_duplicate", item_id, f"id used {n} times")

    by_id = {it["id"]: it for it in items}
    roots: dict[str, str] = {}
    for it in items:
        roots[it["id"]] = it["translation_of"] or it["id"]
        _check_item(report, it, df, pool, in_all_langs)

    _check_translations(report, items, by_id)
    _check_gold_use(report, items, roots)
    _check_questions(report, items)
    _check_provenance_and_review(report, items, freeze)

    originals = [it for it in items if not it["translation_of"]]
    answerable = [it for it in originals if not it["is_trap"]]
    if not partial:
        _check_balance_and_quotas(report, answerable)
    if check_ranks and answerable:
        _check_ranks(report, answerable, df, indexes)
    _warnings(report, answerable, items)
    return report


def _check_item(
    report: Report, it: dict, df: pd.DataFrame, pool: set[str], in_all: set[str]
) -> None:
    item_id = it["id"]
    qtype, gold = it["qtype"], it["gold_article_ids"]
    if it["is_trap"] != (qtype == "trap"):
        report.error("is_trap", item_id, f"is_trap={it['is_trap']} but qtype={qtype!r}")
        return
    if len(it["question"]) > MAX_QUESTION_CHARS:
        report.error(
            "question_length",
            item_id,
            f"{len(it['question'])} characters (limit {MAX_QUESTION_CHARS})",
        )

    if it["is_trap"]:
        problems = []
        if gold:
            problems.append("a trap has no gold_article_ids")
        if it["gold_facts"]:
            problems.append("a trap has no gold_facts")
        if not (isinstance(it["trap_reason"], str) and it["trap_reason"].strip()):
            problems.append("a trap needs a trap_reason")
        if it["trap_type"] is None:
            problems.append("a trap needs a trap_type")
        for key in ("source_lang", "source_act", "difficulty", "lexical_overlap", "bm25_rank"):
            if it[key] is not None:
                problems.append(f"a trap has no {key}")
        for p in problems:
            report.error("trap_fields", item_id, p)
        return

    problems = []
    if it["trap_type"] is not None or it["trap_reason"] is not None:
        problems.append("only traps have trap_type and trap_reason")
    if it["source_lang"] is None:
        problems.append("source_lang is required for an answerable item")
    if it["difficulty"] is None:
        problems.append("difficulty is required for an answerable item")
    if it["bm25_rank"] is None or it["lexical_overlap"] is None:
        problems.append("bm25_rank and lexical_overlap are required for an answerable item")
    for p in problems:
        report.error("answerable_fields", item_id, p)

    expected = 1 if qtype == "single" else 2
    if len(gold) != expected or len(set(gold)) != len(gold):
        report.error(
            "gold_count", item_id, f"{qtype} needs {expected} distinct gold id(s), has {gold}"
        )
        return
    for g in gold:
        if g not in in_all:
            report.error(
                "gold_missing", item_id, f"{g} is not in the corpus in all three languages"
            )
        elif g not in pool:
            report.error("gold_ineligible", item_id, f"{g} is not benchmark-eligible")
        if split_of(g) != it["split"]:
            report.error(
                "split_rule", item_id, f"{g}: rule gives {split_of(g)!r}, item says {it['split']!r}"
            )
    if it["source_act"] != gold[0].partition(":")[0]:
        report.error(
            "source_act", item_id, f"source_act {it['source_act']!r} is not the act of {gold[0]}"
        )

    facts = it["gold_facts"]
    limit = 4 if qtype == "cross_ref" else 3
    if not 1 <= len(facts) <= limit:
        report.error("fact_count", item_id, f"{len(facts)} gold facts (1 to {limit} allowed)")
    for f in facts:
        if f["article_id"] not in gold:
            report.error(
                "fact_article", item_id, f"fact cites {f['article_id']}, not one of {gold}"
            )
        if not f["text"].strip():
            report.error("fact_count", item_id, "a gold fact has empty text")


def _check_translations(report: Report, items: list[dict], by_id: dict[str, dict]) -> None:
    for it in items:
        parent_id = it["translation_of"]
        if parent_id is None:
            continue
        parent = by_id.get(parent_id)
        if parent is None:
            report.error("translation", it["id"], f"translation_of {parent_id} is not an item")
        elif parent["translation_of"]:
            report.error("translation", it["id"], "a translation of a translation")
        else:
            if it["gold_article_ids"] != parent["gold_article_ids"]:
                report.error("translation", it["id"], "gold_article_ids differ from the original")
            if it["split"] != parent["split"]:
                report.error("translation", it["id"], "split differs from the original")
            if it["qtype"] != parent["qtype"]:
                report.error("translation", it["id"], "qtype differs from the original")
            if it["query_lang"] == parent["query_lang"]:
                report.error(
                    "translation", it["id"], "a translation must change the query language"
                )


def _check_gold_use(report: Report, items: list[dict], roots: dict[str, str]) -> None:
    users: dict[str, set[str]] = {}
    splits: dict[str, set[str]] = {}
    for it in items:
        for g in it["gold_article_ids"]:
            users.setdefault(g, set()).add(roots[it["id"]])
            splits.setdefault(g, set()).add(it["split"])
    for g in sorted(users, key=id_key):
        if len(splits[g]) > 1:
            report.error("gold_in_both_splits", g, "gold article used in dev and in test")
        if len(users[g]) > 1:
            report.error(
                "gold_reused", g, f"gold article used by several items: {sorted(users[g])}"
            )


def _check_questions(report: Report, items: list[dict]) -> None:
    exact: dict[str, str] = {}
    grams: dict[str, list[tuple[str, set[str]]]] = {}
    for it in sorted(items, key=lambda x: x["id"]):
        norm = normalise_question(it["question"])
        if norm in exact:
            report.error("question_duplicate", it["id"], f"same question text as {exact[norm]}")
            continue
        exact[norm] = it["id"]
        mine = char_ngrams(norm)
        for other_id, other in grams.setdefault(it["query_lang"], []):
            if jaccard(mine, other) > NEAR_DUPLICATE_JACCARD:
                report.error(
                    "question_near_duplicate",
                    it["id"],
                    f"near-duplicate of {other_id} (5-gram Jaccard > 0.8)",
                )
                break
        grams[it["query_lang"]].append((it["id"], mine))


def _check_provenance_and_review(report: Report, items: list[dict], freeze: bool) -> None:
    for it in items:
        item_id = it["id"]
        for p in provenance_problems(it["provenance"]):
            report.error("provenance", item_id, p)
        if it["provenance"].get("agent_file_sha256") is None and not provenance_problems(
            it["provenance"]
        ):
            report.warn("warn_agent_file", item_id, "provenance has no agent file hash")
        status, by, at = it["review_status"], it["reviewed_by"], it["reviewed_at"]
        if status == "unreviewed":
            if by or at or it["review_note"]:
                report.error("review_fields", item_id, "an unreviewed item has reviewer fields")
        elif not (isinstance(by, str) and by.strip() and isinstance(at, str) and at.strip()):
            report.error(
                "review_fields", item_id, f"status {status!r} needs reviewed_by and reviewed_at"
            )
        if freeze:
            if status not in FROZEN_STATUSES:
                report.error(
                    "freeze_status",
                    item_id,
                    f"review_status is {status!r}; a freeze needs accepted or edited",
                )
            elif isinstance(by, str) and _AGENT_RE.search(by):
                report.error("freeze_reviewer", item_id, f"reviewed_by {by!r} looks like an agent")
            elif not (isinstance(by, str) and by.strip()):
                report.error("freeze_reviewer", item_id, "reviewed_by is empty")


def _check_balance_and_quotas(report: Report, answerable: list[dict]) -> None:
    if answerable:
        langs = Counter(it["query_lang"] for it in answerable)
        for lang in QUERY_LANGS:
            share = langs[lang] / len(answerable)
            if not LANG_SHARE[0] <= share <= LANG_SHARE[1]:
                report.error(
                    "lang_balance",
                    f"query_lang={lang}",
                    f"{langs[lang]} of {len(answerable)} answerable items ({share:.1%}); allowed "
                    f"{LANG_SHARE[0]:.0%} to {LANG_SHARE[1]:.0%}",
                )
    acts = Counter(it["source_act"] for it in answerable)
    for act in sorted(set(ACT_QUOTA) | set(acts), key=lambda a: id_key(f"{a}:")):
        quota = ACT_QUOTA.get(act)
        if quota is None:
            report.error("act_quota", f"act={act}", "act is not in the quota table")
        elif abs(acts[act] - quota) > QUOTA_TOLERANCE:
            report.error(
                "act_quota",
                f"act={act}",
                f"{acts[act]} items, quota {quota} (tolerance {QUOTA_TOLERANCE})",
            )


def _check_ranks(
    report: Report, answerable: list[dict], df: pd.DataFrame, indexes: dict[str, Bm25Index] | None
) -> None:
    indexes = indexes if indexes is not None else build_indexes(df)
    for it in answerable:
        try:
            ranks, overlap = compute_ranks(
                it["question"], it["gold_article_ids"], it["query_lang"], indexes
            )
        except KeyError as exc:
            report.error("rank_mismatch", it["id"], f"cannot rank: {exc} is not in the index")
            continue
        if ranks != it["bm25_rank"] or overlap != it["lexical_overlap"]:
            report.error(
                "rank_mismatch",
                it["id"],
                f"stored bm25_rank {it['bm25_rank']} / lexical_overlap {it['lexical_overlap']}, "
                f"recomputed {ranks} / {overlap}",
            )


def _warnings(report: Report, answerable: list[dict], items: list[dict]) -> None:
    own_language = [it for it in answerable if it["query_lang"] != "en" and it["bm25_rank"]]
    if own_language:
        top = sum(1 for it in own_language if it["bm25_rank"][it["query_lang"]] == 1)
        if top / len(own_language) > SAME_LANG_RANK1_SHARE:
            report.warn(
                "warn_same_lang_rank1",
                "all",
                f"{top} of {len(own_language)} non-English items rank first in the BM25 index "
                f"of their own language ({top / len(own_language):.0%}, limit "
                f"{SAME_LANG_RANK1_SHARE:.0%})",
            )
    cells = Counter((it["query_lang"], it["source_lang"]) for it in answerable)
    for q in QUERY_LANGS:
        for s in CORPUS_LANGS:
            if cells[(q, s)] < MIN_PAIR_CELL:
                report.warn(
                    "warn_pair_cell",
                    f"query={q} source={s}",
                    f"{cells[(q, s)]} items (fewer than {MIN_PAIR_CELL})",
                )


# ------------------------------------------------------------------ files and command


def load_dir(directory: Path, report: Report) -> dict[str, list[dict]]:
    files: dict[str, list[dict]] = {}
    for split in SPLITS:
        path = directory / f"{split}.jsonl"
        if not path.exists():
            report.error("files", path.name, f"{path} does not exist")
            continue
        try:
            files[split] = read_jsonl(path)
        except ValueError as exc:
            report.error("json", path.name, str(exc))
            files[split] = []
        raw = path.read_bytes()
        if b"\r" in raw or (raw and not raw.endswith(b"\n")):
            report.error("files", path.name, "must use \\n line endings and end with a newline")
    return files


def validate_dir(directory: Path, df: pd.DataFrame, **kwargs) -> Report:
    pre = Report()
    files = load_dir(directory, pre)
    report = validate_items(files, df, **kwargs)
    report.errors = pre.errors + report.errors
    return report


def run(argv: list[str] | None = None, corpus: pd.DataFrame | None = None) -> int:
    ap = argparse.ArgumentParser(prog="benchmark validate", description=__doc__.splitlines()[0])
    ap.add_argument(
        "--dir", type=Path, default=SELECTED_DIR, help="folder with dev.jsonl and test.jsonl"
    )
    ap.add_argument("--parquet", type=Path, default=ARTICLES_PARQUET)
    ap.add_argument("--freeze", action="store_true", help="also require accepted/edited by a human")
    ap.add_argument("--partial", action="store_true", help="skip language balance and act quotas")
    ap.add_argument(
        "--skip-rank-check", action="store_true", help="do not recompute the BM25 ranks"
    )
    args = ap.parse_args(argv)
    df = corpus if corpus is not None else load_corpus(args.parquet)
    report = validate_dir(
        args.dir, df, freeze=args.freeze, partial=args.partial, check_ranks=not args.skip_rank_check
    )
    for v in report.warnings:
        print(f"WARNING {v}")
    for v in report.errors:
        print(f"ERROR   {v}")
    print(
        f"validate {args.dir}: {len(report.errors)} error(s), {len(report.warnings)} warning(s)"
        + (" (freeze)" if args.freeze else "")
    )
    return 1 if report.errors else 0
