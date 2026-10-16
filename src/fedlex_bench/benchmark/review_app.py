"""Human review of the selected benchmark items (Streamlit app).

Launch::

    uv run --group review streamlit run src/fedlex_bench/benchmark/review_app.py

Optional folders, after ``--``: ``--dir <folder>`` and ``--parquet <file>``.

The app reads ``benchmark/selected/{dev,test}.jsonl`` and the articles parquet and writes the
review back into the same JSONL files (atomic replace, sorted keys, ``\\n`` endings, same order).
Fields written: ``review_status`` (accepted / edited / rejected), ``reviewed_by``,
``reviewed_at`` (UTC), ``review_note``, plus ``provenance.review_basis`` (the reviewer's
background, sidebar text). An edit keeps the original question and facts in
``provenance.pre_edit``. "Flag for expert" writes ``provenance.expert_flag`` (and an optional
``provenance.expert_flag_note``) and never touches ``review_status``.

Optional, git-ignored English review aids (``data/processed/review_aids``: ``aid-*.jsonl``,
``flags.jsonl``, ``blind-*.jsonl``) are shown when present; the app works without them. The
English article text is an unofficial working translation, the DE/FR/IT text is authoritative.

Machine verdicts (``benchmark/machine_review/<judge>-<n>.jsonl``, see ``machine_review.py``) are
shown under the item with their reasons, except for calibration-sample items the human has not yet
reviewed: those stay hidden until the review is saved so the human judgement is independent. The
sidebar has a "Calibration sample only" filter (plain id order) and a "calibration reviewed x/30"
counter. The app only reads the machine folder.

Everything that matters is a plain function (load, filter, :func:`apply_review`, save) so it is
tested without Streamlit. :func:`main` only wires widgets and imports Streamlit lazily.
``review_status`` is set in :func:`apply_review` and nowhere else.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Collection, Mapping
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from fedlex_bench.benchmark import machine_review
from fedlex_bench.benchmark.bm25 import Bm25Index, compute_ranks
from fedlex_bench.benchmark.common import (
    CORPUS_LANGS,
    SELECTED_DIR,
    SPLITS,
    act_key,
    dumps,
    read_jsonl,
)
from fedlex_bench.benchmark.validate import _AGENT_RE, MAX_QUESTION_CHARS
from fedlex_bench.chunk import ARTICLES_PARQUET, PROCESSED_DIR

ACTIONS = ("accept", "edit", "reject")
STATUS_OF = {"accept": "accepted", "edit": "edited", "reject": "rejected"}
STATUSES = ("unreviewed", "accepted", "edited", "rejected")
TRAP_ACT = "(traps)"
MAX_FACTS = {"single": 3, "cross_ref": 4, "trap": 0}
DEFAULT_REVIEW_BASIS = "non-lawyer, reads English only"
AIDS_DIR = PROCESSED_DIR / "review_aids"
FLAG_TEXT = {
    "blind_miss": "An independent blind solver, given only the question, did not find this article",
    "blind_partial_crossref": "The blind solver found only one of the two articles",
    "blind_not_top1": "Blind solver ranked another article first",
    "solver_unsure": "The blind solver was unsure of its answer",
    "lang_versions_differ": "German/French/Italian versions differ slightly: see notes",
    "critic_competing": "Critic named a similar article (see competing ids)",
    "revised": "Rewritten after critic feedback",
    "hard_or_crossref": "Marked hard, or needs two articles",
    "trap_solver_says_answerable": "Blind solver thinks the corpus answers this trap",
    "solver_says_unanswerable": "Blind solver says the corpus does not answer this question",
}
CHECKLIST = (
    "Does each gold fact match the highlighted supporting sentence?",
    "Does the question ask for what the facts answer, nothing more?",
    "Does it read like a real person's question, in the stated language?",
    "Does the question just repeat the article's words? (it should not)",
    "For traps: is it a plausible question that the listed acts really do not answer?",
)


class ReviewError(ValueError):
    """A review action that must not be applied; the message says why."""


# ------------------------------------------------------------------ load and save


def _path(directory: Path, split: str) -> Path:
    return directory / f"{split}.jsonl"


def load_split(directory: Path, split: str) -> list[dict]:
    return read_jsonl(_path(directory, split))


def load_all(directory: Path) -> list[dict]:
    """Items of dev then test, each file in its own order."""
    return [item for split in SPLITS for item in load_split(directory, split)]


def save_split(directory: Path, split: str, items: list[dict]) -> None:
    """Write ``items`` to ``<directory>/<split>.jsonl``: a temp file in the same folder, then an
    atomic replace. Sorted keys, UTF-8, ``\\n`` endings. A failure leaves the old file as it was."""
    target = _path(directory, split)
    data = "".join(dumps(i) + "\n" for i in items).encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{split}.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


# ------------------------------------------------------------------ review actions


def check_reviewer(name: object) -> str:
    """The stripped reviewer name. It must be non-empty and must not look like an agent (the same
    check the freeze validation applies)."""
    if not isinstance(name, str) or not name.strip():
        raise ReviewError("reviewer name is required (sidebar)")
    if _AGENT_RE.search(name):
        raise ReviewError(f"reviewer {name!r} looks like an agent: enter your own name")
    return name.strip()


def _clean_facts(facts: object) -> list[dict]:
    if not isinstance(facts, list):
        raise ReviewError("gold_facts must be a list")
    out = []
    for f in facts:
        if not isinstance(f, Mapping) or not {"article_id", "text"} <= set(f):
            raise ReviewError("each gold fact needs article_id and text")
        out.append({"article_id": f["article_id"], "text": str(f["text"]).strip()})
    return out


def _checked_edits(item: dict, edits: Mapping | None) -> tuple[str, list[dict]]:
    edits = edits or {}
    question = edits.get("question", item["question"])
    if not isinstance(question, str) or not question.strip():
        raise ReviewError("the question must not be empty")
    question = question.strip()
    if len(question) > MAX_QUESTION_CHARS:
        raise ReviewError(
            f"the question has {len(question)} characters (limit {MAX_QUESTION_CHARS})"
        )
    facts = _clean_facts(edits["gold_facts"]) if "gold_facts" in edits else item["gold_facts"]
    if item["qtype"] == "trap":
        if facts:
            raise ReviewError("a trap has no gold facts: only its question can be edited")
        return question, []
    limit = MAX_FACTS[item["qtype"]]
    if not 1 <= len(facts) <= limit:
        raise ReviewError(f"{len(facts)} gold facts (1 to {limit} allowed for {item['qtype']})")
    for f in facts:
        if f["article_id"] not in item["gold_article_ids"]:
            raise ReviewError(
                f"fact cites {f['article_id']}, not one of the gold articles "
                f"{item['gold_article_ids']}"
            )
        if not f["text"]:
            raise ReviewError("a gold fact has empty text")
    return question, facts


def apply_review(
    item: Mapping,
    action: str,
    reviewer: object,
    note: str | None = None,
    edits: Mapping | None = None,
    now: datetime | None = None,
    indexes: Mapping[str, Bm25Index] | None = None,
    review_basis: str | None = None,
) -> dict:
    """Return a copy of ``item`` with the review applied (the input is not changed).

    ``accept``: status accepted. ``reject``: status rejected, ``note`` required. ``edit``:
    ``edits`` holds a new ``question`` and/or ``gold_facts``; the status is edited and the
    original question and facts go to ``provenance.pre_edit`` (kept from the first edit). With
    ``indexes`` a changed question gets its ``bm25_rank`` and ``lexical_overlap`` recomputed.
    ``provenance.review_basis`` is set to ``review_basis`` (default
    :data:`DEFAULT_REVIEW_BASIS`). Raises :class:`ReviewError` if the action is not allowed.
    """
    if action not in ACTIONS:
        raise ReviewError(f"unknown action {action!r} (accept, edit or reject)")
    name = check_reviewer(reviewer)
    note = note.strip() if isinstance(note, str) and note.strip() else None
    out = copy.deepcopy(dict(item))
    if action == "reject" and note is None:
        raise ReviewError("a rejection needs a note")
    if action == "edit":
        question, facts = _checked_edits(out, edits)
        if question == out["question"] and facts == out["gold_facts"]:
            raise ReviewError("nothing changed: use Accept, or change the question or the facts")
        out["provenance"].setdefault(
            "pre_edit", {"question": out["question"], "gold_facts": out["gold_facts"]}
        )
        question_changed = question != out["question"]
        out["question"], out["gold_facts"] = question, facts
        if question_changed and indexes is not None and out["gold_article_ids"]:
            ranks, overlap = compute_ranks(
                question, out["gold_article_ids"], out["query_lang"], indexes
            )
            out["bm25_rank"], out["lexical_overlap"] = ranks, overlap
    when = (now or datetime.now(UTC)).astimezone(UTC)
    out["review_status"] = STATUS_OF[action]
    out["reviewed_by"] = name
    out["reviewed_at"] = when.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["review_note"] = note
    basis = review_basis.strip() if isinstance(review_basis, str) else ""
    out["provenance"]["review_basis"] = basis or DEFAULT_REVIEW_BASIS
    return out


def apply_expert_flag(item: Mapping, reviewer: object, note: str | None = None) -> dict:
    """Return a copy of ``item`` with ``provenance.expert_flag = True`` and the optional note in
    ``provenance.expert_flag_note``. The review status and fields are left exactly as they are."""
    check_reviewer(reviewer)
    out = copy.deepcopy(dict(item))
    out["provenance"]["expert_flag"] = True
    note = note.strip() if isinstance(note, str) else ""
    if note:
        out["provenance"]["expert_flag_note"] = note
    return out


def _rewrite_item(directory: Path, item_id: str, change) -> dict:
    for split in SPLITS:
        items = load_split(directory, split)
        for pos, item in enumerate(items):
            if item["id"] == item_id:
                items[pos] = change(item)
                save_split(directory, split, items)
                return items[pos]
    raise ReviewError(f"no item {item_id} in {directory}")


def flag_item(directory: Path, item_id: str, reviewer: object, note: str | None = None) -> dict:
    """Flag one item for an expert and write its file back (file read again first)."""
    return _rewrite_item(directory, item_id, lambda i: apply_expert_flag(i, reviewer, note))


def review_item(directory: Path, item_id: str, action: str, reviewer: object, **kwargs) -> dict:
    """Apply a review to one item and write its file back. The file is read again first, so a
    stale screen cannot overwrite a newer review. Nothing is written if the action is refused."""
    return _rewrite_item(directory, item_id, lambda i: apply_review(i, action, reviewer, **kwargs))


# ------------------------------------------------------------------ filters and progress


def progress(items: list[dict]) -> dict:
    counts = Counter(i["review_status"] for i in items)
    out = {s: counts.get(s, 0) for s in STATUSES}
    out["total"] = len(items)
    out["reviewed"] = len(items) - out["unreviewed"]
    return out


def acts_of(items: list[dict]) -> list[str]:
    return sorted({i["source_act"] for i in items if i["source_act"]}, key=act_key)


def filter_items(
    items: list[dict],
    status: str | None = None,
    split: str | None = None,
    act: str | None = None,
    qtype: str | None = None,
    query_lang: str | None = None,
    expert_flagged: bool = False,
    only_ids: Collection[str] | None = None,
) -> list[dict]:
    """Items matching every given filter (``None`` = no filter), in the order given. ``only_ids``
    keeps only items with those ids (the calibration sample)."""

    def keep(i: dict) -> bool:
        if status and i["review_status"] != status:
            return False
        if split and i["split"] != split:
            return False
        if act and i["source_act"] != (None if act == TRAP_ACT else act):
            return False
        if qtype and i["qtype"] != qtype:
            return False
        if expert_flagged and not is_expert_flagged(i):
            return False
        if only_ids is not None and i["id"] not in only_ids:
            return False
        return not (query_lang and i["query_lang"] != query_lang)

    return [i for i in items if keep(i)]


def is_expert_flagged(item: Mapping) -> bool:
    return bool((item.get("provenance") or {}).get("expert_flag"))


def expert_flagged_count(items: list[dict]) -> int:
    return sum(is_expert_flagged(i) for i in items)


def visible_verdicts(
    item: Mapping, verdicts: Mapping[str, Mapping[str, dict]], sample_ids: Collection[str]
) -> dict[str, dict]:
    """``{judge: record}`` (judges sorted) the page may show for ``item``. Empty for a calibration
    sample item the human has not reviewed yet, so the human judgement stays independent."""
    if item["id"] in sample_ids and item["review_status"] == "unreviewed":
        return {}
    by_judge = verdicts.get(item["id"], {})
    return {judge: by_judge[judge] for judge in sorted(by_judge)}


def calibration_progress(items: list[dict], sample_ids: Collection[str]) -> tuple[int, int]:
    """``(reviewed, total)`` of the calibration sample: ``total`` is the number of sample ids;
    reviewed means a sample item whose status is anything but unreviewed."""
    reviewed = sum(i["review_status"] != "unreviewed" for i in items if i["id"] in sample_ids)
    return reviewed, len(sample_ids)


def next_unreviewed(items: list[dict], after_id: str | None) -> str | None:
    """Id of the next unreviewed item after ``after_id`` (wrapping round), or ``None``."""
    ids = [i["id"] for i in items]
    start = ids.index(after_id) + 1 if after_id in ids else 0
    for pos in list(range(start, len(items))) + list(range(0, start)):
        if items[pos]["review_status"] == "unreviewed":
            return items[pos]["id"]
    return None


# ------------------------------------------------------------------ what the page shows


def article_texts(df: pd.DataFrame, article_ids: list[str]) -> dict[str, dict]:
    """``{article_id: {lang: {"title", "text"} or None}}`` for de, fr and it."""
    sub = df[df["article_id"].isin(article_ids)]
    rows = {(r.article_id, r.lang): r for r in sub.itertuples(index=False)}
    out: dict[str, dict] = {}
    for aid in article_ids:
        out[aid] = {
            lang: (
                {"title": rows[(aid, lang)].title, "text": rows[(aid, lang)].text}
                if (aid, lang) in rows
                else None
            )
            for lang in CORPUS_LANGS
        }
    return out


def critic_context(item: Mapping) -> dict:
    """The critic notes and competing ids of the item; keys that are missing or empty are left
    out."""
    prov = item.get("provenance") or {}
    return {k: prov[k] for k in ("critic_notes", "competing_candidates") if prov.get(k)}


# ------------------------------------------------------------------ English review aids


def _read_lenient(path: Path) -> list[dict]:
    """JSON objects (with a string ``id``) of a JSONL file; blank or malformed lines are skipped."""
    out = []
    for line in path.read_bytes().decode("utf-8").split("\n"):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("id"), str):
            out.append(obj)
    return out


def load_aids(directory: Path) -> dict[str, dict]:
    """``{item_id: aid}`` merged from ``aid-*.jsonl`` (question_en, articles, fact_support, notes),
    ``flags.jsonl`` (flags, risk) and ``blind-*.jsonl`` (stored under ``blind``). Files are read
    in sorted order. A missing folder or file gives an empty result."""
    out: dict[str, dict] = {}
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("aid-*.jsonl")):
        for row in _read_lenient(path):
            out.setdefault(row["id"], {}).update(row)
    flags_path = directory / "flags.jsonl"
    for row in _read_lenient(flags_path) if flags_path.is_file() else []:
        out.setdefault(row["id"], {}).update(
            flags=row.get("flags") or [], risk=row.get("risk") or 0
        )
    for path in sorted(directory.glob("blind-*.jsonl")):
        for row in _read_lenient(path):
            out.setdefault(row["id"], {})["blind"] = row
    return out


def sort_items(items: list[dict], aids: Mapping[str, Mapping], by_risk: bool = True) -> list[dict]:
    """Items by risk descending then id (``by_risk``), else by id. Unknown risk counts as 0."""
    if not by_risk:
        return sorted(items, key=lambda i: i["id"])
    return sorted(items, key=lambda i: (-int((aids.get(i["id"]) or {}).get("risk") or 0), i["id"]))


def flag_explanations(flags: list[str]) -> list[tuple[str, str]]:
    """``(flag, plain-English text)`` per flag; an unknown flag is explained by its own name."""
    return [(f, FLAG_TEXT.get(f, f)) for f in flags]


def fact_support_lines(item: Mapping, aid: Mapping | None) -> list[dict]:
    """One dict per gold fact: ``text``, ``article_id``, ``supported`` (True, False, or None when
    there is no aid entry) and ``support_en``."""
    by_index = {fs.get("fact_index"): fs for fs in ((aid or {}).get("fact_support") or [])}
    out = []
    for k, fact in enumerate(item["gold_facts"]):
        fs = by_index.get(k)
        out.append(
            {
                "text": fact["text"],
                "article_id": fact["article_id"],
                "supported": None if fs is None else bool(fs.get("supported")),
                "support_en": "" if fs is None else str(fs.get("support_en") or ""),
            }
        )
    return out


def english_articles(aid: Mapping | None) -> list[dict]:
    """The English renderings of the gold articles (``article_id``, ``title_en``, ``text_en``)."""
    return list((aid or {}).get("articles") or [])


# ------------------------------------------------------------------ Streamlit wiring


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="review_app")
    ap.add_argument("--dir", type=Path, default=SELECTED_DIR)
    ap.add_argument("--parquet", type=Path, default=ARTICLES_PARQUET)
    ap.add_argument("--aids", type=Path, default=AIDS_DIR)
    ap.add_argument("--machine-dir", type=Path, default=machine_review.MACHINE_DIR)
    return ap.parse_known_args(argv)[0]


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - UI wiring
    import streamlit as st

    args = _parse_args(sys.argv[1:] if argv is None else argv)
    st.set_page_config(page_title="Fedlex benchmark review", layout="wide")

    @st.cache_data
    def corpus(parquet: str) -> pd.DataFrame:
        return pd.read_parquet(parquet)

    @st.cache_resource
    def indexes(parquet: str):
        from fedlex_bench.benchmark.bm25 import build_indexes

        return build_indexes(corpus(parquet))

    @st.cache_data
    def aids_of(folder: str) -> dict[str, dict]:
        return load_aids(Path(folder))

    df = corpus(str(args.parquet))
    aids = aids_of(str(args.aids))
    items = load_all(args.dir)
    verdicts, machine_errors = machine_review.load_verdicts(
        args.machine_dir, known_ids=[i["id"] for i in items]
    )
    sample_ids = set(machine_review.load_sample(args.machine_dir))
    ss = st.session_state
    ss.setdefault("reviewer", "")
    ss.setdefault("basis", DEFAULT_REVIEW_BASIS)

    with st.sidebar:
        st.header("Reviewer")
        st.text_input("Your name", key="reviewer", help="Your own name, not an agent's.")
        st.text_input(
            "Reviewer background",
            key="basis",
            help="Saved with every review as provenance.review_basis.",
        )
        st.header("Order and filters")
        plain_order = st.checkbox(
            "Plain id order",
            help="Off: highest risk first (from the blind-solver and critic flags), then id.",
        )
        only_expert = st.checkbox("Only items flagged for expert")
        only_calibration = st.checkbox(
            "Calibration sample only",
            disabled=not sample_ids,
            help="The 30 items every machine judge is compared on; plain id order. Machine "
            "verdicts stay hidden until you have reviewed the item.",
        )
        status = st.selectbox("Status", ["all", *STATUSES])
        split = st.selectbox("Split", ["all", *SPLITS])
        act = st.selectbox("Act", ["all", *acts_of(items), TRAP_ACT])
        qtype = st.selectbox("Question type", ["all", "single", "cross_ref", "trap"])
        lang = st.selectbox("Query language", ["all", "en", "de", "fr", "it"])
        prog = progress(items)
        st.header("Progress")
        st.progress(prog["reviewed"] / prog["total"] if prog["total"] else 0.0)
        st.write(
            f"{prog['reviewed']} of {prog['total']} reviewed: {prog['accepted']} accepted, "
            f"{prog['edited']} edited, {prog['rejected']} rejected, {prog['unreviewed']} left"
        )
        st.write(f"Flagged for expert: **{expert_flagged_count(items)}**")
        if sample_ids:
            done, total = calibration_progress(items, sample_ids)
            st.write(f"Calibration reviewed: **{done}/{total}**")
        if machine_errors:
            st.warning(f"{len(machine_errors)} problem(s) in the machine review files")
        by_risk = not (plain_order or only_calibration)
        if not aids:
            st.caption("No English review aids found: showing the original texts only.")
        ordered = sort_items(items, aids, by_risk=by_risk)
        ss.setdefault("current", ordered[0]["id"] if ordered else None)
        if st.button("Next unreviewed"):
            nxt = next_unreviewed(ordered, ss["current"])
            if nxt:
                ss["current"] = nxt
            else:
                st.info("Nothing left to review.")

    with st.expander("How to review without legal training"):
        st.markdown("\n".join(f"{k}. {t}" for k, t in zip("abcde", CHECKLIST, strict=True)))
        st.markdown(
            "If you are unsure about the law, press **Flag for expert**: it needs no edit and "
            "does not change the review status."
        )

    shown = filter_items(
        ordered,
        status=None if status == "all" else status,
        split=None if split == "all" else split,
        act=None if act == "all" else act,
        qtype=None if qtype == "all" else qtype,
        query_lang=None if lang == "all" else lang,
        expert_flagged=only_expert,
        only_ids=sample_ids if only_calibration else None,
    )
    if not shown:
        st.warning("No item matches the filters.")
        return
    ids = [i["id"] for i in shown]
    pos = ids.index(ss["current"]) if ss["current"] in ids else 0
    ss["current"] = ids[pos]
    left, mid, right = st.columns([1, 4, 1])
    if left.button("Previous", disabled=pos == 0):
        ss["current"] = ids[pos - 1]
        st.rerun()
    if right.button("Next", disabled=pos == len(ids) - 1):
        ss["current"] = ids[pos + 1]
        st.rerun()
    mid.caption(f"Item {pos + 1} of {len(ids)} in the current filter")
    item = shown[pos]
    aid = aids.get(item["id"])

    st.subheader(f"{item['id']}  ({item['review_status']})")
    st.write(
        f"split **{item['split']}** | qtype **{item['qtype']}** | query language "
        f"**{item['query_lang']}** | difficulty **{item['difficulty']}** | source language "
        f"{item['source_lang']} | act {item['source_act']}"
    )
    if is_expert_flagged(item):
        note_txt = item["provenance"].get("expert_flag_note")
        st.warning("Flagged for expert" + (f": {note_txt}" if note_txt else ""))
    if aid and aid.get("flags"):
        st.markdown(f"**Why this item needs a careful look** (risk {aid.get('risk', 0)})")
        for flag, text in flag_explanations(aid["flags"]):
            st.markdown(f"- :orange[**{flag}**] {text}")

    question_en = (aid or {}).get("question_en")
    st.markdown("**Question (English)**" if question_en else "**Question**")
    st.info(question_en or item["question"])
    if question_en and item["query_lang"] != "en":
        st.caption(f"Original question ({item['query_lang']})")
        st.write(item["question"])

    if item["is_trap"]:
        st.warning(
            f"Trap type: {item['trap_type']}\n\n"
            f"{(aid or {}).get('trap_reason') or item['trap_reason']}"
        )
    else:
        st.markdown("**Gold facts, with the sentence that supports each**")
        for line in fact_support_lines(item, aid):
            tag = f"*(supported by {line['article_id']})*"
            if line["supported"] is True:
                st.success(f"{line['text']} {tag}\n\nSupporting sentence: {line['support_en']}")
            elif line["supported"] is False:
                st.error(f"{line['text']} {tag}\n\n**NOT FOUND in article**")
            else:
                st.write(f"- {line['text']}  {tag}")

    blind = (aid or {}).get("blind")
    if blind:
        st.markdown(
            f"**Gold article(s):** {', '.join(item['gold_article_ids']) or 'none (trap)'}  \n"
            f"**Blind solver answered:** {', '.join(blind.get('answer_ids') or []) or 'none'} "
            f"(answerable in corpus: {blind.get('answerable_in_corpus')}, "
            f"confidence: {blind.get('confidence')})"
        )
        if blind.get("note"):
            st.caption(f"Blind solver note: {blind['note']}")
    if (aid or {}).get("notes"):
        st.markdown(f"**Notes:** {aid['notes']}")
    if item["review_status"] != "unreviewed":
        st.caption(
            f"Reviewed by {item['reviewed_by']} at {item['reviewed_at']}"
            + (f": {item['review_note']}" if item["review_note"] else "")
        )
    ctx = critic_context(item)
    if ctx.get("critic_notes"):
        st.markdown(f"**Critic notes:** {ctx['critic_notes']}")
    if ctx.get("competing_candidates"):
        st.markdown("**Competing candidates:** " + "; ".join(ctx["competing_candidates"]))

    shown_verdicts = visible_verdicts(item, verdicts, sample_ids)
    if shown_verdicts:
        st.markdown("**Machine review** (the status above is the human's)")
        for judge, rec in shown_verdicts.items():
            line = f"{judge}: **{rec['verdict']}**" + (
                f" (confidence {rec['confidence']})" if rec.get("confidence") is not None else ""
            )
            if rec["reason"]:
                line += f". {rec['reason']}"
            if rec["issues"]:
                line += " Issues: " + "; ".join(rec["issues"])
            (st.success if rec["verdict"] == "approve" else st.error)(line)
    elif item["id"] in sample_ids and item["review_status"] == "unreviewed":
        st.caption("Calibration item: machine verdicts are shown after you have reviewed it.")

    en_articles = english_articles(aid)
    if en_articles:
        st.markdown("### Gold article(s) in English")
        st.caption("Unofficial working translation for review only, not Fedlex text.")
        for art in en_articles:
            st.markdown(f"**{art.get('article_id')}**: *{art.get('title_en', '')}*")
            st.text(art.get("text_en", ""))

    with st.expander("Original German / French / Italian article text (authoritative)"):
        for art_id, by_lang in article_texts(df, item["gold_article_ids"]).items():
            st.markdown(f"### Gold article {art_id}")
            for col, code in zip(st.columns(3), CORPUS_LANGS, strict=True):
                with col:
                    st.markdown(f"**{code.upper()}**")
                    art = by_lang[code]
                    if art is None:
                        st.error("missing in the parquet")
                    else:
                        st.markdown(f"*{art['title']}*")
                        st.text(art["text"])

    st.divider()
    note = st.text_input(
        "Note (required for Reject, optional for Flag for expert)", key=f"note-{item['id']}"
    )

    def act_on(action: str, **kw) -> None:
        try:
            review_item(
                args.dir,
                item["id"],
                action,
                ss["reviewer"],
                note=note,
                review_basis=ss["basis"],
                **kw,
            )
        except ReviewError as exc:
            st.error(str(exc))
            return
        fresh = sort_items(load_all(args.dir), aids, by_risk=by_risk)
        if only_calibration:
            fresh = filter_items(fresh, only_ids=sample_ids)
        nxt = next_unreviewed(fresh, item["id"])
        if nxt:
            ss["current"] = nxt
        st.rerun()

    a, r, f, _ = st.columns([1, 1, 1.5, 3])
    if a.button("Accept", type="primary"):
        act_on("accept")
    if r.button("Reject"):
        act_on("reject")
    if f.button("Flag for expert", help="Unsure about the law? Flag it; the status is unchanged."):
        try:
            flag_item(args.dir, item["id"], ss["reviewer"], note=note)
        except ReviewError as exc:
            st.error(str(exc))
        else:
            st.rerun()

    with st.expander("Edit"):
        with st.form(f"edit-{item['id']}"):
            question = st.text_area("Question", item["question"])
            new_facts = []
            for k, fact in enumerate(item["gold_facts"]):
                c1, c2 = st.columns([4, 1])
                text = c1.text_area(f"Fact {k + 1}", fact["text"], key=f"ft-{item['id']}-{k}")
                art_id = c2.selectbox(
                    "Article",
                    item["gold_article_ids"],
                    index=item["gold_article_ids"].index(fact["article_id"]),
                    key=f"fa-{item['id']}-{k}",
                )
                new_facts.append({"text": text, "article_id": art_id})
            if st.form_submit_button("Save as edited"):
                act_on(
                    "edit",
                    edits={"question": question, "gold_facts": new_facts},
                    indexes=indexes(str(args.parquet)),
                )


if __name__ == "__main__":  # pragma: no cover - run by `streamlit run`
    main()
