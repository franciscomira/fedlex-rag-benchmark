"""Human review of the selected benchmark items (Streamlit app).

Launch::

    uv run --group review streamlit run src/fedlex_bench/benchmark/review_app.py

Optional folders, after ``--``: ``--dir <folder>`` and ``--parquet <file>``.

The app reads ``benchmark/selected/{dev,test}.jsonl`` and the articles parquet and writes the
review back into the same JSONL files (atomic replace, sorted keys, ``\\n`` endings, same order).
Fields written: ``review_status`` (accepted / edited / rejected), ``reviewed_by``,
``reviewed_at`` (UTC), ``review_note``. An edit keeps the original question and facts in
``provenance.pre_edit``.

Everything that matters is a plain function (load, filter, :func:`apply_review`, save) so it is
tested without Streamlit. :func:`main` only wires widgets and imports Streamlit lazily.
``review_status`` is set in :func:`apply_review` and nowhere else.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

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
from fedlex_bench.chunk import ARTICLES_PARQUET

ACTIONS = ("accept", "edit", "reject")
STATUS_OF = {"accept": "accepted", "edit": "edited", "reject": "rejected"}
STATUSES = ("unreviewed", "accepted", "edited", "rejected")
TRAP_ACT = "(traps)"
MAX_FACTS = {"single": 3, "cross_ref": 4, "trap": 0}


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
) -> dict:
    """Return a copy of ``item`` with the review applied (the input is not changed).

    ``accept``: status accepted. ``reject``: status rejected, ``note`` required. ``edit``:
    ``edits`` holds a new ``question`` and/or ``gold_facts``; the status is edited and the
    original question and facts go to ``provenance.pre_edit`` (kept from the first edit). With
    ``indexes`` a changed question gets its ``bm25_rank`` and ``lexical_overlap`` recomputed.
    Raises :class:`ReviewError` if the action is not allowed.
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
    return out


def review_item(directory: Path, item_id: str, action: str, reviewer: object, **kwargs) -> dict:
    """Apply a review to one item and write its file back. The file is read again first, so a
    stale screen cannot overwrite a newer review. Nothing is written if the action is refused."""
    for split in SPLITS:
        items = load_split(directory, split)
        for pos, item in enumerate(items):
            if item["id"] == item_id:
                items[pos] = apply_review(item, action, reviewer, **kwargs)
                save_split(directory, split, items)
                return items[pos]
    raise ReviewError(f"no item {item_id} in {directory}")


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
) -> list[dict]:
    """Items matching every given filter (``None`` = no filter), in the order given."""

    def keep(i: dict) -> bool:
        if status and i["review_status"] != status:
            return False
        if split and i["split"] != split:
            return False
        if act and i["source_act"] != (None if act == TRAP_ACT else act):
            return False
        if qtype and i["qtype"] != qtype:
            return False
        return not (query_lang and i["query_lang"] != query_lang)

    return [i for i in items if keep(i)]


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


# ------------------------------------------------------------------ Streamlit wiring


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="review_app")
    ap.add_argument("--dir", type=Path, default=SELECTED_DIR)
    ap.add_argument("--parquet", type=Path, default=ARTICLES_PARQUET)
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

    df = corpus(str(args.parquet))
    items = load_all(args.dir)
    ss = st.session_state
    ss.setdefault("reviewer", "")
    ss.setdefault("current", items[0]["id"] if items else None)

    with st.sidebar:
        st.header("Reviewer")
        st.text_input("Your name", key="reviewer", help="Your own name, not an agent's.")
        st.header("Filters")
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
        if st.button("Next unreviewed"):
            nxt = next_unreviewed(items, ss["current"])
            if nxt:
                ss["current"] = nxt
            else:
                st.info("Nothing left to review.")

    shown = filter_items(
        items,
        status=None if status == "all" else status,
        split=None if split == "all" else split,
        act=None if act == "all" else act,
        qtype=None if qtype == "all" else qtype,
        query_lang=None if lang == "all" else lang,
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

    st.subheader(f"{item['id']}  ({item['review_status']})")
    st.write(
        f"split **{item['split']}** | qtype **{item['qtype']}** | query language "
        f"**{item['query_lang']}** | difficulty **{item['difficulty']}** | source language "
        f"{item['source_lang']} | act {item['source_act']}"
    )
    st.markdown("**Question**")
    st.info(item["question"])
    if item["is_trap"]:
        st.warning(f"Trap type: {item['trap_type']}\n\n{item['trap_reason']}")
    else:
        st.markdown("**Gold facts**")
        for f in item["gold_facts"]:
            st.write(f"- {f['text']}  *(supported by {f['article_id']})*")
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

    for aid, by_lang in article_texts(df, item["gold_article_ids"]).items():
        st.markdown(f"### Gold article {aid}")
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
    note = st.text_input("Note (required for Reject)", key=f"note-{item['id']}")

    def act_on(action: str, **kw) -> None:
        try:
            review_item(args.dir, item["id"], action, ss["reviewer"], note=note, **kw)
        except ReviewError as exc:
            st.error(str(exc))
            return
        nxt = next_unreviewed(load_all(args.dir), item["id"])
        if nxt:
            ss["current"] = nxt
        st.rerun()

    a, r, _ = st.columns([1, 1, 4])
    if a.button("Accept", type="primary"):
        act_on("accept")
    if r.button("Reject"):
        act_on("reject")

    with st.expander("Edit"):
        with st.form(f"edit-{item['id']}"):
            question = st.text_area("Question", item["question"])
            new_facts = []
            for k, f in enumerate(item["gold_facts"]):
                c1, c2 = st.columns([4, 1])
                text = c1.text_area(f"Fact {k + 1}", f["text"], key=f"ft-{item['id']}-{k}")
                aid = c2.selectbox(
                    "Article",
                    item["gold_article_ids"],
                    index=item["gold_article_ids"].index(f["article_id"]),
                    key=f"fa-{item['id']}-{k}",
                )
                new_facts.append({"text": text, "article_id": aid})
            if st.form_submit_button("Save as edited"):
                act_on(
                    "edit",
                    edits={"question": question, "gold_facts": new_facts},
                    indexes=indexes(str(args.parquet)),
                )


if __name__ == "__main__":  # pragma: no cover - run by `streamlit run`
    main()
