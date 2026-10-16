"""Review app logic (no Streamlit): load, filter, apply a review action, save atomically."""

from __future__ import annotations

import copy
import os
from datetime import UTC, datetime

import pytest

from benchmark_fixtures import make_corpus, provenance
from fedlex_bench.benchmark import review_app as ra
from fedlex_bench.benchmark.bm25 import build_indexes
from fedlex_bench.benchmark.common import dumps, read_jsonl, write_jsonl

NOW = datetime(2026, 10, 20, 9, 30, 15, tzinfo=UTC)


@pytest.fixture(scope="module")
def corpus():
    return make_corpus()


def _item(i: int, split="dev", qtype="single", lang="de", act="220", **kw) -> dict:
    gold = [f"{act}:{i}"] if qtype == "single" else [f"{act}:{i}", f"{act}:{i + 1}"]
    facts = [{"article_id": g, "text": f"Fact {g}."} for g in gold]
    item = {
        "id": f"v1-{i:04d}",
        "split": split,
        "qtype": qtype,
        "is_trap": False,
        "query_lang": lang,
        "question": f"Question number {i} about something?",
        "source_lang": "de",
        "gold_article_ids": gold,
        "gold_facts": facts,
        "source_act": act,
        "difficulty": "medium",
        "lexical_overlap": None,
        "bm25_rank": {"de": 5, "fr": None, "it": 7},
        "trap_type": None,
        "trap_reason": None,
        "translation_of": None,
        "provenance": provenance(f"s-{i:04d}"),
        "review_status": "unreviewed",
        "reviewed_by": None,
        "reviewed_at": None,
        "review_note": None,
    }
    item.update(kw)
    return item


def _trap(i: int, split="dev", lang="en") -> dict:
    return _item(
        i,
        split=split,
        qtype="trap",
        lang=lang,
        is_trap=True,
        source_lang=None,
        gold_article_ids=[],
        gold_facts=[],
        source_act=None,
        difficulty=None,
        bm25_rank=None,
        trap_type="out_of_area",
        trap_reason="No act regulates this.",
    )


@pytest.fixture
def folder(tmp_path):
    dev = [_item(1), _item(2, qtype="cross_ref", lang="fr"), _trap(3)]
    test = [_item(4, split="test", lang="it", act="210"), _item(5, split="test", lang="en")]
    write_jsonl(tmp_path / "dev.jsonl", dev)
    write_jsonl(tmp_path / "test.jsonl", test)
    return tmp_path


# ------------------------------------------------------------------ load / save


def test_load_keeps_the_file_order(folder):
    items = ra.load_all(folder)
    assert [i["id"] for i in items] == [f"v1-000{n}" for n in (1, 2, 3, 4, 5)]
    assert ra.load_split(folder, "test")[0]["id"] == "v1-0004"


def test_save_is_atomic_sorted_lf_and_leaves_no_temp_file(folder):
    items = ra.load_split(folder, "dev")
    items[0]["question"] = "Changed?"
    ra.save_split(folder, "dev", items)
    raw = (folder / "dev.jsonl").read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    assert raw.decode("utf-8") == "".join(dumps(i) + "\n" for i in items)
    assert sorted(p.name for p in folder.iterdir()) == ["dev.jsonl", "test.jsonl"]


def test_failed_replace_keeps_the_original_file(folder, monkeypatch):
    before = (folder / "dev.jsonl").read_bytes()

    def boom(src, dst):
        raise OSError("disk gone")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        ra.save_split(folder, "dev", ra.load_split(folder, "dev"))
    monkeypatch.undo()
    assert (folder / "dev.jsonl").read_bytes() == before
    assert sorted(p.name for p in folder.iterdir()) == ["dev.jsonl", "test.jsonl"]


# ------------------------------------------------------------------ accept / reject


def test_accept_sets_the_review_fields_and_does_not_mutate_the_input():
    item = _item(1)
    snapshot = copy.deepcopy(item)
    out = ra.apply_review(item, "accept", "Francisco Mira", now=NOW)
    assert item == snapshot
    assert out["review_status"] == "accepted"
    assert out["reviewed_by"] == "Francisco Mira"
    assert out["reviewed_at"] == "2026-10-20T09:30:15Z"
    assert out["review_note"] is None
    assert out["question"] == item["question"]
    assert "pre_edit" not in out["provenance"]


def test_accept_keeps_an_optional_note():
    out = ra.apply_review(_item(1), "accept", "Francisco", note="  fine  ", now=NOW)
    assert out["review_note"] == "fine"


def test_reject_needs_a_note():
    with pytest.raises(ra.ReviewError, match="note"):
        ra.apply_review(_item(1), "reject", "Francisco", now=NOW)
    with pytest.raises(ra.ReviewError, match="note"):
        ra.apply_review(_item(1), "reject", "Francisco", note="   ", now=NOW)
    out = ra.apply_review(_item(1), "reject", "Francisco", note="two answers", now=NOW)
    assert out["review_status"] == "rejected" and out["review_note"] == "two answers"


@pytest.mark.parametrize("name", ["", "   ", None])
def test_reviewer_name_is_required(name):
    with pytest.raises(ra.ReviewError, match="reviewer"):
        ra.apply_review(_item(1), "accept", name, now=NOW)


@pytest.mark.parametrize("name", ["claude", "Opus 4", "question-critic", "GPT", "my AI", "bot"])
def test_reviewer_name_must_not_look_like_an_agent(name):
    with pytest.raises(ra.ReviewError, match="agent"):
        ra.apply_review(_item(1), "accept", name, now=NOW)


def test_unknown_action():
    with pytest.raises(ra.ReviewError, match="action"):
        ra.apply_review(_item(1), "approve", "Francisco", now=NOW)


def test_a_rejected_item_can_be_accepted_later():
    rejected = ra.apply_review(_item(1), "reject", "Francisco", note="x", now=NOW)
    accepted = ra.apply_review(rejected, "accept", "Francisco", now=NOW)
    assert accepted["review_status"] == "accepted" and accepted["review_note"] is None


# ------------------------------------------------------------------ edit


def test_edit_changes_question_and_facts_and_keeps_the_original_in_pre_edit():
    item = _item(1)
    edits = {
        "question": "  A better question?  ",
        "gold_facts": [{"article_id": "220:1", "text": " New fact. "}],
    }
    out = ra.apply_review(item, "edit", "Francisco", edits=edits, now=NOW)
    assert out["review_status"] == "edited"
    assert out["question"] == "A better question?"
    assert out["gold_facts"] == [{"article_id": "220:1", "text": "New fact."}]
    assert out["provenance"]["pre_edit"] == {
        "question": item["question"],
        "gold_facts": item["gold_facts"],
    }
    assert out["gold_article_ids"] == item["gold_article_ids"]


def test_second_edit_keeps_the_first_original():
    item = _item(1)
    first = ra.apply_review(item, "edit", "F", edits={"question": "Second?"}, now=NOW)
    again = ra.apply_review(first, "edit", "F", edits={"question": "Third?"}, now=NOW)
    assert again["question"] == "Third?"
    assert again["provenance"]["pre_edit"]["question"] == item["question"]


def test_edit_without_a_change_is_refused():
    item = _item(1)
    with pytest.raises(ra.ReviewError, match="nothing changed"):
        ra.apply_review(item, "edit", "F", edits={"question": item["question"]}, now=NOW)
    with pytest.raises(ra.ReviewError, match="nothing changed"):
        ra.apply_review(item, "edit", "F", edits={}, now=NOW)
    with pytest.raises(ra.ReviewError, match="nothing changed"):
        ra.apply_review(item, "edit", "F", edits=None, now=NOW)


@pytest.mark.parametrize(
    ("edits", "match"),
    [
        ({"question": "  "}, "question"),
        ({"question": "x" * 401}, "400"),
        ({"gold_facts": []}, "1 to 3"),
        (
            {"gold_facts": [{"article_id": "220:1", "text": f"f{k}"} for k in range(4)]},
            "1 to 3",
        ),
        ({"gold_facts": [{"article_id": "220:99", "text": "f"}]}, "gold"),
        ({"gold_facts": [{"article_id": "220:1", "text": " "}]}, "empty"),
    ],
)
def test_invalid_edits_are_refused(edits, match):
    with pytest.raises(ra.ReviewError, match=match):
        ra.apply_review(_item(1), "edit", "F", edits=edits, now=NOW)


def test_cross_ref_allows_four_facts():
    item = _item(2, qtype="cross_ref")
    facts = [{"article_id": "220:2", "text": f"f{k}"} for k in range(4)]
    out = ra.apply_review(item, "edit", "F", edits={"gold_facts": facts}, now=NOW)
    assert len(out["gold_facts"]) == 4


def test_a_trap_can_only_change_its_question():
    trap = _trap(3)
    out = ra.apply_review(trap, "edit", "F", edits={"question": "Other?"}, now=NOW)
    assert out["question"] == "Other?" and out["gold_facts"] == []
    assert out["provenance"]["pre_edit"] == {"question": trap["question"], "gold_facts": []}
    with pytest.raises(ra.ReviewError, match="trap"):
        ra.apply_review(
            trap,
            "edit",
            "F",
            edits={"gold_facts": [{"article_id": "220:1", "text": "x"}]},
            now=NOW,
        )


def test_editing_the_question_recomputes_the_ranks_when_indexes_are_given(corpus):
    indexes = build_indexes(corpus)
    item = _item(1, lang="de", bm25_rank={"de": 999, "fr": 999, "it": 999}, lexical_overlap=999)
    out = ra.apply_review(
        item, "edit", "F", edits={"question": "Titel 1 ganz anders?"}, now=NOW, indexes=indexes
    )
    assert out["bm25_rank"] != item["bm25_rank"]
    assert out["lexical_overlap"] == out["bm25_rank"]["de"]
    # editing only the facts leaves the ranks alone
    same = ra.apply_review(
        item,
        "edit",
        "F",
        edits={"gold_facts": [{"article_id": "220:1", "text": "other"}]},
        now=NOW,
        indexes=indexes,
    )
    assert same["bm25_rank"] == item["bm25_rank"]


def test_an_english_question_has_no_lexical_overlap_after_an_edit(corpus):
    indexes = build_indexes(corpus)
    item = _item(1, lang="en")
    out = ra.apply_review(
        item, "edit", "F", edits={"question": "Titel 1 other?"}, now=NOW, indexes=indexes
    )
    assert out["lexical_overlap"] is None


# ------------------------------------------------------------------ review_item (disk)


def test_review_item_rewrites_only_that_item_and_keeps_the_order(folder):
    before = (folder / "dev.jsonl").read_text(encoding="utf-8").split("\n")
    out = ra.review_item(folder, "v1-0002", "accept", "Francisco", now=NOW)
    assert out["review_status"] == "accepted"
    after = (folder / "dev.jsonl").read_text(encoding="utf-8").split("\n")
    assert after[0] == before[0] and after[2] == before[2]
    assert after[1] != before[1]
    assert [i["id"] for i in read_jsonl(folder / "dev.jsonl")] == ["v1-0001", "v1-0002", "v1-0003"]


def test_review_item_unknown_id(folder):
    with pytest.raises(ra.ReviewError, match="v1-9999"):
        ra.review_item(folder, "v1-9999", "accept", "Francisco", now=NOW)


def test_a_refused_action_writes_nothing(folder):
    before = (folder / "test.jsonl").read_bytes()
    with pytest.raises(ra.ReviewError):
        ra.review_item(folder, "v1-0004", "reject", "Francisco", now=NOW)
    assert (folder / "test.jsonl").read_bytes() == before


# ------------------------------------------------------------------ filters and progress


def test_filters_and_progress(folder):
    ra.review_item(folder, "v1-0001", "accept", "F", now=NOW)
    ra.review_item(folder, "v1-0004", "reject", "F", note="n", now=NOW)
    items = ra.load_all(folder)
    assert ra.progress(items) == {
        "unreviewed": 3,
        "accepted": 1,
        "edited": 0,
        "rejected": 1,
        "total": 5,
        "reviewed": 2,
    }
    ids = lambda **kw: [i["id"] for i in ra.filter_items(items, **kw)]  # noqa: E731
    assert ids() == ["v1-0001", "v1-0002", "v1-0003", "v1-0004", "v1-0005"]
    assert ids(status="unreviewed") == ["v1-0002", "v1-0003", "v1-0005"]
    assert ids(split="test") == ["v1-0004", "v1-0005"]
    assert ids(act="210") == ["v1-0004"]
    assert ids(qtype="trap") == ["v1-0003"]
    assert ids(query_lang="en") == ["v1-0003", "v1-0005"]
    assert ids(split="dev", qtype="single", status="accepted") == ["v1-0001"]


def test_traps_have_no_act_so_the_act_filter_has_a_trap_option(folder):
    items = ra.load_all(folder)
    assert ra.acts_of(items) == ["210", "220"]
    assert [i["id"] for i in ra.filter_items(items, act=ra.TRAP_ACT)] == ["v1-0003"]


def test_next_unreviewed_wraps_and_skips_reviewed(folder):
    ra.review_item(folder, "v1-0002", "accept", "F", now=NOW)
    items = ra.load_all(folder)
    assert ra.next_unreviewed(items, after_id="v1-0001") == "v1-0003"
    assert ra.next_unreviewed(items, after_id="v1-0005") == "v1-0001"
    assert ra.next_unreviewed(items, after_id=None) == "v1-0001"
    done = [dict(i, review_status="accepted") for i in items]
    assert ra.next_unreviewed(done, after_id="v1-0001") is None


# ------------------------------------------------------------------ article text


def test_article_texts_gives_all_three_languages(corpus):
    texts = ra.article_texts(corpus, ["220:7", "220:8"])
    assert list(texts) == ["220:7", "220:8"]
    assert set(texts["220:7"]) == {"de", "fr", "it"}
    row = corpus[(corpus.article_id == "220:7") & (corpus.lang == "fr")].iloc[0]
    assert texts["220:7"]["fr"] == {"title": row["title"], "text": row["text"]}


def test_article_texts_missing_article_is_reported_not_dropped(corpus):
    texts = ra.article_texts(corpus, ["220:99999"])
    assert texts["220:99999"] == {"de": None, "fr": None, "it": None}


def test_critic_context_omits_what_is_missing():
    item = _item(1)
    item["provenance"]["critic_notes"] = "ok"
    item["provenance"]["competing_candidates"] = ["220:9 may also answer"]
    assert ra.critic_context(item) == {
        "critic_notes": "ok",
        "competing_candidates": ["220:9 may also answer"],
    }
    item["provenance"].pop("competing_candidates")
    item["provenance"]["critic_notes"] = None
    assert ra.critic_context(item) == {}


# ------------------------------------------------------------------ app module


def test_the_app_module_imports_without_streamlit_and_has_a_main():
    assert callable(ra.main)
    assert not hasattr(ra, "st")
