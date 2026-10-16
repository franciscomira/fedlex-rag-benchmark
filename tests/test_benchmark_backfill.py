"""``backfill``: a flagged item is replaced by the best reserve item of the same cell."""

from __future__ import annotations

import copy

import pytest

from fedlex_bench.benchmark import backfill, cli
from fedlex_bench.benchmark.backfill import BackfillError
from fedlex_bench.benchmark.common import read_jsonl, write_jsonl


def item(
    n,
    gold,
    *,
    split="test",
    act="220",
    qtype="single",
    lang="de",
    source="fr",
    slot=None,
    status="unreviewed",
):
    return {
        "id": f"v1-{n:04d}" if n else None,
        "split": split,
        "source_act": act,
        "qtype": qtype,
        "is_trap": qtype == "trap",
        "query_lang": lang,
        "source_lang": source,
        "gold_article_ids": list(gold),
        "question": f"question {slot or n}",
        "review_status": status,
        "reviewed_by": "someone" if status != "unreviewed" else None,
        "reviewed_at": "2026-10-20" if status != "unreviewed" else None,
        "review_note": None,
        "translation_of": None,
        "provenance": {"slot_id": slot or f"s-{n:04d}"},
    }


@pytest.fixture
def selected():
    return {
        "dev": [item(1, ["220:1"], split="dev")],
        "test": [
            item(2, ["220:10"], status="accepted"),
            item(3, ["220:11"], lang="en", source="it"),
            item(4, ["210:5"], act="210"),
        ],
    }


@pytest.fixture
def reserve():
    return {
        "dev": [],
        "test": [
            item(0, ["220:20"], lang="fr", source="de", slot="s-0100"),  # same cell, far match
            item(0, ["220:21"], lang="de", source="it", slot="s-0101"),  # same query language
            item(0, ["220:22"], lang="de", source="fr", slot="s-0102"),  # best for v1-0002
            item(0, ["220:10"], lang="de", source="fr", slot="s-0103"),  # gold already used
            item(0, ["210:6"], act="210", lang="de", slot="s-0104"),  # other cell
        ],
    }


def run_core(ids, selected, reserve, **kw):
    return backfill.backfill_items(ids, selected, reserve, **kw)


def test_best_reserve_item_same_query_language_then_source_language(selected, reserve):
    sel, res, removed, reps = run_core(["v1-0002"], selected, reserve)
    new = sel["test"][0]
    assert new["id"] == "v1-0002" and new["gold_article_ids"] == ["220:22"]
    assert new["provenance"]["backfill"] == {"replaces": "v1-0002"}
    assert new["provenance"]["slot_id"] == "s-0102"
    assert new["review_status"] == "unreviewed"
    assert new["reviewed_by"] is None and new["reviewed_at"] is None
    assert [r["provenance"]["slot_id"] for r in res["test"]] == [
        "s-0100",
        "s-0101",
        "s-0103",
        "s-0104",
    ]
    assert removed["test"][0]["id"] == "v1-0002"
    assert removed["test"][0]["removal_reason"] == "expert-flag"
    assert reps[0].tier == "same query and source language"
    assert [i["id"] for i in sel["test"]] == ["v1-0002", "v1-0003", "v1-0004"]  # order kept


def test_falls_back_to_source_language_then_anything_in_the_cell(selected, reserve):
    # v1-0003 is an English question from Italian: no en reserve item, nothing in it
    sel, _, _, reps = run_core(["v1-0003"], selected, reserve)
    assert sel["test"][1]["provenance"]["slot_id"] == "s-0101"  # source it, query de
    assert reps[0].tier == "same source language"
    reserve["test"] = [r for r in reserve["test"] if r["provenance"]["slot_id"] != "s-0101"]
    sel, _, _, reps = run_core(["v1-0003"], selected, reserve)
    assert sel["test"][1]["provenance"]["slot_id"] == "s-0100"  # first in slot order
    assert reps[0].tier == "same cell only"


def test_gold_used_twin_and_flagged_reserve_items_are_skipped(selected, reserve):
    # s-0102 is the best; twin it, flag s-0101, reject s-0100: only s-0103 (gold used) remains
    reserve["test"][2]["gold_article_ids"] = ["220:99"]
    reserve["test"][1]["provenance"]["expert_flag"] = "x"
    reserve["test"][0]["review_status"] = "rejected"
    with pytest.raises(BackfillError, match="v1-0002"):
        run_core(["v1-0002"], selected, reserve, twin_articles={"220:99"})
    sel, *_ = run_core(["v1-0002"], selected, reserve)
    assert sel["test"][0]["gold_article_ids"] == ["220:99"]


def test_the_removed_items_gold_stays_blocked(selected, reserve):
    reserve["test"] = [item(0, ["220:10"], slot="s-0200")]  # same gold as the flagged item
    with pytest.raises(BackfillError):
        run_core(["v1-0002"], selected, reserve)


def test_two_ids_do_not_pick_the_same_gold_or_reserve_item(selected, reserve):
    selected["test"].append(item(5, ["220:12"], lang="de", source="fr"))
    reserve["test"].append(item(0, ["220:22"], lang="de", source="fr", slot="s-0105"))  # twin gold
    sel, res, _, reps = run_core(["v1-0002", "v1-0005"], selected, reserve)
    golds = [i["gold_article_ids"][0] for i in sel["test"]]
    assert len(golds) == len(set(golds))
    assert len(reps) == 2 and len(res["test"]) == 5 - 2 + 1


def test_unknown_id_translation_children_and_missing_replacement(selected, reserve):
    with pytest.raises(BackfillError, match="not a selected item"):
        run_core(["v1-9999"], selected, reserve)
    selected["test"].append(
        dict(item(5, ["220:10"], lang="fr"), id="v1-0002-fr", translation_of="v1-0002")
    )
    with pytest.raises(BackfillError, match="has translations"):
        run_core(["v1-0002"], selected, reserve)
    with pytest.raises(BackfillError, match="translation cannot be backfilled"):
        run_core(["v1-0002-fr"], selected, reserve)
    with pytest.raises(BackfillError, match="no usable reserve"):
        run_core(["v1-0001"], selected, reserve)  # dev reserve is empty


def test_inputs_are_not_modified_and_a_backfill_is_not_backfilled_twice(selected, reserve, capsys):
    before = copy.deepcopy((selected, reserve))
    sel, res, _, _ = run_core(["v1-0002"], selected, reserve)
    assert (selected, reserve) == before
    sel2, res2, removed2, reps2 = run_core(["v1-0002"], sel, res)
    assert reps2 == [] and sel2 == sel and res2 == res and removed2 == {"dev": [], "test": []}
    assert "already a backfill" in capsys.readouterr().out
    _, _, _, reps3 = run_core(["v1-0002"], sel, res, again=True)
    assert len(reps3) == 1


# ------------------------------------------------------------------ command


@pytest.fixture
def folders(tmp_path, selected, reserve):
    s, r = tmp_path / "selected", tmp_path / "reserve"
    for split in ("dev", "test"):
        write_jsonl(s / f"{split}.jsonl", selected[split])
        write_jsonl(r / f"{split}.jsonl", reserve[split])
    return s, r, tmp_path / "removed"


def argv_for(folders, tmp_path, *extra):
    s, r, removed = folders
    return [
        "backfill", "--dir", str(s), "--reserve", str(r), "--removed", str(removed),
        "--twins", str(tmp_path / "no-twins.jsonl"), *extra,
    ]  # fmt: skip


def test_command_writes_selected_reserve_and_removed(folders, tmp_path, capsys):
    s, r, removed = folders
    assert cli.main(argv_for(folders, tmp_path, "--ids", "v1-0002,v1-0003")) == 0
    out = capsys.readouterr().out
    assert "v1-0002" in out and "replaced 2 of 2" in out
    sel = read_jsonl(s / "test.jsonl")
    assert [i["id"] for i in sel] == ["v1-0002", "v1-0003", "v1-0004"]
    assert all(i["provenance"].get("backfill") for i in sel[:2])
    assert len(read_jsonl(r / "test.jsonl")) == 3
    gone = read_jsonl(removed / "test.jsonl")
    assert [i["id"] for i in gone] == ["v1-0002", "v1-0003"]
    assert not (removed / "dev.jsonl").exists()  # nothing removed from dev
    # again: nothing changes
    snapshot = (s / "test.jsonl").read_bytes(), (r / "test.jsonl").read_bytes()
    assert cli.main(argv_for(folders, tmp_path, "--ids", "v1-0002,v1-0003")) == 0
    assert ((s / "test.jsonl").read_bytes(), (r / "test.jsonl").read_bytes()) == snapshot
    assert len(read_jsonl(removed / "test.jsonl")) == 2


def test_command_appends_to_an_existing_removed_file_with_a_custom_reason(folders, tmp_path):
    s, r, removed = folders
    write_jsonl(removed / "test.jsonl", [{"id": "v1-0777", "removal_reason": "earlier"}])
    argv = argv_for(folders, tmp_path, "--ids", "v1-0002", "--reason", "wrong article")
    assert cli.main(argv) == 0
    gone = read_jsonl(removed / "test.jsonl")
    assert [g["id"] for g in gone] == ["v1-0777", "v1-0002"]
    assert gone[1]["removal_reason"] == "wrong article"


def test_command_dry_run_and_failures_write_nothing(folders, tmp_path, capsys):
    s, r, removed = folders
    snapshot = {p: p.read_bytes() for p in (*s.glob("*.jsonl"), *r.glob("*.jsonl"))}
    assert cli.main(argv_for(folders, tmp_path, "--ids", "v1-0002", "--dry-run")) == 0
    assert "would replace 1 of 1" in capsys.readouterr().out
    assert cli.main(argv_for(folders, tmp_path, "--ids", "v1-0001")) == 1  # no dev reserve
    assert cli.main(argv_for(folders, tmp_path, "--ids", "v1-0002,v1-0001")) == 1  # all or nothing
    assert cli.main(argv_for(folders, tmp_path, "--ids", "v1-9999")) == 1
    assert {p: p.read_bytes() for p in (*s.glob("*.jsonl"), *r.glob("*.jsonl"))} == snapshot
    assert not removed.exists()


def test_command_reads_twins_file(folders, tmp_path):
    s, r, _ = folders
    twins = tmp_path / "twins.jsonl"
    write_jsonl(twins, [{"article_id": "220:22", "jaccard": 0.8, "twin_id": "220:23"}])
    argv = argv_for(folders, tmp_path, "--ids", "v1-0002")
    argv[argv.index("--twins") + 1] = str(twins)
    assert cli.main(argv) == 0
    assert read_jsonl(s / "test.jsonl")[0]["gold_article_ids"] != ["220:22"]


def test_a_replacement_does_not_inherit_a_stale_review_basis(selected, reserve):
    stale = reserve["test"][2]
    stale["provenance"]["review_basis"] = "non-lawyer, reads English only"
    sel, _, _, _ = run_core(["v1-0002"], selected, reserve)
    new = next(i for i in sel["test"] if i["id"] == "v1-0002")
    assert new["provenance"]["slot_id"] == "s-0102"
    assert "review_basis" not in new["provenance"]
    assert new["review_status"] == "unreviewed" and new["reviewed_by"] is None
