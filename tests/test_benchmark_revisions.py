"""``apply-revisions``: swap revised candidates into a selection, reset review, never half-write."""

from __future__ import annotations

import copy
import json

import pytest

from benchmark_fixtures import make_corpus, make_valid_items
from fedlex_bench.benchmark import cli, revisions, validate
from fedlex_bench.benchmark.common import read_jsonl, write_jsonl


@pytest.fixture(scope="module")
def corpus():
    return make_corpus()


@pytest.fixture(scope="module")
def valid_items(corpus):
    return make_valid_items(corpus)


@pytest.fixture
def items(valid_items):
    out = copy.deepcopy(valid_items)
    for it in out:
        it["review_status"] = "unreviewed"
    return out


def put(folder, items):
    write_jsonl(folder / "dev.jsonl", [i for i in items if i["split"] == "dev"])
    write_jsonl(folder / "test.jsonl", [i for i in items if i["split"] == "test"])


@pytest.fixture
def selected(tmp_path, items):
    folder = tmp_path / "selected"
    put(folder, items)
    return folder


def candidate_for(item, **over):
    cand = {
        "slot_id": item["provenance"]["slot_id"],
        "question": "REVISED " + item["question"],
        "gold_facts": [
            {"text": "revised fact", "article_id": a} for a in item["gold_article_ids"][:1]
        ],
        "difficulty": "hard" if item["difficulty"] != "hard" else "easy",
        "candidate_sha256": "c" * 64,
        "bm25_rank": item["bm25_rank"],
        "lexical_overlap": 5 if item["lexical_overlap"] is not None else None,
        "gold_article_ids": list(item["gold_article_ids"]),
        "query_lang": item["query_lang"],
        "qtype": item["qtype"],
        "split": item["split"],
        "source_act": item["source_act"],
        "trap_reason": "new reason" if item["is_trap"] else None,
        "id": None,
    }
    if item["is_trap"]:
        cand["difficulty"] = None
    cand.update(over)
    return cand


def first(items, **match):
    return next(i for i in items if all(i[k] == v for k, v in match.items()))


def snapshot(folder):
    return {p.name: p.read_bytes() for p in sorted(folder.iterdir())}


def load(folder):
    return {i["id"]: i for s in ("dev", "test") for i in read_jsonl(folder / f"{s}.jsonl")}


def test_replaces_fields_and_resets_review(selected, items):
    item = first(items, is_trap=False, split="test")
    item["review_status"] = "accepted"
    item["reviewed_by"] = "Someone"
    item["reviewed_at"] = "2026-10-20T08:00:00Z"
    item["review_note"] = "fine"
    item["provenance"]["expert_flag"] = True
    put(selected, items)
    cand = candidate_for(item)
    changes = revisions.apply_revisions([cand], [cand["slot_id"]], selected)
    assert [(c.item_id, c.old_status) for c in changes] == [(item["id"], "accepted")]
    new = load(selected)
    got = new[item["id"]]
    assert got["question"] == cand["question"]
    assert got["gold_facts"] == cand["gold_facts"]
    assert got["difficulty"] == cand["difficulty"]
    assert got["lexical_overlap"] == cand["lexical_overlap"]
    assert got["review_status"] == "unreviewed"
    assert got["reviewed_by"] is got["reviewed_at"] is got["review_note"] is None
    prov = got["provenance"]
    assert prov["candidate_sha256"] == "c" * 64
    assert prov["revision"] == "machine-feedback-1"
    assert prov["expert_flag"] is True
    assert prov["pre_revision"] == {
        "question": item["question"],
        "gold_facts": item["gold_facts"],
    }
    assert all(new[i["id"]] == i for i in items if i["id"] != item["id"])


def test_trap_reason_replaced_for_traps(selected, items):
    item = first(items, is_trap=True)
    cand = candidate_for(item)
    revisions.apply_revisions([cand], [cand["slot_id"]], selected)
    got = load(selected)[item["id"]]
    assert got["trap_reason"] == "new reason"
    assert got["difficulty"] is None


def test_pre_revision_is_not_overwritten(selected, items):
    item = first(items, is_trap=False, split="dev")
    item["provenance"]["pre_revision"] = {"question": "orig", "gold_facts": []}
    item["provenance"]["revision"] = "machine-feedback-0"
    put(selected, items)
    cand = candidate_for(item)
    revisions.apply_revisions([cand], [cand["slot_id"]], selected)
    prov = load(selected)[item["id"]]["provenance"]
    assert prov["pre_revision"] == {"question": "orig", "gold_facts": []}
    assert prov["revision"] == "machine-feedback-1"


def test_already_applied_is_skipped_and_keeps_human_review(selected, items):
    item = first(items, is_trap=False, split="dev")
    cand = candidate_for(item)
    revisions.apply_revisions([cand], [cand["slot_id"]], selected)
    path = selected / "dev.jsonl"
    rows = read_jsonl(path)
    for r in rows:
        if r["id"] == item["id"]:
            r["review_status"] = "accepted"
            r["reviewed_by"] = "Francisco Mira"
    write_jsonl(path, rows)
    before = snapshot(selected)
    assert revisions.apply_revisions([cand], [cand["slot_id"]], selected) == []
    assert snapshot(selected) == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("gold_article_ids", ["999:1"]),
        ("query_lang", "xx"),
        ("qtype", "other"),
        ("split", "other"),
        ("source_act", "999"),
    ],
)
def test_mismatch_refuses_and_writes_nothing(selected, items, field, value):
    good = first(items, is_trap=False, split="dev")
    bad = first(items, is_trap=False, split="test")
    cands = [candidate_for(good), candidate_for(bad, **{field: value})]
    before = snapshot(selected)
    with pytest.raises(revisions.RevisionError, match=field):
        revisions.apply_revisions(cands, [c["slot_id"] for c in cands], selected)
    assert snapshot(selected) == before


def test_unknown_slot_refuses(selected, items):
    item = first(items, is_trap=False)
    before = snapshot(selected)
    with pytest.raises(revisions.RevisionError, match="s-nope"):
        revisions.apply_revisions([candidate_for(item)], ["s-nope"], selected)
    cand2 = candidate_for(item, slot_id="s-new")
    with pytest.raises(revisions.RevisionError, match="s-new"):
        revisions.apply_revisions([cand2], ["s-new"], selected)
    assert snapshot(selected) == before


def test_dry_run_writes_nothing_but_reports(selected, items):
    item = first(items, is_trap=False)
    cand = candidate_for(item)
    before = snapshot(selected)
    changes = revisions.apply_revisions([cand], [cand["slot_id"]], selected, dry_run=True)
    assert [c.item_id for c in changes] == [item["id"]]
    assert snapshot(selected) == before


def test_output_format_order_and_no_stray_files(selected, items):
    item = first(items, is_trap=False, split="dev")
    cand = candidate_for(item)
    revisions.apply_revisions([cand], [cand["slot_id"]], selected)
    assert sorted(p.name for p in selected.iterdir()) == ["dev.jsonl", "test.jsonl"]
    raw = (selected / "dev.jsonl").read_bytes()
    assert b"\r" not in raw
    rows = read_jsonl(selected / "dev.jsonl")
    assert [r["id"] for r in rows] == [i["id"] for i in items if i["split"] == "dev"]
    for line in raw.decode("utf-8").splitlines():
        assert json.dumps(json.loads(line), sort_keys=True, ensure_ascii=False) == line


def test_file_changed_during_run_aborts(selected, items, monkeypatch):
    item = first(items, is_trap=False, split="dev")
    cand = candidate_for(item)
    real = revisions._replace_atomic

    def racing(path, data, expected):
        path.write_bytes(path.read_bytes() + b"\n")  # the review app wrote meanwhile
        return real(path, data, expected)

    monkeypatch.setattr(revisions, "_replace_atomic", racing)
    with pytest.raises(revisions.RevisionError, match="changed"):
        revisions.apply_revisions([cand], [cand["slot_id"]], selected)


def test_result_passes_validator(selected, items, corpus):
    chosen = [first(items, is_trap=False, split="dev"), first(items, is_trap=True)]
    cands = [candidate_for(i, question=i["question"]) for i in chosen]
    revisions.apply_revisions(cands, [c["slot_id"] for c in cands], selected)
    report = validate.validate_dir(selected, corpus, check_ranks=True)
    assert report.errors == []


def test_cli_wiring(selected, items, tmp_path, capsys):
    item = first(items, is_trap=False, split="dev")
    cand = candidate_for(item)
    cfile = tmp_path / "cands.jsonl"
    write_jsonl(cfile, [cand])
    sfile = tmp_path / "slots.json"
    sfile.write_text(json.dumps([cand["slot_id"]]), encoding="utf-8")
    argv = ["apply-revisions", "--candidates", str(cfile), "--dir", str(selected)]
    argv += ["--slots-file", str(sfile)]
    before = snapshot(selected)
    assert cli.main([*argv, "--dry-run"]) == 0
    assert snapshot(selected) == before
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert item["id"] in out and "unreviewed" in out
    sfile.write_text(json.dumps(["s-nope"]), encoding="utf-8")
    assert cli.main(argv) == 1


def test_a_revision_clears_the_stale_review_basis_but_keeps_the_expert_flag(selected, items):
    item = first(items, is_trap=False, split="test")
    item["provenance"]["review_basis"] = "non-lawyer, reads English only"
    item["provenance"]["expert_flag"] = True
    put(selected, items)
    cand = candidate_for(item)
    revisions.apply_revisions([cand], [cand["slot_id"]], selected)
    prov = load(selected)[item["id"]]["provenance"]
    assert "review_basis" not in prov and prov["expert_flag"] is True
