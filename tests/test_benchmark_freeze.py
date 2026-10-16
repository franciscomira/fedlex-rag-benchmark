"""``freeze``: gates, outputs, checksums."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from benchmark_fixtures import make_corpus, make_valid_items
from fedlex_bench.benchmark import cli, freeze
from fedlex_bench.benchmark.common import read_jsonl, write_jsonl


@pytest.fixture(scope="module")
def corpus():
    return make_corpus()


@pytest.fixture(scope="module")
def parquet(tmp_path_factory, corpus):
    path = tmp_path_factory.mktemp("corpus") / "articles.parquet"
    corpus.to_parquet(path, index=False)
    return path


@pytest.fixture(scope="module")
def valid_items(corpus):
    items = make_valid_items(corpus)
    for it in items:
        it.update(
            review_status="accepted",
            reviewed_by="Francisco Mira",
            reviewed_at="2026-10-20T08:00:00Z",
        )
    return items


@pytest.fixture
def items(valid_items):
    return copy.deepcopy(valid_items)


def put(folder, items):
    write_jsonl(folder / "dev.jsonl", [i for i in items if i["split"] == "dev"])
    write_jsonl(folder / "test.jsonl", [i for i in items if i["split"] == "test"])


@pytest.fixture
def selected(tmp_path, items):
    folder = tmp_path / "selected"
    put(folder, items)
    return folder


def run(selected, out, parquet, *extra):
    return cli.main(
        ["freeze", "--dir", str(selected), "--out", str(out), "--parquet", str(parquet), *extra]
    )


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_freeze_writes_the_four_files(selected, tmp_path, parquet, capsys):
    out = tmp_path / "v1"
    assert run(selected, out, parquet) == 0
    assert sorted(p.name for p in out.iterdir()) == [
        "SHA256SUMS",
        "dev.jsonl",
        "test.jsonl",
        "test_gold_articles.txt",
    ]
    for split in ("dev", "test"):
        assert read_jsonl(out / f"{split}.jsonl") == read_jsonl(selected / f"{split}.jsonl")
        ids = [i["id"] for i in read_jsonl(out / f"{split}.jsonl")]
        assert ids == sorted(ids)
    printed = capsys.readouterr().out
    for name in ("dev.jsonl", "test.jsonl", "test_gold_articles.txt"):
        assert sha(out / name) in printed


def test_the_gold_file_lists_the_sorted_unique_test_gold_ids(selected, tmp_path, parquet):
    out = tmp_path / "v1"
    run(selected, out, parquet)
    raw = (out / "test_gold_articles.txt").read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    lines = raw.decode("utf-8").split("\n")[:-1]
    expected = {g for it in read_jsonl(selected / "test.jsonl") for g in it["gold_article_ids"]}
    assert set(lines) == expected and len(lines) == len(expected)
    from fedlex_bench.benchmark.common import id_key

    assert lines == sorted(lines, key=id_key)
    assert not set(lines) & {
        g for it in read_jsonl(selected / "dev.jsonl") for g in it["gold_article_ids"]
    }


def test_sha256sums_has_the_three_files_sorted_with_lf(selected, tmp_path, parquet):
    out = tmp_path / "v1"
    run(selected, out, parquet)
    raw = (out / "SHA256SUMS").read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    lines = raw.decode("utf-8").split("\n")[:-1]
    assert lines == [
        f"{sha(out / name)}  {name}"
        for name in ("dev.jsonl", "test.jsonl", "test_gold_articles.txt")
    ]


def test_freeze_is_idempotent_with_force(selected, tmp_path, parquet):
    out = tmp_path / "v1"
    run(selected, out, parquet)
    first = {p.name: p.read_bytes() for p in out.iterdir()}
    assert run(selected, out, parquet, "--force") == 0
    assert {p.name: p.read_bytes() for p in out.iterdir()} == first


def test_an_existing_v1_is_not_overwritten_without_force(selected, tmp_path, parquet, capsys):
    out = tmp_path / "v1"
    out.mkdir()
    (out / "dev.jsonl").write_text("keep me\n")
    assert run(selected, out, parquet) == 1
    assert "--force" in capsys.readouterr().out
    assert (out / "dev.jsonl").read_text() == "keep me\n"
    assert [p.name for p in out.iterdir()] == ["dev.jsonl"]


def test_items_are_written_sorted_by_id_even_if_the_selection_is_not(tmp_path, items, parquet):
    folder = tmp_path / "selected"
    dev = [i for i in items if i["split"] == "dev"]
    write_jsonl(folder / "dev.jsonl", list(reversed(dev)))
    write_jsonl(folder / "test.jsonl", [i for i in items if i["split"] == "test"])
    out = tmp_path / "v1"
    # the validator insists on sorted ids, so an unsorted selection is refused, not silently fixed
    assert run(folder, out, parquet) == 1
    assert not out.exists()


def test_rejected_items_are_refused_and_the_short_cells_are_named(tmp_path, items, parquet, capsys):
    victims = [
        next(i for i in items if i["split"] == "dev" and i["source_act"] == "220"),
        next(i for i in items if i["split"] == "test" and i["is_trap"]),
    ]
    for v in victims:
        v.update(review_status="rejected", review_note="wrong")
    folder = tmp_path / "selected"
    put(folder, items)
    out = tmp_path / "v1"
    assert run(folder, out, parquet) == 1
    printed = capsys.readouterr().out
    assert "dev/220" in printed and "test/trap" in printed
    assert victims[0]["id"] in printed and victims[1]["id"] in printed
    assert "backfill" in printed.lower()
    assert not out.exists()


def test_unreviewed_items_are_refused_by_the_freeze_validation(tmp_path, items, parquet, capsys):
    items[0].update(review_status="unreviewed", reviewed_by=None, reviewed_at=None)
    folder = tmp_path / "selected"
    put(folder, items)
    out = tmp_path / "v1"
    assert run(folder, out, parquet) == 1
    assert "freeze_status" in capsys.readouterr().out
    assert not out.exists()


def test_an_agent_reviewer_is_refused(tmp_path, items, parquet, capsys):
    items[0]["reviewed_by"] = "claude"
    folder = tmp_path / "selected"
    put(folder, items)
    out = tmp_path / "v1"
    assert run(folder, out, parquet) == 1
    assert "freeze_reviewer" in capsys.readouterr().out
    assert not out.exists()


def test_edited_items_are_frozen_like_accepted_ones(tmp_path, items, parquet):
    items[1]["review_status"] = "edited"
    folder = tmp_path / "selected"
    put(folder, items)
    out = tmp_path / "v1"
    assert run(folder, out, parquet) == 0
    statuses = {i["review_status"] for s in ("dev", "test") for i in read_jsonl(out / f"{s}.jsonl")}
    assert statuses == {"accepted", "edited"}


def test_short_cells_helper_counts_per_cell(items):
    rejected = [i for i in items if i["split"] == "dev" and i["source_act"] == "220"][:2]
    for r in rejected:
        r["review_status"] = "rejected"
    cells = freeze.short_cells(items)
    assert cells["dev/220"]["rejected"] == [r["id"] for r in rejected]
    assert cells["dev/220"]["kept"] + 2 == cells["dev/220"]["quota"]
    assert set(cells) == {"dev/220"}


# ------------------------------------------------------------------ --accept-machine-approved


def unreview(item):
    item.update(review_status="unreviewed", reviewed_by=None, reviewed_at=None, review_note=None)


def write_verdicts(mdir, name, items, reject=()):
    mdir.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {"id": i["id"], "verdict": "reject" if i["id"] in reject else "approve", "issues": []}
        )
        for i in items
    ]
    (mdir / name).write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def run_machine(selected, out, parquet, mdir, *extra):
    return run(
        selected, out, parquet, "--accept-machine-approved", "--machine-dir", str(mdir), *extra
    )


@pytest.fixture
def mixed(tmp_path, items):
    """Four items without a human review, the rest accepted by the human; one full judge."""
    machine = [items[0], items[1], items[-2], items[-1]]
    for it in machine:
        unreview(it)
    folder = tmp_path / "selected"
    put(folder, items)
    mdir = tmp_path / "mr"
    write_verdicts(mdir, "j-1.jsonl", items)
    return folder, mdir, items, machine


def test_without_the_option_unreviewed_items_still_stop_the_freeze(mixed, tmp_path, parquet):
    folder, mdir, _, _ = mixed
    out = tmp_path / "v1"
    assert run(folder, out, parquet) == 1 and not out.exists()


def test_machine_approved_items_are_frozen_and_counted_honestly(mixed, tmp_path, parquet, capsys):
    folder, mdir, items, machine = mixed
    out = tmp_path / "v1"
    assert run_machine(folder, out, parquet, mdir) == 0
    printed = capsys.readouterr().out
    assert f"human-reviewed {len(items) - 4}" in printed and "machine-approved only 4" in printed
    frozen = {i["id"] for s in ("dev", "test") for i in read_jsonl(out / f"{s}.jsonl")}
    assert frozen == {i["id"] for i in items}
    cov = json.loads((out / "REVIEW_COVERAGE.json").read_text(encoding="utf-8"))
    assert cov["human_reviewed"] == len(items) - 4 and cov["machine_approved_only"] == 4
    assert cov["total"] == len(items) and cov["judges"] == ["j"]
    want = {s: sorted(i["id"] for i in machine if i["split"] == s) for s in ("dev", "test")}
    assert cov["machine_only_ids"] == want
    for s in ("dev", "test"):
        assert cov["by_split"][s]["machine_approved_only"] == len(want[s])
        assert cov["by_split"][s]["human_reviewed"] + len(want[s]) == len(
            read_jsonl(out / f"{s}.jsonl")
        )
    # the items are written unchanged: their review_status is still the human's (unreviewed)
    by_id = {i["id"]: i for s in ("dev", "test") for i in read_jsonl(out / f"{s}.jsonl")}
    assert all(by_id[m["id"]]["review_status"] == "unreviewed" for m in machine)


def test_an_item_with_both_counts_as_human(mixed, tmp_path, parquet):
    folder, mdir, items, machine = mixed  # judge j approves every item, accepted ones included
    out = tmp_path / "v1"
    run_machine(folder, out, parquet, mdir)
    cov = json.loads((out / "REVIEW_COVERAGE.json").read_text(encoding="utf-8"))
    accepted = [i for i in items if i["review_status"] == "accepted"]
    assert cov["human_reviewed"] == len(accepted)
    assert not set(sum(cov["machine_only_ids"].values(), [])) & {i["id"] for i in accepted}


def test_the_coverage_file_is_checksummed_and_the_freeze_is_idempotent(mixed, tmp_path, parquet):
    folder, mdir, _, _ = mixed
    out = tmp_path / "v1"
    run_machine(folder, out, parquet, mdir)
    sums = (out / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert f"{sha(out / 'REVIEW_COVERAGE.json')}  REVIEW_COVERAGE.json" in sums
    first = {p.name: p.read_bytes() for p in out.iterdir()}
    assert run_machine(folder, out, parquet, mdir, "--force") == 0
    assert {p.name: p.read_bytes() for p in out.iterdir()} == first


def test_a_machine_rejected_item_is_not_frozen(mixed, tmp_path, parquet, capsys):
    folder, mdir, items, machine = mixed
    write_verdicts(mdir, "j-1.jsonl", items, reject={machine[0]["id"]})
    out = tmp_path / "v1"
    assert run_machine(folder, out, parquet, mdir) == 1
    printed = capsys.readouterr().out
    assert "freeze_status" in printed and machine[0]["id"] in printed
    assert not out.exists()


def test_every_covering_judge_must_approve(mixed, tmp_path, parquet):
    folder, mdir, items, machine = mixed
    write_verdicts(mdir, "k-1.jsonl", items, reject={machine[1]["id"]})
    assert run_machine(folder, tmp_path / "v1", parquet, mdir) == 1


def test_a_judge_that_does_not_cover_all_items_is_ignored(mixed, tmp_path, parquet):
    folder, mdir, items, machine = mixed
    write_verdicts(mdir, "part-1.jsonl", machine[:1], reject={machine[0]["id"]})
    assert run_machine(folder, tmp_path / "v1", parquet, mdir) == 0


def test_an_expert_flagged_unreviewed_item_is_not_machine_accepted(mixed, tmp_path, parquet):
    folder, mdir, items, machine = mixed
    machine[0]["provenance"]["expert_flag"] = True
    put(folder, items)
    out = tmp_path / "v1"
    assert run_machine(folder, out, parquet, mdir) == 1 and not out.exists()


def test_an_expert_flagged_item_the_human_accepted_is_still_frozen(mixed, tmp_path, parquet):
    folder, mdir, items, _ = mixed
    accepted = next(i for i in items if i["review_status"] == "accepted")
    accepted["provenance"]["expert_flag"] = True
    put(folder, items)
    assert run_machine(folder, tmp_path / "v1", parquet, mdir) == 0


def test_a_human_rejection_beats_machine_approval(mixed, tmp_path, parquet, capsys):
    folder, mdir, items, machine = mixed
    machine[0].update(review_status="rejected", reviewed_by="F", reviewed_at="t", review_note="x")
    put(folder, items)
    out = tmp_path / "v1"
    assert run_machine(folder, out, parquet, mdir) == 1
    assert "backfill" in capsys.readouterr().out.lower() and not out.exists()


def test_without_a_judge_covering_all_items_the_option_refuses(mixed, tmp_path, parquet, capsys):
    folder, _, items, _ = mixed
    empty = tmp_path / "empty"
    out = tmp_path / "v1"
    assert run_machine(folder, out, parquet, empty) == 1
    assert "covers all" in capsys.readouterr().out and not out.exists()


def test_malformed_verdict_files_refuse_the_freeze(mixed, tmp_path, parquet, capsys):
    folder, mdir, items, _ = mixed
    with (mdir / "j-1.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"id": "v1-9999", "verdict": "approve"}) + "\n")
    out = tmp_path / "v1"
    assert run_machine(folder, out, parquet, mdir) == 1
    assert "v1-9999" in capsys.readouterr().out and not out.exists()


def test_validate_freeze_takes_the_same_option(mixed, tmp_path, parquet, capsys):
    folder, mdir, _, _ = mixed
    base = ["validate", "--dir", str(folder), "--parquet", str(parquet), "--freeze"]
    assert cli.main(base) == 1
    assert "freeze_status" in capsys.readouterr().out
    code = cli.main([*base, "--accept-machine-approved", "--machine-dir", str(mdir)])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "machine-approved only 4" in out
