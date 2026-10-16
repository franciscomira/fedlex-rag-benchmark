"""``freeze``: gates, outputs, checksums."""

from __future__ import annotations

import copy
import hashlib

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
