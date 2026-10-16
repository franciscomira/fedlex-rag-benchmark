"""Machine-review sidecars, the calibration sample and report, and the machine-approved gate.

Everything runs on temp copies: nothing here touches ``benchmark/selected``."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from fedlex_bench.benchmark import cli
from fedlex_bench.benchmark import machine_review as mr
from fedlex_bench.benchmark.common import read_jsonl, write_jsonl

LANG_COUNTS = {"de": 37, "en": 38, "fr": 38, "it": 37}  # answerable items per query language
TRAPS_PER_LANG = 5


def make_items() -> list[dict]:
    """150 answerable + 20 trap items, ids v1-0001.. in a mixed dev/test layout (27% dev)."""
    items = []
    n = 0
    for lang, count in LANG_COUNTS.items():
        for _ in range(count):
            n += 1
            items.append(_mk(n, lang, trap=False))
    for lang in LANG_COUNTS:
        for _ in range(TRAPS_PER_LANG):
            n += 1
            items.append(_mk(n, lang, trap=True))
    # deterministic, interleaved split: every 4th item (about a quarter) is dev
    for it in items:
        it["split"] = "dev" if int(it["id"][3:]) % 4 == 0 else "test"
    return sorted(items, key=lambda i: i["id"])


def _mk(n: int, lang: str, trap: bool, **kw) -> dict:
    item = {
        "id": f"v1-{n:04d}",
        "split": "dev",
        "is_trap": trap,
        "qtype": "trap" if trap else "single",
        "query_lang": lang,
        "review_status": "unreviewed",
        "reviewed_by": None,
        "reviewed_at": None,
        "review_note": None,
        "provenance": {},
    }
    item.update(kw)
    return item


def put(folder, items):
    folder.mkdir(parents=True, exist_ok=True)
    write_jsonl(folder / "dev.jsonl", [i for i in items if i["split"] == "dev"])
    write_jsonl(folder / "test.jsonl", [i for i in items if i["split"] == "test"])


def verdict_file(folder, name, records):
    folder.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r) + "\n" for r in records)
    (folder / name).write_bytes(text.encode("utf-8"))


def rec(item_id, verdict="approve", **kw):
    out = {"id": item_id, "verdict": verdict, "issues": [], "reason": "ok", "confidence": 0.9}
    out.update(kw)
    return out


@pytest.fixture
def items():
    return make_items()


@pytest.fixture
def selected(tmp_path, items):
    folder = tmp_path / "selected"
    put(folder, items)
    return folder


# ------------------------------------------------------------------ judge names and parsing


@pytest.mark.parametrize(
    ("name", "judge"),
    [
        ("claude-judge-1.jsonl", "claude-judge"),
        ("claude-judge-12.jsonl", "claude-judge"),
        ("a-b-c-3.jsonl", "a-b-c"),
        ("solo-0.jsonl", "solo"),
    ],
)
def test_judge_name_is_the_filename_prefix_before_the_last_dash_number(name, judge):
    assert mr.judge_of(name) == judge


@pytest.mark.parametrize("name", ["claude-judge.jsonl", "-3.jsonl", "judge-x.jsonl", "j-1.json"])
def test_a_file_name_without_a_judge_and_number_is_not_a_judge_file(name):
    assert mr.judge_of(name) is None


def test_load_merges_files_per_judge_and_ignores_other_files(tmp_path):
    d = tmp_path / "mr"
    verdict_file(d, "claude-judge-2.jsonl", [rec("v1-0003", "reject", issues=["x"])])
    verdict_file(d, "claude-judge-1.jsonl", [rec("v1-0001"), rec("v1-0002")])
    verdict_file(d, "other-1.jsonl", [rec("v1-0001", "reject")])
    (d / "calibration_sample.json").write_text("{}", encoding="utf-8")
    verdicts, errors = mr.load_verdicts(d, known_ids={"v1-0001", "v1-0002", "v1-0003"})
    assert errors == []
    assert set(verdicts) == {"v1-0001", "v1-0002", "v1-0003"}
    assert set(verdicts["v1-0001"]) == {"claude-judge", "other"}
    assert verdicts["v1-0003"]["claude-judge"]["issues"] == ["x"]
    assert mr.judges_of(verdicts) == ["claude-judge", "other"]


def test_a_missing_folder_gives_no_verdicts_and_no_error(tmp_path):
    assert mr.load_verdicts(tmp_path / "nope") == ({}, [])


def test_unknown_ids_are_errors(tmp_path):
    d = tmp_path / "mr"
    verdict_file(d, "j-1.jsonl", [rec("v1-0001"), rec("v1-9999")])
    verdicts, errors = mr.load_verdicts(d, known_ids={"v1-0001"})
    assert len(errors) == 1 and "v1-9999" in errors[0] and "j-1.jsonl" in errors[0]
    assert "v1-9999" not in verdicts


def test_a_duplicate_judge_and_id_is_an_error_even_across_files(tmp_path):
    d = tmp_path / "mr"
    verdict_file(d, "j-1.jsonl", [rec("v1-0001"), rec("v1-0001", "reject")])
    verdict_file(d, "j-2.jsonl", [rec("v1-0001")])
    verdicts, errors = mr.load_verdicts(d)
    assert len(errors) == 2 and all("duplicate" in e and "v1-0001" in e for e in errors)
    assert verdicts["v1-0001"]["j"]["verdict"] == "approve"  # the first record is kept


def test_a_different_judge_may_judge_the_same_id(tmp_path):
    d = tmp_path / "mr"
    verdict_file(d, "a-1.jsonl", [rec("v1-0001")])
    verdict_file(d, "b-1.jsonl", [rec("v1-0001", "reject")])
    verdicts, errors = mr.load_verdicts(d)
    assert errors == [] and set(verdicts["v1-0001"]) == {"a", "b"}


@pytest.mark.parametrize(
    "bad",
    [
        rec("v1-0001", "maybe"),
        rec("v1-0001", None),
        {"verdict": "approve"},
        {"id": 5, "verdict": "approve"},
        rec("v1-0001", issues="not a list"),
        rec("v1-0001", issues=[1]),
        rec("v1-0001", reason=3),
        ["not", "an", "object"],
    ],
)
def test_malformed_records_are_errors_and_not_loaded(tmp_path, bad):
    d = tmp_path / "mr"
    verdict_file(d, "j-1.jsonl", [bad])
    verdicts, errors = mr.load_verdicts(d)
    assert verdicts == {} and len(errors) == 1 and "j-1.jsonl:1" in errors[0]


def test_invalid_json_is_an_error_naming_the_line(tmp_path):
    d = tmp_path / "mr"
    d.mkdir()
    (d / "j-1.jsonl").write_bytes(
        (json.dumps(rec("v1-0001")) + "\n{broken\n" + json.dumps(rec("v1-0002")) + "\n").encode()
    )
    verdicts, errors = mr.load_verdicts(d)
    assert set(verdicts) == {"v1-0001", "v1-0002"}
    assert len(errors) == 1 and "j-1.jsonl:2" in errors[0]


def test_optional_fields_default(tmp_path):
    d = tmp_path / "mr"
    verdict_file(d, "j-1.jsonl", [{"id": "v1-0001", "verdict": "reject"}])
    verdicts, errors = mr.load_verdicts(d)
    assert errors == []
    assert verdicts["v1-0001"]["j"]["issues"] == [] and verdicts["v1-0001"]["j"]["reason"] == ""


# ------------------------------------------------------------------ machine_approved


def test_machine_approved_needs_every_required_judge_to_approve():
    v = {
        "a": {"a": rec("a", "approve"), "b": rec("a", "approve")},
        "b": {"a": rec("b", "approve"), "b": rec("b", "reject")},
        "c": {"a": rec("c", "approve")},
        "d": {},
    }
    assert mr.machine_approved("a", ["a", "b"], v)
    assert not mr.machine_approved("b", ["a", "b"], v)
    assert not mr.machine_approved("c", ["a", "b"], v)  # b has no verdict: not approved
    assert mr.machine_approved("c", ["a"], v)
    assert not mr.machine_approved("zz", ["a"], v)  # unknown item
    assert not mr.machine_approved("a", [], v)  # no required judge: nothing is approved


def test_judges_covering_all_ids():
    v = {
        "1": {"full": rec("1"), "part": rec("1")},
        "2": {"full": rec("2")},
    }
    assert mr.judges_covering(v, ["1", "2"]) == ["full"]
    assert mr.judges_covering(v, ["1"]) == ["full", "part"]
    assert mr.judges_covering(v, []) == ["full", "part"]


# ------------------------------------------------------------------ calibration sample


def test_apportion_uses_largest_remainders_with_ties_in_key_order():
    assert mr.apportion(26, {"de": 37, "en": 38, "fr": 38, "it": 37}) == {
        "de": 6,
        "en": 7,
        "fr": 7,
        "it": 6,
    }
    assert mr.apportion(4, {"a": 5, "b": 5, "c": 5, "d": 5}) == {"a": 1, "b": 1, "c": 1, "d": 1}
    assert mr.apportion(1, {"a": 1, "b": 1}) == {"a": 1, "b": 0}
    assert sum(mr.apportion(7, {"x": 3, "y": 11, "z": 2}).values()) == 7


def test_the_sample_has_4_traps_26_answerable_and_covers_every_language(items):
    ids = mr.calibration_sample(items)
    assert ids == sorted(ids) and len(set(ids)) == 30
    by_id = {i["id"]: i for i in items}
    chosen = [by_id[i] for i in ids]
    assert sum(i["is_trap"] for i in chosen) == 4
    traps = Counter(i["query_lang"] for i in chosen if i["is_trap"])
    assert traps == {"de": 1, "en": 1, "fr": 1, "it": 1}
    answerable = Counter(i["query_lang"] for i in chosen if not i["is_trap"])
    assert answerable == {"de": 6, "en": 7, "fr": 7, "it": 6}
    assert {i["split"] for i in chosen} == {"dev", "test"}


def test_the_sample_follows_the_dev_test_share_of_each_group(items):
    by_id = {i["id"]: i for i in items}
    chosen = [by_id[i] for i in mr.calibration_sample(items)]
    dev = Counter(i["is_trap"] for i in chosen if i["split"] == "dev")
    # fixture: 5 of 20 traps and 37 of 150 answerable items are dev -> 1 and 6 (largest remainder)
    assert dev == {True: 1, False: 6}


def test_a_short_test_pool_is_made_up_from_dev():
    few = [_mk(n, "de", trap=False, split="test" if n <= 2 else "dev") for n in range(1, 41)]
    few += [_mk(100 + n, "de", trap=True, split="dev") for n in range(5)]
    ids = mr.calibration_sample(few)
    assert len(ids) == 30


def test_the_sample_is_deterministic_and_independent_of_item_order(items):
    first = mr.calibration_sample(items)
    assert mr.calibration_sample(list(reversed(items))) == first
    assert mr.calibration_sample(items) == first
    assert mr.calibration_sample(items, seed=1) != first


def test_the_sample_ignores_review_state(items):
    first = mr.calibration_sample(items)
    for it in items[:40]:
        it["review_status"] = "accepted"
    assert mr.calibration_sample(items) == first


def test_the_sample_command_writes_a_sorted_file_and_is_idempotent(selected, tmp_path, capsys):
    mdir = tmp_path / "mr"
    args = ["calibration-sample", "--dir", str(selected), "--machine-dir", str(mdir)]
    assert cli.main(args) == 0
    path = mdir / "calibration_sample.json"
    first = path.read_bytes()
    doc = json.loads(first)
    assert doc["ids"] == sorted(doc["ids"]) and len(doc["ids"]) == 30
    assert doc["seed"] == 20261016 and "rule" in doc
    assert cli.main(args) == 0 and path.read_bytes() == first
    assert "30" in capsys.readouterr().out


def test_the_sample_command_refuses_to_overwrite_a_different_sample(selected, tmp_path, capsys):
    mdir = tmp_path / "mr"
    mdir.mkdir()
    path = mdir / "calibration_sample.json"
    path.write_text(json.dumps({"ids": ["v1-0001"]}), encoding="utf-8")
    args = ["calibration-sample", "--dir", str(selected), "--machine-dir", str(mdir)]
    assert cli.main(args) == 1
    assert "--force" in capsys.readouterr().out
    assert json.loads(path.read_text(encoding="utf-8")) == {"ids": ["v1-0001"]}
    assert cli.main([*args, "--force"]) == 0
    assert len(json.loads(path.read_text(encoding="utf-8"))["ids"]) == 30


def test_load_sample_reads_ids_and_a_missing_file_is_empty(selected, tmp_path):
    mdir = tmp_path / "mr"
    assert mr.load_sample(mdir) == []
    cli.main(["calibration-sample", "--dir", str(selected), "--machine-dir", str(mdir)])
    ids = mr.load_sample(mdir)
    assert len(ids) == 30 and ids == sorted(ids)


# ------------------------------------------------------------------ calibration report


def reviewed(items, ids_to_status):
    by_id = {i["id"]: i for i in items}
    for item_id, status in ids_to_status.items():
        by_id[item_id].update(review_status=status, reviewed_by="F", reviewed_at="t")


def test_the_report_compares_reviewed_sample_items_with_each_judge(items):
    sample = ["v1-0001", "v1-0002", "v1-0003", "v1-0004", "v1-0005"]
    reviewed(items, {"v1-0001": "accepted", "v1-0002": "edited", "v1-0003": "rejected"})
    reviewed(items, {"v1-0010": "accepted"})  # reviewed but outside the sample: not counted
    verdicts = {
        "v1-0001": {"a": rec("v1-0001", "approve"), "b": rec("v1-0001", "reject")},
        "v1-0002": {"a": rec("v1-0002", "reject"), "b": rec("v1-0002", "approve")},
        "v1-0003": {"a": rec("v1-0003", "reject"), "b": rec("v1-0003", "reject")},
        "v1-0004": {"a": rec("v1-0004", "approve")},  # not human-reviewed yet
        "v1-0010": {"a": rec("v1-0010", "approve")},
    }
    report = mr.calibration_report(items, verdicts, sample)
    assert report["sample_size"] == 5 and report["n_reviewed"] == 3
    a, b = report["judges"]["a"], report["judges"]["b"]
    assert a["n_compared"] == 3 and a["agreement"] == pytest.approx(2 / 3)
    assert a["confusion"] == {
        "human_approve": {"judge_approve": 1, "judge_reject": 1},
        "human_reject": {"judge_approve": 0, "judge_reject": 1},
    }
    assert a["disagreements"] == ["v1-0002"]
    assert b["agreement"] == pytest.approx(2 / 3)
    assert b["confusion"]["human_approve"] == {"judge_approve": 1, "judge_reject": 1}
    assert b["confusion"]["human_reject"] == {"judge_approve": 0, "judge_reject": 1}
    assert b["disagreements"] == ["v1-0001"]


def test_the_report_names_reviewed_items_a_judge_has_not_judged(items):
    reviewed(items, {"v1-0001": "accepted", "v1-0002": "accepted"})
    verdicts = {"v1-0001": {"a": rec("v1-0001")}}
    report = mr.calibration_report(items, verdicts, ["v1-0001", "v1-0002"])
    a = report["judges"]["a"]
    assert a["n_compared"] == 1 and a["no_verdict"] == ["v1-0002"] and a["agreement"] == 1.0


def test_with_nothing_reviewed_the_agreement_is_none_not_a_made_up_number(items):
    report = mr.calibration_report(items, {"v1-0001": {"a": rec("v1-0001")}}, ["v1-0001"])
    assert report["n_reviewed"] == 0 and report["judges"]["a"]["agreement"] is None


def test_sample_ids_missing_from_the_selection_are_reported(items):
    report = mr.calibration_report(items, {}, ["v1-0001", "v1-9999"])
    assert report["missing_from_selection"] == ["v1-9999"]


def test_the_calibration_command_prints_and_writes_the_report(selected, tmp_path, capsys):
    mdir = tmp_path / "mr"
    cli.main(["calibration-sample", "--dir", str(selected), "--machine-dir", str(mdir)])
    sample = mr.load_sample(mdir)
    items = read_jsonl(selected / "dev.jsonl") + read_jsonl(selected / "test.jsonl")
    reviewed_ids = sample[:3]
    reviewed(items, {reviewed_ids[0]: "accepted", reviewed_ids[1]: "rejected"})
    put(selected, items)
    verdict_file(mdir, "j-1.jsonl", [rec(i) for i in sample])
    capsys.readouterr()
    assert cli.main(["calibration", "--dir", str(selected), "--machine-dir", str(mdir)]) == 0
    out = capsys.readouterr().out
    assert "reviewed 2 of 30" in out and "j" in out
    doc = json.loads((mdir / "calibration_report.json").read_text(encoding="utf-8"))
    assert doc["n_reviewed"] == 2 and doc["judges"]["j"]["agreement"] == 0.5
    assert doc["judges"]["j"]["disagreements"] == [reviewed_ids[1]]


def test_the_calibration_command_needs_a_sample(selected, tmp_path, capsys):
    code = cli.main(["calibration", "--dir", str(selected), "--machine-dir", str(tmp_path / "mr")])
    assert code == 1 and "calibration-sample" in capsys.readouterr().out


# ------------------------------------------------------------------ machine-status


def test_status_counts_verdicts_and_items_without_one(items):
    verdicts = {
        items[0]["id"]: {"a": rec(items[0]["id"], "approve")},
        items[1]["id"]: {"a": rec(items[1]["id"], "reject"), "b": rec(items[1]["id"])},
    }
    status = mr.machine_status(items, verdicts)
    assert status["n_items"] == len(items)
    assert status["judges"]["a"]["approve"] == 1 and status["judges"]["a"]["reject"] == 1
    assert status["judges"]["a"]["without_verdict"] == len(items) - 2
    assert status["judges"]["b"]["approve"] == 1 and status["judges"]["b"]["reject"] == 0
    assert status["judges"]["b"]["without_verdict"] == len(items) - 1


def test_the_status_command_prints_per_judge_counts(selected, tmp_path, items, capsys):
    mdir = tmp_path / "mr"
    verdict_file(mdir, "j-1.jsonl", [rec(items[0]["id"]), rec(items[1]["id"], "reject")])
    assert cli.main(["machine-status", "--dir", str(selected), "--machine-dir", str(mdir)]) == 0
    out = capsys.readouterr().out
    assert "j" in out and "approve 1" in out and "reject 1" in out
    assert f"without verdict {len(items) - 2}" in out


def test_the_status_command_reports_file_errors(selected, tmp_path, capsys):
    mdir = tmp_path / "mr"
    verdict_file(mdir, "j-1.jsonl", [rec("v1-9999")])
    assert cli.main(["machine-status", "--dir", str(selected), "--machine-dir", str(mdir)]) == 1
    assert "v1-9999" in capsys.readouterr().out


# ------------------------------------------------------------------ the machine-acceptance gate


def split_files(items):
    return {s: [i for i in items if i["split"] == s] for s in ("dev", "test")}


def test_acceptance_needs_a_judge_that_covers_every_item(tmp_path, items):
    mdir = tmp_path / "mr"
    verdict_file(mdir, "part-1.jsonl", [rec(i["id"]) for i in items[:-1]])
    acc = mr.machine_acceptance(split_files(items), mdir)
    assert acc.judges == [] and acc.ids == set()
    assert acc.errors and "covers all" in acc.errors[0]


def test_acceptance_without_any_sidecar_is_an_error(tmp_path, items):
    acc = mr.machine_acceptance(split_files(items), tmp_path / "none")
    assert acc.errors and acc.ids == set()


def test_acceptance_requires_every_covering_judge_and_skips_partial_ones(tmp_path, items):
    mdir = tmp_path / "mr"
    ids = [i["id"] for i in items]
    verdict_file(mdir, "full-1.jsonl", [rec(i) for i in ids])
    verdict_file(
        mdir, "full2-1.jsonl", [rec(i, "reject" if i == ids[1] else "approve") for i in ids]
    )
    verdict_file(mdir, "part-1.jsonl", [rec(ids[2], "reject")])  # does not cover all: ignored
    acc = mr.machine_acceptance(split_files(items), mdir)
    assert acc.errors == [] and acc.judges == ["full", "full2"]
    assert acc.ids == set(ids) - {ids[1]}


def test_acceptance_excludes_rejected_and_expert_flagged_items(tmp_path, items):
    mdir = tmp_path / "mr"
    ids = [i["id"] for i in items]
    verdict_file(mdir, "j-1.jsonl", [rec(i) for i in ids])
    items[0]["review_status"] = "rejected"
    items[1]["provenance"]["expert_flag"] = True
    items[2].update(review_status="accepted", reviewed_by="F", reviewed_at="t")
    acc = mr.machine_acceptance(split_files(items), mdir)
    assert ids[0] not in acc.ids and ids[1] not in acc.ids
    assert ids[2] in acc.ids and ids[3] in acc.ids


def test_acceptance_propagates_file_errors(tmp_path, items):
    mdir = tmp_path / "mr"
    ids = [i["id"] for i in items]
    verdict_file(mdir, "j-1.jsonl", [rec(i) for i in ids] + [rec("v1-9999")])
    acc = mr.machine_acceptance(split_files(items), mdir)
    assert any("v1-9999" in e for e in acc.errors)


def test_human_verdict_mapping():
    assert mr.human_verdict({"review_status": "accepted"}) == "approve"
    assert mr.human_verdict({"review_status": "edited"}) == "approve"
    assert mr.human_verdict({"review_status": "rejected"}) == "reject"
    assert mr.human_verdict({"review_status": "unreviewed"}) is None
