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
    h = "h1"
    v = {
        "a": {"a": rec("a", content_sha256=h), "b": rec("a", content_sha256=h)},
        "b": {"a": rec("b", content_sha256=h), "b": rec("b", "reject", content_sha256=h)},
        "c": {"a": rec("c", content_sha256=h)},
        "d": {},
    }
    assert mr.machine_approved("a", ["a", "b"], v, h)
    assert not mr.machine_approved("b", ["a", "b"], v, h)
    assert not mr.machine_approved("c", ["a", "b"], v, h)  # b has no verdict: not approved
    assert mr.machine_approved("c", ["a"], v, h)
    assert not mr.machine_approved("zz", ["a"], v, h)  # unknown item
    assert not mr.machine_approved("a", [], v, h)  # no required judge: nothing is approved


def test_machine_approved_is_bound_to_the_item_content():
    v = {"a": {"j": rec("a", content_sha256="old")}, "b": {"j": rec("b")}}
    assert mr.machine_approved("a", ["j"], v, "old")
    assert not mr.machine_approved("a", ["j"], v, "new")  # item text changed since the verdict
    assert not mr.machine_approved("b", ["j"], v, "old")  # verdict without a hash
    assert not mr.machine_approved("a", ["j"], v, None)


def test_content_sha256_is_the_canonical_hash_of_question_and_facts():
    import hashlib

    item = {"question": "Wer h\u00e4ftet?", "gold_facts": [{"text": "x", "article_id": "220:1"}]}
    item["id"] = "z"
    want = json.dumps(
        {"question": item["question"], "gold_facts": item["gold_facts"]},
        sort_keys=True,
        ensure_ascii=False,
    )
    assert mr.content_sha256(item) == hashlib.sha256(want.encode("utf-8")).hexdigest()
    other = {**item, "id": "y", "review_status": "accepted"}
    assert mr.content_sha256(other) == mr.content_sha256(item)  # only question and facts count
    other["question"] += "?"
    assert mr.content_sha256(other) != mr.content_sha256(item)


def test_load_verdicts_keeps_the_content_hash(tmp_path):
    d = tmp_path / "mr"
    verdict_file(d, "j-1.jsonl", [rec("v1-0001", content_sha256="abc"), rec("v1-0002")])
    verdicts, errors = mr.load_verdicts(d)
    assert errors == []
    assert verdicts["v1-0001"]["j"]["content_sha256"] == "abc"
    assert verdicts["v1-0002"]["j"]["content_sha256"] is None


def test_a_non_string_content_hash_is_an_error(tmp_path):
    d = tmp_path / "mr"
    verdict_file(d, "j-1.jsonl", [rec("v1-0001", content_sha256=5)])
    verdicts, errors = mr.load_verdicts(d)
    assert verdicts == {} and any("content_sha256" in e for e in errors)


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


def brec(item, verdict="approve", **kw):
    """A verdict bound to the item's current text."""
    return rec(item["id"], verdict, content_sha256=mr.content_sha256(item), **kw)


def split_files(items):
    return {s: [i for i in items if i["split"] == s] for s in ("dev", "test")}


def test_acceptance_needs_a_judge_that_covers_every_item(tmp_path, items):
    mdir = tmp_path / "mr"
    verdict_file(mdir, "part-1.jsonl", [brec(i) for i in items[:-1]])
    acc = mr.machine_acceptance(split_files(items), mdir)
    assert acc.judges == [] and acc.ids == set()
    assert acc.errors and "covers all" in acc.errors[0]


def test_acceptance_without_any_sidecar_is_an_error(tmp_path, items):
    acc = mr.machine_acceptance(split_files(items), tmp_path / "none")
    assert acc.errors and acc.ids == set()


def test_acceptance_requires_every_covering_judge_and_skips_partial_ones(tmp_path, items):
    mdir = tmp_path / "mr"
    ids = [i["id"] for i in items]
    verdict_file(mdir, "full-1.jsonl", [brec(i) for i in items])
    verdict_file(
        mdir,
        "full2-1.jsonl",
        [brec(i, "reject" if i["id"] == ids[1] else "approve") for i in items],
    )
    verdict_file(mdir, "part-1.jsonl", [brec(items[2], "reject")])  # does not cover all: ignored
    acc = mr.machine_acceptance(split_files(items), mdir)
    assert acc.errors == [] and acc.judges == ["full", "full2"]
    assert acc.ids == set(ids) - {ids[1]}


def test_acceptance_excludes_rejected_and_expert_flagged_items(tmp_path, items):
    mdir = tmp_path / "mr"
    ids = [i["id"] for i in items]
    verdict_file(mdir, "j-1.jsonl", [brec(i) for i in items])
    items[0]["review_status"] = "rejected"
    items[1]["provenance"]["expert_flag"] = True
    items[2].update(review_status="accepted", reviewed_by="F", reviewed_at="t")
    acc = mr.machine_acceptance(split_files(items), mdir)
    assert ids[0] not in acc.ids and ids[1] not in acc.ids
    assert ids[2] in acc.ids and ids[3] in acc.ids


def test_acceptance_propagates_file_errors(tmp_path, items):
    mdir = tmp_path / "mr"
    verdict_file(mdir, "j-1.jsonl", [brec(i) for i in items] + [rec("v1-9999")])
    acc = mr.machine_acceptance(split_files(items), mdir)
    assert any("v1-9999" in e for e in acc.errors)


def test_acceptance_refuses_verdicts_not_bound_to_the_current_text(tmp_path, items):
    mdir = tmp_path / "mr"
    stale, unhashed = items[3], items[4]
    recs = [brec(i) for i in items]
    recs[3]["content_sha256"] = "0" * 64
    del recs[4]["content_sha256"]
    verdict_file(mdir, "j-1.jsonl", recs)
    acc = mr.machine_acceptance(split_files(items), mdir)
    assert stale["id"] not in acc.ids and unhashed["id"] not in acc.ids
    assert acc.unbound == sorted([stale["id"], unhashed["id"]])
    assert any(
        stale["id"] in e and unhashed["id"] in e and "content_sha256" in e for e in acc.errors
    )


def test_a_changed_item_text_unbinds_its_verdict(tmp_path, items):
    mdir = tmp_path / "mr"
    verdict_file(mdir, "j-1.jsonl", [brec(i) for i in items])
    items[0]["question"] = "A revised question?"
    acc = mr.machine_acceptance(split_files(items), mdir)
    assert acc.unbound == [items[0]["id"]] and items[0]["id"] not in acc.ids


def test_human_reviewed_and_rejected_items_need_no_binding(tmp_path, items):
    mdir = tmp_path / "mr"
    verdict_file(mdir, "j-1.jsonl", [rec(i["id"]) for i in items])  # no hashes at all
    items[0].update(review_status="accepted", reviewed_by="F", reviewed_at="t")
    items[1]["review_status"] = "rejected"
    acc = mr.machine_acceptance(split_files(items), mdir)
    assert items[0]["id"] not in acc.unbound and items[1]["id"] not in acc.unbound
    assert len(acc.unbound) == len(items) - 2


# ------------------------------------------------------------------ machine-review-bind


def facts(item):
    return [{"text": f["text"], "article_id": f["article_id"]} for f in item["gold_facts"]]


def judge_in(folder, name, items, **changes):
    """Judge input lines as the judges saw them (``changes``: {id: {field: value}})."""
    folder.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in items:
        row = {
            "id": i["id"],
            "question": i["question"],
            "gold_facts": facts(i),
            "articles": {},
            "qtype": i["qtype"],
            "query_lang": i["query_lang"],
        }
        row.update(changes.get(i["id"], {}))
        lines.append(json.dumps(row, ensure_ascii=False))
    (folder / name).write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


@pytest.fixture
def bindable(tmp_path):
    items = make_items()[:6]
    for n, it in enumerate(items):
        it["question"] = f"Frage {n} \u00e4?"
        it["gold_facts"] = [{"text": f"fact {n}", "article_id": f"220:{n + 1}"}]
    folder, mdir, aids = tmp_path / "selected", tmp_path / "mr", tmp_path / "aids"
    put(folder, items)
    verdict_file(mdir, "claude-judge-1.jsonl", [rec(i["id"]) for i in items[:3]])
    verdict_file(mdir, "claude-judge-2.jsonl", [rec(i["id"]) for i in items[3:]])
    return items, folder, mdir, aids


def bind(folder, mdir, aids, *extra):
    argv = ["--dir", str(folder), "--machine-dir", str(mdir), "--aids-dir", str(aids), *extra]
    return cli.main(["machine-review-bind", *argv])


def hashes(mdir):
    out = {}
    for path in sorted(mdir.glob("claude-judge-*.jsonl")):
        for r in read_jsonl(path):
            out[r["id"]] = r.get("content_sha256")
    return out


def sidecar_bytes(mdir):
    return {p.name: p.read_bytes() for p in sorted(mdir.glob("*.jsonl"))}


def test_bind_fills_the_hash_of_what_the_judge_saw(bindable, capsys):
    items, folder, mdir, aids = bindable
    judge_in(aids, "judge-in-1.jsonl", items)
    assert bind(folder, mdir, aids) == 0
    assert hashes(mdir) == {i["id"]: mr.content_sha256(i) for i in items}
    assert "bound 6" in capsys.readouterr().out


def test_bind_is_idempotent_and_leaves_other_fields_alone(bindable):
    items, folder, mdir, aids = bindable
    judge_in(aids, "judge-in-1.jsonl", items)
    before = {r["id"]: r for p in sorted(mdir.glob("*-?.jsonl")) for r in read_jsonl(p)}
    bind(folder, mdir, aids)
    first = sidecar_bytes(mdir)
    bind(folder, mdir, aids)
    assert sidecar_bytes(mdir) == first
    after = {r["id"]: r for p in sorted(mdir.glob("*-?.jsonl")) for r in read_jsonl(p)}
    for item_id, r in after.items():
        assert {k: v for k, v in r.items() if k != "content_sha256"} == before[item_id]


def test_bind_reports_and_skips_an_item_whose_text_changed_after_judging(bindable, capsys):
    items, folder, mdir, aids = bindable
    judge_in(aids, "judge-in-1.jsonl", items, **{items[2]["id"]: {"question": "Older text?"}})
    assert bind(folder, mdir, aids) == 0
    h = hashes(mdir)
    assert h[items[2]["id"]] is None and h[items[0]["id"]] == mr.content_sha256(items[0])
    out = capsys.readouterr().out
    assert items[2]["id"] in out and "differs" in out and "unbound 1" in out


def test_bind_a_changed_gold_fact_is_also_a_difference(bindable):
    items, folder, mdir, aids = bindable
    bad = [{"text": "other fact", "article_id": "220:1"}]
    judge_in(aids, "judge-in-1.jsonl", items, **{items[0]["id"]: {"gold_facts": bad}})
    bind(folder, mdir, aids)
    assert hashes(mdir)[items[0]["id"]] is None


def test_bind_later_judge_input_files_override_earlier_ones(bindable):
    items, folder, mdir, aids = bindable
    old = {items[0]["id"]: {"question": "first version?"}}
    judge_in(aids, "judge-in-1.jsonl", items, **old)
    judge_in(aids, "judge-in-r2-1.jsonl", items[:1], **old)
    judge_in(aids, "judge-in-r3-1.jsonl", items[:1])  # r3 sorts after r2 and 1
    judge_in(aids, "judge-in-tr-1.jsonl", items[1:2], **{items[1]["id"]: {"question": "x?"}})
    bind(folder, mdir, aids)
    h = hashes(mdir)
    assert h[items[0]["id"]] == mr.content_sha256(items[0])
    assert h[items[1]["id"]] is None  # the tr file (latest) shows other text than the item has


def test_bind_input_file_order_is_numeric_then_r2_r3_bf_tr(tmp_path):
    names = [
        "judge-in-tr-2",
        "judge-in-10",
        "judge-in-bf-1",
        "judge-in-2",
        "judge-in-r3-1",
        "judge-in-r2-1",
        "judge-in-tr-1",
        "judge-in-1",
    ]
    for n in names:
        (tmp_path / f"{n}.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "flags.jsonl").write_text("", encoding="utf-8")
    got = [p.stem for p in mr.judge_input_files(tmp_path)]
    assert got == [
        "judge-in-1",
        "judge-in-2",
        "judge-in-10",
        "judge-in-r2-1",
        "judge-in-r3-1",
        "judge-in-bf-1",
        "judge-in-tr-1",
        "judge-in-tr-2",
    ]


def test_bind_reports_items_the_judges_never_saw(bindable, capsys):
    items, folder, mdir, aids = bindable
    judge_in(aids, "judge-in-1.jsonl", items[:5])
    assert bind(folder, mdir, aids) == 0
    assert hashes(mdir)[items[5]["id"]] is None
    out = capsys.readouterr().out
    assert items[5]["id"] in out and "no judge input" in out


def test_bind_dry_run_writes_nothing(bindable):
    items, folder, mdir, aids = bindable
    judge_in(aids, "judge-in-1.jsonl", items)
    before = sidecar_bytes(mdir)
    assert bind(folder, mdir, aids, "--dry-run") == 0
    assert sidecar_bytes(mdir) == before


def test_bound_sidecars_make_the_items_machine_approved(bindable):
    items, folder, mdir, aids = bindable
    judge_in(aids, "judge-in-1.jsonl", items)
    bind(folder, mdir, aids)
    acc = mr.machine_acceptance(split_files(items), mdir)
    assert acc.errors == [] and acc.unbound == [] and acc.ids == {i["id"] for i in items}


def test_human_verdict_mapping():
    assert mr.human_verdict({"review_status": "accepted"}) == "approve"
    assert mr.human_verdict({"review_status": "edited"}) == "approve"
    assert mr.human_verdict({"review_status": "rejected"}) == "reject"
    assert mr.human_verdict({"review_status": "unreviewed"}) is None


def test_bind_a_trap_without_gold_facts_in_the_judge_input_binds_as_empty(bindable):
    items, folder, mdir, aids = bindable
    items[0].update(is_trap=True, qtype="trap", gold_facts=[])
    put(folder, items)
    judge_in(aids, "judge-in-1.jsonl", items)
    path = aids / "judge-in-1.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    del rows[0]["gold_facts"]  # traps are shown with a trap reason instead
    rows[0]["trap_reason"] = "out of area"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    bind(folder, mdir, aids)
    assert hashes(mdir)[items[0]["id"]] == mr.content_sha256(items[0])
