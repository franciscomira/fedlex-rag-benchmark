"""``benchmark select``: critic verdicts + candidates -> dev/test/reserve with final ids."""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from benchmark_fixtures import make_corpus, provenance, words
from fedlex_bench.benchmark import plan, select
from fedlex_bench.benchmark.common import TRAP_FINAL, read_jsonl


@pytest.fixture(scope="module")
def slots():
    return plan.build_plan(make_corpus()).slots


def candidate(slot: dict) -> dict:
    trap = slot["qtype"] == "trap"
    gold = slot["gold_article_ids"]
    return {
        "id": None,
        "slot_id": slot["slot_id"],
        "split": slot["split"],
        "qtype": slot["qtype"],
        "is_trap": trap,
        "query_lang": slot["query_lang"],
        "question": words(slot["slot_id"]) + "?",
        "source_lang": slot["source_lang"],
        "gold_article_ids": gold,
        "gold_facts": [] if trap else [{"text": "A fact.", "article_id": gold[0]}],
        "source_act": slot["act"],
        "difficulty": None if trap else "medium",
        "lexical_overlap": None if trap else 5,
        "bm25_rank": None if trap else {"de": 5, "fr": 6, "it": 7},
        "trap_type": slot["trap_type"],
        "trap_reason": "No act regulates this." if trap else None,
        "translation_of": None,
        "provenance": {
            **provenance(slot["slot_id"]),
            "critic_verdict": None,
            "critic_rounds": 0,
            "critic_notes": None,
        },
        "review_status": "unreviewed",
        "reviewed_by": None,
        "reviewed_at": None,
        "review_note": None,
    }


@pytest.fixture(scope="module")
def candidates(slots):
    return [candidate(s) for s in slots]


def verdicts_for(candidates, verdict="keep", overrides=None):
    out = {}
    for c in candidates:
        v = (overrides or {}).get(c["slot_id"], verdict)
        out[c["slot_id"]] = [
            {"slot_id": c["slot_id"], "verdict": v, "notes": f"note {c['slot_id']}"}
        ]
    return out


def run_select(candidates, verdicts, **kw):
    return select.select(copy.deepcopy(candidates), verdicts, **kw)


def cell(item):
    return (item["split"], item["source_act"])


# ------------------------------------------------------------------ selection


def test_keeping_everything_yields_the_final_mix(candidates):
    res = run_select(candidates, verdicts_for(candidates))
    selected = res.dev + res.test
    assert len(selected) == 170
    assert sum(1 for i in selected if not i["is_trap"]) == 150
    assert Counter(i["source_act"] for i in selected if not i["is_trap"])["220"] == 90
    assert Counter(i["split"] for i in selected if i["is_trap"]) == TRAP_FINAL
    assert len(res.reserve) == 258 - 170
    assert res.shortfalls == []


def test_the_first_quota_kept_slots_of_each_cell_are_selected(candidates):
    res = run_select(candidates, verdicts_for(candidates))
    by_cell: dict[tuple, list[str]] = {}
    for c in candidates:
        by_cell.setdefault(cell(c), []).append(c["slot_id"])
    chosen = {i["provenance"]["slot_id"] for i in res.dev + res.test}
    for key, slot_ids in by_cell.items():
        quota = select.cell_quota(*key)
        assert [s for s in slot_ids if s in chosen] == sorted(slot_ids)[:quota], key


def test_dropped_and_fix_verdicts_are_backfilled_from_later_slots(candidates):
    first = sorted((c for c in candidates if cell(c) == ("dev", "272")), key=lambda c: c["slot_id"])
    overrides = {first[0]["slot_id"]: "drop", first[1]["slot_id"]: "fix"}
    res = run_select(candidates, verdicts_for(candidates, overrides=overrides))
    chosen = {i["provenance"]["slot_id"] for i in res.dev + res.test}
    reserve = {i["provenance"]["slot_id"] for i in res.reserve}
    assert first[0]["slot_id"] not in chosen | reserve  # dropped: nowhere
    assert first[1]["slot_id"] not in chosen | reserve  # needs a fix first
    quota = select.cell_quota("dev", "272")
    assert [s["slot_id"] for s in first[2 : 2 + quota]] == [
        s["slot_id"] for s in first if s["slot_id"] in chosen
    ]
    assert len([i for i in res.dev if i["source_act"] == "272"]) == quota


def test_candidates_without_a_verdict_are_not_selected_and_are_counted(candidates):
    verdicts = verdicts_for(candidates)
    gone = sorted(c["slot_id"] for c in candidates if cell(c) == ("test", "241"))[0]
    del verdicts[gone]
    res = run_select(candidates, verdicts)
    chosen = {i["provenance"]["slot_id"] for i in res.dev + res.test}
    assert gone not in chosen
    assert res.cells[("test", "241")]["no_verdict"] == 1


def test_a_shortfall_is_reported_per_cell(candidates):
    kept = sorted(c["slot_id"] for c in candidates if cell(c) == ("test", "235.1"))
    overrides = {s: "drop" for s in kept[3:]}  # only 3 survive, the quota is 6
    res = run_select(candidates, verdicts_for(candidates, overrides=overrides))
    assert res.shortfalls == [{"cell": ("test", "235.1"), "quota": 6, "selected": 3, "short": 3}]
    assert len([i for i in res.test if i["source_act"] == "235.1"]) == 3


def test_trap_cells_use_5_and_15(candidates):
    res = run_select(candidates, verdicts_for(candidates))
    assert len([i for i in res.dev if i["is_trap"]]) == 5
    assert len([i for i in res.test if i["is_trap"]]) == 15
    assert select.cell_quota("dev", None) == 5 and select.cell_quota("test", None) == 15


# ------------------------------------------------------------------ ids and content


def test_ids_are_sequential_and_ordered_by_split_act_and_slot(candidates):
    res = run_select(candidates, verdicts_for(candidates))
    everything = res.dev + res.test
    assert [i["id"] for i in everything] == [f"v1-{n:04d}" for n in range(1, 171)]
    keys = [
        (i["split"], i["source_act"] is None, i["source_act"] or "", i["provenance"]["slot_id"])
        for i in everything
    ]
    acts = [k[2] for k in keys if k[0] == "dev" and not k[1]]
    assert acts[0] == "210"  # natural act order, not alphabetical accident
    assert all(i["split"] == "dev" for i in res.dev) and all(i["split"] == "test" for i in res.test)
    assert res.dev[0]["id"] == "v1-0001" and res.test[0]["id"] == f"v1-{len(res.dev) + 1:04d}"
    assert [i["is_trap"] for i in res.dev][-5:] == [
        True
    ] * 5  # traps after the acts, inside a split


def test_selected_items_get_critic_provenance_and_lose_the_top_level_slot_id(candidates):
    verdicts = verdicts_for(candidates)
    res = run_select(candidates, verdicts)
    item = res.dev[0]
    assert "slot_id" not in item and item["provenance"]["slot_id"].startswith("s-")
    prov = item["provenance"]
    assert prov["critic_verdict"] == "keep" and prov["critic_rounds"] == 1
    assert prov["critic_notes"] == f"note {prov['slot_id']}"
    assert item["review_status"] == "unreviewed"
    assert item["reviewed_by"] is None
    from fedlex_bench.benchmark.validate import ITEM_KEYS

    assert set(item) == ITEM_KEYS


def test_reserve_items_have_no_id_but_the_same_fields(candidates):
    res = run_select(candidates, verdicts_for(candidates))
    assert all(i["id"] is None and "slot_id" not in i for i in res.reserve)
    assert all(i["provenance"]["critic_verdict"] == "keep" for i in res.reserve)
    keys = [
        (i["split"], i["source_act"] is None, i["source_act"] or "", i["provenance"]["slot_id"])
        for i in res.reserve
    ]
    assert keys == sorted(keys)


def test_selection_does_not_depend_on_candidate_order(candidates):
    a = run_select(candidates, verdicts_for(candidates))
    b = run_select(list(reversed(candidates)), verdicts_for(candidates))
    assert (a.dev, a.test, a.reserve) == (b.dev, b.test, b.reserve)


def test_inputs_are_not_modified(candidates):
    before = copy.deepcopy(candidates)
    select.select(candidates, verdicts_for(candidates))
    assert candidates == before


# ------------------------------------------------------------------ verdict files


def write(dir_: Path, name: str, records: list) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    text = "".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in records)
    (dir_ / name).write_bytes(text.encode("utf-8"))


def test_load_verdicts_reads_critic_files_and_keeps_extra_keys_out(tmp_path, candidates):
    ids = [c["slot_id"] for c in candidates[:3]]
    write(
        tmp_path,
        "a.jsonl",
        [
            {
                "slot_id": ids[0],
                "verdict": "keep",
                "reason_code": "ok",
                "notes": "fine",
                "competing_ids": [],
                "fix": None,
            },
            {"slot_id": ids[1], "verdict": "drop", "notes": "bad"},
        ],
    )
    write(tmp_path, "b.jsonl", [{"slot_id": ids[2], "verdict": "fix", "notes": "rewrite"}])
    verdicts, problems = select.load_verdicts(tmp_path, set(ids))
    assert problems == []
    assert {k: v[0]["verdict"] for k, v in verdicts.items()} == {
        ids[0]: "keep",
        ids[1]: "drop",
        ids[2]: "fix",
    }


@pytest.mark.parametrize(
    ("record", "needle"),
    [
        ({"slot_id": "s-9999", "verdict": "keep", "notes": ""}, "unknown slot"),
        ({"verdict": "keep", "notes": ""}, "slot_id"),
        ({"slot_id": "S", "verdict": "maybe", "notes": ""}, "verdict"),
        ({"slot_id": "S", "notes": ""}, "verdict"),
        ({"slot_id": "S", "verdict": "keep", "notes": 5}, "notes"),
        ({"slot_id": "S", "verdict": "keep", "notes": "", "round": "one"}, "round"),
        ("{not json", "JSON"),
    ],
)
def test_bad_verdict_records_are_listed(tmp_path, candidates, record, needle):
    sid = candidates[0]["slot_id"]
    if isinstance(record, dict) and record.get("slot_id") == "S":
        record = {**record, "slot_id": sid}
    write(tmp_path, "a.jsonl", [record])
    _, problems = select.load_verdicts(tmp_path, {sid})
    assert len(problems) == 1 and needle in problems[0] and "a.jsonl:1" in problems[0]


def test_duplicate_verdicts_need_distinct_rounds(tmp_path, candidates):
    sid = candidates[0]["slot_id"]
    write(tmp_path, "a.jsonl", [{"slot_id": sid, "verdict": "fix", "notes": "x"}])
    write(tmp_path, "b.jsonl", [{"slot_id": sid, "verdict": "keep", "notes": "y"}])
    _, problems = select.load_verdicts(tmp_path, {sid})
    assert len(problems) == 1 and "duplicate" in problems[0]


def test_critic_rounds_the_latest_round_wins_and_is_counted(candidates):
    sid = sorted(c["slot_id"] for c in candidates if cell(c) == ("dev", "210"))[0]
    verdicts = verdicts_for(candidates)
    verdicts[sid] = [
        {"slot_id": sid, "verdict": "keep", "notes": "after the fix", "round": 2},
        {"slot_id": sid, "verdict": "fix", "notes": "first look", "round": 1},
    ]
    res = run_select(candidates, verdicts)
    item = next(i for i in res.dev if i["provenance"]["slot_id"] == sid)
    assert item["provenance"]["critic_verdict"] == "keep"
    assert item["provenance"]["critic_rounds"] == 2
    assert item["provenance"]["critic_notes"] == "after the fix"
    verdicts[sid][0]["verdict"] = "fix"
    verdicts[sid][1]["verdict"] = "keep"  # round 1 keep, round 2 fix: the later round decides
    res = run_select(candidates, verdicts)
    assert sid not in {i["provenance"]["slot_id"] for i in res.dev + res.reserve}


def test_missing_directory_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        select.load_verdicts(tmp_path / "nope", set())


# ------------------------------------------------------------------ the command


def prepare(tmp_path, candidates, verdicts):
    cand = tmp_path / "assembled" / "candidates.jsonl"
    cand.parent.mkdir(parents=True)
    cand.write_bytes("".join(json.dumps(c, sort_keys=True) + "\n" for c in candidates).encode())
    critic = tmp_path / "critic"
    write(critic, "all.jsonl", [r for records in verdicts.values() for r in records])
    return cand, critic


def args(tmp_path, cand, critic):
    return [
        "--candidates",
        str(cand),
        "--critic",
        str(critic),
        "--out-dir",
        str(tmp_path / "selected"),
        "--reserve-dir",
        str(tmp_path / "reserve"),
    ]


def test_command_writes_dev_test_and_reserve(tmp_path, candidates, capsys):
    cand, critic = prepare(tmp_path, candidates, verdicts_for(candidates))
    assert select.run(args(tmp_path, cand, critic)) == 0
    dev = read_jsonl(tmp_path / "selected" / "dev.jsonl")
    test = read_jsonl(tmp_path / "selected" / "test.jsonl")
    assert len(dev) + len(test) == 170 and dev[0]["id"] == "v1-0001"
    assert (
        len(read_jsonl(tmp_path / "reserve" / "dev.jsonl"))
        + len(read_jsonl(tmp_path / "reserve" / "test.jsonl"))
        == 88
    )
    raw = (tmp_path / "selected" / "dev.jsonl").read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    assert [list(r) == sorted(r) for r in dev[:3]] == [True] * 3
    out = capsys.readouterr().out
    assert "selected 170" in out and "reserve 88" in out


def test_command_is_idempotent(tmp_path, candidates):
    cand, critic = prepare(tmp_path, candidates, verdicts_for(candidates))
    select.run(args(tmp_path, cand, critic))
    first = (tmp_path / "selected" / "test.jsonl").read_bytes()
    select.run(args(tmp_path, cand, critic))
    assert (tmp_path / "selected" / "test.jsonl").read_bytes() == first


def test_command_reports_shortfall_and_the_40_percent_stop_condition(tmp_path, candidates, capsys):
    overrides = {c["slot_id"]: "drop" for c in candidates[:120]}
    cand, critic = prepare(tmp_path, candidates, verdicts_for(candidates, overrides=overrides))
    code = select.run(args(tmp_path, cand, critic))
    out = capsys.readouterr().out
    assert "STOP" in out and "40%" in out
    assert "short" in out.lower()
    assert code == 0  # a shortfall is reported, the files are still written
    assert (tmp_path / "selected" / "dev.jsonl").exists()


def test_command_with_verdict_problems_writes_nothing(tmp_path, candidates, capsys):
    cand, critic = prepare(tmp_path, candidates, verdicts_for(candidates))
    write(critic, "bad.jsonl", [{"slot_id": "s-9999", "verdict": "keep", "notes": ""}])
    assert select.run(args(tmp_path, cand, critic)) == 1
    assert not (tmp_path / "selected").exists()
    assert "unknown slot" in capsys.readouterr().out
