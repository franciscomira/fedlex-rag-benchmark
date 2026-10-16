"""``benchmark select``: critic verdicts + candidates -> dev/test/reserve with final ids."""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from benchmark_fixtures import make_corpus, provenance, words
from fedlex_bench.benchmark import plan, select
from fedlex_bench.benchmark.assemble import candidate_hash
from fedlex_bench.benchmark.common import TRAP_FINAL, read_jsonl


@pytest.fixture(scope="module")
def slots():
    return plan.build_plan(make_corpus()).slots


def candidate(slot: dict) -> dict:
    trap = slot["qtype"] == "trap"
    gold = slot["gold_article_ids"]
    question = words(slot["slot_id"]) + "?"
    facts = [] if trap else [{"text": "A fact.", "article_id": gold[0]}]
    return {
        "id": None,
        "slot_id": slot["slot_id"],
        "candidate_sha256": candidate_hash(question, facts),
        "split": slot["split"],
        "qtype": slot["qtype"],
        "is_trap": trap,
        "query_lang": slot["query_lang"],
        "question": question,
        "source_lang": slot["source_lang"],
        "gold_article_ids": gold,
        "gold_facts": facts,
        "source_act": slot["act"],
        "difficulty": None if trap else "medium",
        "lexical_overlap": None,
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


def verdict(c, v="keep", **extra):
    return {
        "slot_id": c["slot_id"],
        "verdict": v,
        "notes": f"note {c['slot_id']}",
        "candidate_sha256": c["candidate_sha256"],
        **extra,
    }


def verdicts_for(candidates, verdict_="keep", overrides=None):
    return {
        c["slot_id"]: [verdict(c, (overrides or {}).get(c["slot_id"], verdict_))]
        for c in candidates
    }


def run_select(candidates, verdicts, slots):
    return select.select(copy.deepcopy(candidates), verdicts, slots)


def cell(item):
    return (item["split"], item["source_act"])


def plan_cell(slots, split, act):
    return sorted(
        (s for s in slots if (s["split"], s["act"]) == (split, act)), key=lambda s: s["slot_id"]
    )


# ------------------------------------------------------------------ selection


def test_keeping_everything_yields_the_final_mix(candidates, slots):
    res = run_select(candidates, verdicts_for(candidates), slots)
    selected = res.dev + res.test
    assert len(selected) == 170
    assert sum(1 for i in selected if not i["is_trap"]) == 150
    assert Counter(i["source_act"] for i in selected if not i["is_trap"])["220"] == 90
    assert Counter(i["split"] for i in selected if i["is_trap"]) == TRAP_FINAL
    assert len(res.reserve) == 258 - 170
    assert res.shortfalls == [] and res.backfills == []


def test_the_first_quota_slots_of_each_cell_are_selected_when_all_are_kept(candidates, slots):
    res = run_select(candidates, verdicts_for(candidates), slots)
    chosen = {i["provenance"]["slot_id"] for i in res.dev + res.test}
    for key in {(c["split"], c["source_act"]) for c in candidates}:
        ids = [s["slot_id"] for s in plan_cell(slots, *key)]
        assert [s for s in ids if s in chosen] == ids[: select.cell_quota(*key)], key


def test_a_dropped_prefix_slot_is_replaced_by_the_best_matching_kept_surplus_slot(
    candidates, slots
):
    cell_slots = plan_cell(slots, "test", "272")
    quota = select.cell_quota("test", "272")
    prefix, surplus = cell_slots[:quota], cell_slots[quota:]
    victim = prefix[0]
    twin = next(
        (
            s
            for s in surplus
            if (s["query_lang"], s["qtype"]) == (victim["query_lang"], victim["qtype"])
        ),
        None,
    )
    res = run_select(
        candidates, verdicts_for(candidates, overrides={victim["slot_id"]: "drop"}), slots
    )
    (b,) = [b for b in res.backfills if b["dropped"] == victim["slot_id"]]
    chosen = {i["provenance"]["slot_id"] for i in res.test}
    assert victim["slot_id"] not in chosen and b["replacement"] in chosen
    assert b["replacement"] in {s["slot_id"] for s in surplus}
    if twin is not None:
        replacement = next(s for s in surplus if s["slot_id"] == b["replacement"])
        assert (replacement["query_lang"], replacement["qtype"]) == (
            victim["query_lang"],
            victim["qtype"],
        )
        assert b["tier"] in (1, 2)


def test_backfill_tiers_prefer_language_and_type_then_source_then_anything(slots):
    missing = {"query_lang": "de", "qtype": "single", "source_lang": "fr", "slot_id": "m"}

    def opt(slot_id, lang, qtype, src):
        return {"slot_id": slot_id, "query_lang": lang, "qtype": qtype, "source_lang": src}

    exact = opt("a", "de", "single", "fr")
    lang_type = opt("b", "de", "single", "it")
    source = opt("c", "fr", "cross_ref", "fr")
    anything = opt("d", "en", "cross_ref", "it")
    assert [select._tier(missing, o) for o in (exact, lang_type, source, anything)] == [1, 2, 3, 4]


def test_backfill_uses_tier_two_before_tier_three_and_the_earliest_slot(candidates, slots):
    cell_slots = plan_cell(slots, "dev", "220")
    quota = select.cell_quota("dev", "220")
    victim = cell_slots[0]
    surplus = cell_slots[quota:]
    res = run_select(
        candidates, verdicts_for(candidates, overrides={victim["slot_id"]: "fix"}), slots
    )
    (b,) = [b for b in res.backfills if b["dropped"] == victim["slot_id"]]
    options = sorted(
        surplus,
        key=lambda o: (select._tier(victim, o), o["slot_id"]),
    )
    assert b["replacement"] == options[0]["slot_id"]
    assert b["tier"] == select._tier(victim, options[0])


def test_a_slot_the_writer_skipped_or_never_drafted_is_backfilled_too(candidates, slots):
    cell_slots = plan_cell(slots, "dev", "210")
    victim = cell_slots[0]
    fewer = [c for c in candidates if c["slot_id"] != victim["slot_id"]]
    res = run_select(fewer, verdicts_for(fewer), slots)
    assert len([i for i in res.dev if i["source_act"] == "210"]) == select.cell_quota("dev", "210")
    assert [b["dropped"] for b in res.backfills] == [victim["slot_id"]]


def test_candidates_without_a_verdict_are_not_selected_and_are_counted(candidates, slots):
    verdicts = verdicts_for(candidates)
    gone = plan_cell(slots, "test", "241")[0]["slot_id"]
    del verdicts[gone]
    res = run_select(candidates, verdicts, slots)
    chosen = {i["provenance"]["slot_id"] for i in res.dev + res.test}
    assert gone not in chosen
    assert res.cells[("test", "241")]["no_verdict"] == 1
    assert res.cells[("test", "241")]["backfilled"] == 1


def test_a_shortfall_is_reported_per_cell(candidates, slots):
    ids = [s["slot_id"] for s in plan_cell(slots, "test", "235.1")]
    overrides = {s: "drop" for s in ids[3:]}  # only 3 survive, the quota is 6
    res = run_select(candidates, verdicts_for(candidates, overrides=overrides), slots)
    assert res.shortfalls == [{"cell": ("test", "235.1"), "quota": 6, "selected": 3, "short": 3}]
    assert len([i for i in res.test if i["source_act"] == "235.1"]) == 3


def test_trap_cells_use_5_and_15(candidates, slots):
    res = run_select(candidates, verdicts_for(candidates), slots)
    assert len([i for i in res.dev if i["is_trap"]]) == 5
    assert len([i for i in res.test if i["is_trap"]]) == 15
    assert select.cell_quota("dev", None) == 5 and select.cell_quota("test", None) == 15


def test_a_dropped_trap_is_backfilled_with_a_trap(candidates, slots):
    victim = plan_cell(slots, "test", None)[0]
    res = run_select(
        candidates, verdicts_for(candidates, overrides={victim["slot_id"]: "drop"}), slots
    )
    (b,) = [b for b in res.backfills if b["dropped"] == victim["slot_id"]]
    replacement = next(c for c in candidates if c["slot_id"] == b["replacement"])
    assert replacement["is_trap"]
    assert len([i for i in res.test if i["is_trap"]]) == 15


# ------------------------------------------------------------------ ids and content


def test_ids_are_sequential_and_ordered_by_split_act_and_slot(candidates, slots):
    res = run_select(candidates, verdicts_for(candidates), slots)
    everything = res.dev + res.test
    assert [i["id"] for i in everything] == [f"v1-{n:04d}" for n in range(1, 171)]
    keys = [
        (i["split"], i["source_act"] is None, i["source_act"] or "", i["provenance"]["slot_id"])
        for i in everything
    ]
    assert keys == sorted(keys, key=lambda k: (k[0], k[1], select.act_key(k[2] or None), k[3]))
    assert [i["source_act"] for i in res.dev if not i["is_trap"]][0] == "210"
    assert all(i["split"] == "dev" for i in res.dev) and all(i["split"] == "test" for i in res.test)
    assert res.dev[0]["id"] == "v1-0001" and res.test[0]["id"] == f"v1-{len(res.dev) + 1:04d}"
    assert [i["is_trap"] for i in res.dev][-5:] == [True] * 5


def test_selected_items_get_critic_provenance_and_lose_the_top_level_fields(candidates, slots):
    res = run_select(candidates, verdicts_for(candidates), slots)
    item = res.dev[0]
    assert "slot_id" not in item and "candidate_sha256" not in item
    prov = item["provenance"]
    assert prov["slot_id"].startswith("s-")
    assert prov["critic_verdict"] == "keep" and prov["critic_rounds"] == 1
    assert prov["critic_notes"] == f"note {prov['slot_id']}"
    assert prov["candidate_sha256"] == candidate_hash(item["question"], item["gold_facts"])
    assert item["review_status"] == "unreviewed" and item["reviewed_by"] is None
    from fedlex_bench.benchmark.validate import ITEM_KEYS

    assert set(item) == ITEM_KEYS


def test_reserve_items_have_no_id_but_the_same_fields(candidates, slots):
    res = run_select(candidates, verdicts_for(candidates), slots)
    assert all(i["id"] is None and "slot_id" not in i for i in res.reserve)
    assert all(i["provenance"]["critic_verdict"] == "keep" for i in res.reserve)
    keys = [
        (i["split"], select.act_key(i["source_act"]), i["provenance"]["slot_id"])
        for i in res.reserve
    ]
    assert keys == sorted(keys)


def test_selection_does_not_depend_on_candidate_order(candidates, slots):
    a = run_select(candidates, verdicts_for(candidates), slots)
    b = run_select(list(reversed(candidates)), verdicts_for(candidates), list(reversed(slots)))
    assert (a.dev, a.test, a.reserve) == (b.dev, b.test, b.reserve)


def test_inputs_are_not_modified(candidates, slots):
    before = copy.deepcopy(candidates)
    select.select(candidates, verdicts_for(candidates), slots)
    assert candidates == before


# ------------------------------------------------------------------ verdict files


def write(dir_: Path, name: str, records: list) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    text = "".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in records)
    (dir_ / name).write_bytes(text.encode("utf-8"))


def hashes(candidates):
    return {c["slot_id"]: c["candidate_sha256"] for c in candidates}


def test_load_verdicts_reads_critic_files_and_ignores_extra_keys(tmp_path, candidates):
    cs = candidates[:3]
    write(
        tmp_path,
        "a.jsonl",
        [
            verdict(cs[0], reason_code="ok", competing_ids=[], fix=None),
            verdict(cs[1], "drop"),
        ],
    )
    write(tmp_path, "b.jsonl", [verdict(cs[2], "fix")])
    verdicts, problems = select.load_verdicts(tmp_path, hashes(cs))
    assert problems == []
    assert {k: v[0]["verdict"] for k, v in verdicts.items()} == {
        cs[0]["slot_id"]: "keep",
        cs[1]["slot_id"]: "drop",
        cs[2]["slot_id"]: "fix",
    }


def test_a_verdict_must_quote_the_hash_of_the_text_it_judged(tmp_path, candidates):
    c = candidates[0]
    record = verdict(c)
    del record["candidate_sha256"]
    write(tmp_path, "a.jsonl", [record])
    _, problems = select.load_verdicts(tmp_path, hashes([c]))
    assert len(problems) == 1 and "candidate_sha256" in problems[0] and "a.jsonl:1" in problems[0]


def test_a_verdict_for_rewritten_text_is_rejected_with_a_clear_message(tmp_path, candidates):
    c = candidates[0]
    write(tmp_path, "a.jsonl", [{**verdict(c), "candidate_sha256": "0" * 64}])
    verdicts, problems = select.load_verdicts(tmp_path, hashes([c]))
    assert verdicts == {}
    assert len(problems) == 1
    assert "different text" in problems[0] and c["slot_id"] in problems[0]
    assert "new verdict" in problems[0]


def test_only_the_latest_round_has_to_quote_the_current_text(tmp_path, candidates):
    c = candidates[0]
    old = {**verdict(c, "fix", round=1), "candidate_sha256": "1" * 64}  # judged the first draft
    new = verdict(c, "keep", round=2)  # judged the rewrite
    write(tmp_path, "a.jsonl", [old, new])
    verdicts, problems = select.load_verdicts(tmp_path, hashes([c]))
    assert problems == [] and len(verdicts[c["slot_id"]]) == 2
    # a stale latest round is rejected
    write(tmp_path, "a.jsonl", [verdict(c, "keep", round=1), {**old, "round": 2}])
    _, problems = select.load_verdicts(tmp_path, hashes([c]))
    assert len(problems) == 1 and "different text" in problems[0]


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
    c = candidates[0]
    if isinstance(record, dict) and record.get("slot_id") == "S":
        record = {**record, "slot_id": c["slot_id"], "candidate_sha256": c["candidate_sha256"]}
    write(tmp_path, "a.jsonl", [record])
    _, problems = select.load_verdicts(tmp_path, hashes([c]))
    assert len(problems) == 1 and needle in problems[0] and "a.jsonl:1" in problems[0]


def test_duplicate_verdicts_need_distinct_rounds(tmp_path, candidates):
    c = candidates[0]
    write(tmp_path, "a.jsonl", [verdict(c, "fix")])
    write(tmp_path, "b.jsonl", [verdict(c, "keep")])
    _, problems = select.load_verdicts(tmp_path, hashes([c]))
    assert len(problems) == 1 and "duplicate" in problems[0]


def test_verdict_files_are_split_at_line_feeds_only(tmp_path, candidates):
    c = candidates[0]
    record = verdict(c, notes="two lines\u0085in one note")
    (tmp_path / "a.jsonl").write_bytes((json.dumps(record, ensure_ascii=False) + "\n").encode())
    verdicts, problems = select.load_verdicts(tmp_path, hashes([c]))
    assert problems == [] and " " in verdicts[c["slot_id"]][0]["notes"]


def test_critic_rounds_the_latest_round_wins_and_is_counted(candidates, slots):
    sid = plan_cell(slots, "dev", "210")[0]["slot_id"]
    c = next(x for x in candidates if x["slot_id"] == sid)
    verdicts = verdicts_for(candidates)
    verdicts[sid] = [
        verdict(c, "keep", notes="after the fix", round=2),
        verdict(c, "fix", notes="first look", round=1),
    ]
    res = run_select(candidates, verdicts, slots)
    item = next(i for i in res.dev if i["provenance"]["slot_id"] == sid)
    assert item["provenance"]["critic_verdict"] == "keep"
    assert item["provenance"]["critic_rounds"] == 2
    assert item["provenance"]["critic_notes"] == "after the fix"
    verdicts[sid][0]["verdict"] = "fix"
    verdicts[sid][1]["verdict"] = "keep"  # round 1 keep, round 2 fix: the later round decides
    res = run_select(candidates, verdicts, slots)
    assert sid not in {i["provenance"]["slot_id"] for i in res.dev}


def test_missing_directory_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        select.load_verdicts(tmp_path / "nope", {})


# ------------------------------------------------------------------ the command


def prepare(tmp_path, candidates, slots, verdicts):
    cand = tmp_path / "assembled" / "candidates.jsonl"
    cand.parent.mkdir(parents=True)
    cand.write_bytes("".join(json.dumps(c, sort_keys=True) + "\n" for c in candidates).encode())
    slot_file = tmp_path / "slots.jsonl"
    plan.write_slots(slots, slot_file)
    critic = tmp_path / "critic"
    write(critic, "all.jsonl", [r for records in verdicts.values() for r in records])
    return cand, slot_file, critic


def args(tmp_path, cand, slot_file, critic, *extra):
    return [
        "--candidates", str(cand),
        "--slots", str(slot_file),
        "--critic", str(critic),
        "--out-dir", str(tmp_path / "selected"),
        "--reserve-dir", str(tmp_path / "reserve"),
        *extra,
    ]  # fmt: skip


def test_command_writes_dev_test_and_reserve(tmp_path, candidates, slots, capsys):
    cand, slot_file, critic = prepare(tmp_path, candidates, slots, verdicts_for(candidates))
    assert select.run(args(tmp_path, cand, slot_file, critic)) == 0
    dev = read_jsonl(tmp_path / "selected" / "dev.jsonl")
    test = read_jsonl(tmp_path / "selected" / "test.jsonl")
    assert len(dev) + len(test) == 170 and dev[0]["id"] == "v1-0001"
    reserve = len(read_jsonl(tmp_path / "reserve" / "dev.jsonl")) + len(
        read_jsonl(tmp_path / "reserve" / "test.jsonl")
    )
    assert reserve == 88
    raw = (tmp_path / "selected" / "dev.jsonl").read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    out = capsys.readouterr().out
    assert "selected 170" in out and "reserve 88" in out and "STOP" not in out


def test_command_is_idempotent(tmp_path, candidates, slots):
    cand, slot_file, critic = prepare(tmp_path, candidates, slots, verdicts_for(candidates))
    select.run(args(tmp_path, cand, slot_file, critic))
    first = (tmp_path / "selected" / "test.jsonl").read_bytes()
    select.run(args(tmp_path, cand, slot_file, critic))
    assert (tmp_path / "selected" / "test.jsonl").read_bytes() == first


def test_command_prints_every_backfill_mismatch(tmp_path, candidates, slots, capsys):
    victim = plan_cell(slots, "dev", "220")[0]
    verdicts = verdicts_for(candidates, overrides={victim["slot_id"]: "drop"})
    cand, slot_file, critic = prepare(tmp_path, candidates, slots, verdicts)
    assert select.run(args(tmp_path, cand, slot_file, critic)) == 0
    out = capsys.readouterr().out
    quota = select.cell_quota("dev", "220")
    surplus = plan_cell(slots, "dev", "220")[quota:]
    best = min(surplus, key=lambda o: (select._tier(victim, o), o["slot_id"]))
    if select._tier(victim, best) > 1:
        assert f"BACKFILL dev/220: {victim['slot_id']}" in out
    else:
        assert "BACKFILL" not in out


def test_stop_prints_exits_non_zero_and_writes_nothing(tmp_path, candidates, slots, capsys):
    overrides = {c["slot_id"]: "drop" for c in candidates[:120]}
    cand, slot_file, critic = prepare(
        tmp_path, candidates, slots, verdicts_for(candidates, overrides=overrides)
    )
    code = select.run(args(tmp_path, cand, slot_file, critic))
    out = capsys.readouterr().out
    assert code == 1
    assert "STOP" in out and "40%" in out and "short" in out.lower()
    assert not (tmp_path / "selected").exists() and not (tmp_path / "reserve").exists()


def test_force_writes_the_files_although_stop_applies(tmp_path, candidates, slots, capsys):
    overrides = {c["slot_id"]: "drop" for c in candidates[:120]}
    cand, slot_file, critic = prepare(
        tmp_path, candidates, slots, verdicts_for(candidates, overrides=overrides)
    )
    code = select.run(args(tmp_path, cand, slot_file, critic, "--force"))
    out = capsys.readouterr().out
    assert code == 0 and "STOP" in out and "--force" in out
    assert (tmp_path / "selected" / "dev.jsonl").exists()


def test_a_small_run_such_as_the_pilot_needs_force(tmp_path, candidates, slots, capsys):
    few = [c for c in candidates if c["slot_id"] in {s["slot_id"] for s in slots if s["pilot"]}]
    cand, slot_file, critic = prepare(tmp_path, few, slots, verdicts_for(few))
    assert select.run(args(tmp_path, cand, slot_file, critic)) == 1
    assert "only" in capsys.readouterr().out
    assert select.run(args(tmp_path, cand, slot_file, critic, "--force")) == 0


def test_command_with_verdict_problems_writes_nothing(tmp_path, candidates, slots, capsys):
    cand, slot_file, critic = prepare(tmp_path, candidates, slots, verdicts_for(candidates))
    write(critic, "bad.jsonl", [{"slot_id": "s-9999", "verdict": "keep", "notes": ""}])
    assert select.run(args(tmp_path, cand, slot_file, critic)) == 1
    assert not (tmp_path / "selected").exists()
    assert "unknown slot" in capsys.readouterr().out


def test_command_rejects_a_verdict_for_other_text(tmp_path, candidates, slots, capsys):
    verdicts = verdicts_for(candidates)
    sid = candidates[0]["slot_id"]
    verdicts[sid][0]["candidate_sha256"] = "f" * 64
    cand, slot_file, critic = prepare(tmp_path, candidates, slots, verdicts)
    assert select.run(args(tmp_path, cand, slot_file, critic, "--force")) == 1
    out = capsys.readouterr().out
    assert "different text" in out and sid in out
    assert not (tmp_path / "selected").exists()
