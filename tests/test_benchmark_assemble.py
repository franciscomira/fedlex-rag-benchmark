"""Draft records -> candidates (``benchmark assemble``). Strict parsing, ids copied by code."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark_fixtures import make_corpus
from fedlex_bench.benchmark import assemble, drafts, plan
from fedlex_bench.benchmark.bm25 import build_indexes, compute_ranks


@pytest.fixture(scope="module")
def corpus():
    return make_corpus()


@pytest.fixture(scope="module")
def slots(corpus):
    return plan.build_plan(corpus).slots


@pytest.fixture(scope="module")
def indexes(corpus):
    return build_indexes(corpus)


def pick(slots, **match):
    return next(s for s in slots if all(s[k] == v for k, v in match.items()))


def ans_record(slot_id, **over):
    rec = {
        "slot_id": slot_id,
        "skip": False,
        "skip_reason": None,
        "question": "Wie lange dauert die Frist?",
        "gold_facts": [{"text": "The period is ten days.", "article": "A"}],
        "difficulty": "medium",
        "competing_candidates": [],
        "note": "",
    }
    rec.update(over)
    return rec


def cross_record(slot_id, **over):
    facts = [
        {"text": "Fact from the first article.", "article": "A"},
        {"text": "Fact from the second article.", "article": "B"},
    ]
    return ans_record(slot_id, **{"gold_facts": facts, **over})


def trap_record(slot_id, **over):
    rec = {
        "slot_id": slot_id,
        "skip": False,
        "skip_reason": None,
        "question": "How high is the income tax in canton X?",
        "trap_reason": "Tax law is not part of the eight acts.",
    }
    rec.update(over)
    return rec


def write(dir_: Path, name: str, records: list) -> Path:
    path = dir_ / name
    path.write_text(
        "".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in records),
        encoding="utf-8",
    )
    return path


# ------------------------------------------------------------------ parsing


def test_valid_drafts_are_loaded(tmp_path, slots):
    s1, s2, s3 = (pick(slots, qtype=t) for t in ("single", "cross_ref", "trap"))
    write(
        tmp_path,
        "a.jsonl",
        [ans_record(s1["slot_id"]), cross_record(s2["slot_id"]), trap_record(s3["slot_id"])],
    )
    loaded = drafts.load_drafts(tmp_path, slots)
    assert loaded.problems == []
    assert set(loaded.records) == {s1["slot_id"], s2["slot_id"], s3["slot_id"]}


def problems_for(tmp_path, slots, *records):
    write(tmp_path, "w.jsonl", list(records))
    return drafts.load_drafts(tmp_path, slots).problems


def test_invalid_json_is_reported_with_file_and_line(tmp_path, slots):
    s = pick(slots, qtype="single")
    write(tmp_path, "w.jsonl", [ans_record(s["slot_id"]), "{not json"])
    loaded = drafts.load_drafts(tmp_path, slots)
    assert any("w.jsonl:2" in p and "JSON" in p for p in loaded.problems)
    assert s["slot_id"] in loaded.records  # the good line is still parsed


def test_unknown_slot_id_is_rejected(tmp_path, slots):
    probs = problems_for(tmp_path, slots, ans_record("s-9999"))
    assert len(probs) == 1 and "s-9999" in probs[0] and "unknown slot" in probs[0]


def test_duplicate_slot_ids_are_rejected_within_and_across_files(tmp_path, slots):
    s = pick(slots, qtype="single")
    rec = ans_record(s["slot_id"])
    write(tmp_path, "a.jsonl", [rec, rec])
    write(tmp_path, "b.jsonl", [rec])
    probs = drafts.load_drafts(tmp_path, slots).problems
    assert sum("duplicate" in p for p in probs) == 2
    assert all(s["slot_id"] in p for p in probs)


@pytest.mark.parametrize("missing", ["question", "gold_facts", "difficulty", "skip", "slot_id"])
def test_missing_required_fields_are_reported(tmp_path, slots, missing):
    s = pick(slots, qtype="single")
    rec = ans_record(s["slot_id"])
    del rec[missing]
    probs = problems_for(tmp_path, slots, rec)
    assert len(probs) == 1 and missing in probs[0]


def test_trap_needs_a_reason(tmp_path, slots):
    s = pick(slots, qtype="trap")
    rec = trap_record(s["slot_id"])
    del rec["trap_reason"]
    assert "trap_reason" in problems_for(tmp_path, slots, rec)[0]
    assert (
        "trap_reason"
        in problems_for(tmp_path, slots, trap_record(s["slot_id"], trap_reason=" "))[0]
    )


def test_unknown_keys_are_rejected(tmp_path, slots):
    s = pick(slots, qtype="single")
    probs = problems_for(tmp_path, slots, ans_record(s["slot_id"], gold_article_ids=["220:1"]))
    assert len(probs) == 1 and "gold_article_ids" in probs[0]


def test_skip_needs_a_reason_and_nothing_else(tmp_path, slots):
    s = pick(slots, qtype="single")
    ok = {"slot_id": s["slot_id"], "skip": True, "skip_reason": "article too vague"}
    assert problems_for(tmp_path, slots, ok) == []
    bad = {"slot_id": s["slot_id"], "skip": True, "skip_reason": ""}
    assert "skip_reason" in problems_for(tmp_path, slots, bad)[0]


def test_label_and_fact_rules(tmp_path, slots):
    single, cross = pick(slots, qtype="single"), pick(slots, qtype="cross_ref")
    cases = [
        (ans_record(single["slot_id"], gold_facts=[{"text": "x", "article": "B"}]), "label"),
        (ans_record(single["slot_id"], gold_facts=[]), "gold_facts"),
        (
            ans_record(
                single["slot_id"], gold_facts=[{"text": f"f{i}", "article": "A"} for i in range(4)]
            ),
            "gold_facts",
        ),
        (ans_record(single["slot_id"], gold_facts=[{"text": "", "article": "A"}]), "text"),
        (
            ans_record(single["slot_id"], gold_facts=[{"text": "x", "article": "A", "id": "1"}]),
            "key",
        ),
        (ans_record(single["slot_id"], gold_facts=["plain string"]), "gold_facts"),
        (ans_record(single["slot_id"], difficulty="trivial"), "difficulty"),
        (ans_record(single["slot_id"], question=""), "question"),
        (ans_record(single["slot_id"], question=42), "question"),
        (ans_record(single["slot_id"], competing_candidates="x"), "competing_candidates"),
        (ans_record(single["slot_id"], skip="no"), "skip"),
        (ans_record(single["slot_id"], trap_reason="x"), "trap_reason"),
        # cross_ref: both labels must be used, A and B only
        (ans_record(cross["slot_id"]), "both"),
        (
            ans_record(
                cross["slot_id"],
                gold_facts=[{"text": "x", "article": "A"}, {"text": "y", "article": "C"}],
            ),
            "label",
        ),
        (
            cross_record(
                cross["slot_id"],
                gold_facts=[{"text": f"f{i}", "article": "AB"[i % 2]} for i in range(5)],
            ),
            "gold_facts",
        ),
    ]
    for rec, needle in cases:
        write(tmp_path, "w.jsonl", [rec])
        probs = drafts.load_drafts(tmp_path, slots).problems
        assert len(probs) >= 1 and needle in " ".join(probs), (rec, probs)


def test_up_to_four_facts_are_allowed_for_cross_ref(tmp_path, slots):
    cross = pick(slots, qtype="cross_ref")
    facts = [{"text": f"f{i}", "article": "AB"[i % 2]} for i in range(4)]
    assert problems_for(tmp_path, slots, cross_record(cross["slot_id"], gold_facts=facts)) == []


def test_trap_slot_with_facts_or_difficulty_is_rejected(tmp_path, slots):
    s = pick(slots, qtype="trap")
    assert problems_for(
        tmp_path, slots, trap_record(s["slot_id"], gold_facts=[{"text": "x", "article": "A"}])
    )
    assert problems_for(tmp_path, slots, trap_record(s["slot_id"], difficulty="easy"))
    assert (
        problems_for(tmp_path, slots, trap_record(s["slot_id"], gold_facts=[], difficulty=None))
        == []
    )


def test_a_writer_typing_an_article_id_is_rejected(tmp_path, slots):
    s = pick(slots, qtype="single")
    probs = problems_for(tmp_path, slots, ans_record(s["slot_id"], question="Was sagt 220:257d?"))
    assert len(probs) == 1 and "article id" in probs[0]
    facts = [{"text": "See 210:140 for details.", "article": "A"}]
    assert (
        "article id" in problems_for(tmp_path, slots, ans_record(s["slot_id"], gold_facts=facts))[0]
    )


def test_all_problems_are_listed_not_just_the_first(tmp_path, slots):
    a, b = pick(slots, qtype="single"), pick(slots, qtype="trap")
    write(
        tmp_path,
        "w.jsonl",
        [
            ans_record(a["slot_id"], difficulty="bad"),
            trap_record(b["slot_id"], trap_reason=""),
            "oops",
        ],
    )
    assert len(drafts.load_drafts(tmp_path, slots).problems) == 3


def test_empty_or_missing_directory(tmp_path, slots):
    assert drafts.load_drafts(tmp_path, slots).records == {}
    with pytest.raises(FileNotFoundError):
        drafts.load_drafts(tmp_path / "nope", slots)


# ------------------------------------------------------------------ assemble


def assembled(tmp_path, slots, corpus, indexes, records, **kw):
    write(tmp_path, "w.jsonl", records)
    loaded = drafts.load_drafts(tmp_path, slots)
    assert loaded.problems == []
    return assemble.assemble(
        slots, loaded.records, corpus, indexes=indexes, drafted_on="2026-10-16", **kw
    )


def test_ids_acts_languages_and_split_are_copied_from_the_plan(tmp_path, slots, corpus, indexes):
    s = pick(slots, qtype="single", query_lang="fr", source_lang="de")
    res = assembled(tmp_path, slots, corpus, indexes, [ans_record(s["slot_id"])])
    (item,) = res.candidates
    assert item["gold_article_ids"] == s["gold_article_ids"]
    assert item["source_act"] == s["act"]
    assert (item["split"], item["query_lang"], item["source_lang"], item["qtype"]) == (
        s["split"],
        "fr",
        "de",
        "single",
    )
    assert item["is_trap"] is False
    assert item["slot_id"] == s["slot_id"]
    assert item["id"] is None
    assert item["gold_facts"] == [
        {"text": "The period is ten days.", "article_id": s["gold_article_ids"][0]}
    ]
    assert item["difficulty"] == "medium" and item["question"] == "Wie lange dauert die Frist?"


def test_cross_ref_labels_map_to_the_two_gold_ids_in_plan_order(tmp_path, slots, corpus, indexes):
    s = pick(slots, qtype="cross_ref")
    (item,) = assembled(tmp_path, slots, corpus, indexes, [cross_record(s["slot_id"])]).candidates
    a, b = s["gold_article_ids"]
    assert [f["article_id"] for f in item["gold_facts"]] == [a, b]
    assert item["gold_article_ids"] == [a, b] and item["source_act"] == a.partition(":")[0]


def test_trap_item_has_no_gold_and_no_ranks(tmp_path, slots, corpus, indexes):
    s = pick(slots, qtype="trap")
    (item,) = assembled(tmp_path, slots, corpus, indexes, [trap_record(s["slot_id"])]).candidates
    assert item["is_trap"] is True and item["qtype"] == "trap"
    assert item["gold_article_ids"] == [] and item["gold_facts"] == []
    assert item["source_act"] is None and item["source_lang"] is None
    assert item["trap_type"] == s["trap_type"]
    assert item["trap_reason"] == "Tax law is not part of the eight acts."
    assert item["bm25_rank"] is None and item["lexical_overlap"] is None
    assert item["difficulty"] is None


def test_bm25_fields_are_computed_by_code(tmp_path, slots, corpus, indexes):
    s = pick(slots, qtype="single", query_lang="de")
    q = "Inhalt zu Nummer 3 inhalt3 wortlaut"
    (item,) = assembled(
        tmp_path, slots, corpus, indexes, [ans_record(s["slot_id"], question=q)]
    ).candidates
    ranks, overlap = compute_ranks(q, s["gold_article_ids"], "de", indexes)
    assert item["bm25_rank"] == ranks and item["lexical_overlap"] == overlap == ranks["de"]
    assert set(item["bm25_rank"]) == {"de", "fr", "it"}
    en = pick(slots, qtype="single", query_lang="en")
    (item,) = assembled(
        tmp_path, slots, corpus, indexes, [ans_record(en["slot_id"], question=q)]
    ).candidates
    assert item["lexical_overlap"] == min(item["bm25_rank"].values())


def test_review_fields_stay_unreviewed_and_translation_is_empty(tmp_path, slots, corpus, indexes):
    s = pick(slots, qtype="single")
    (item,) = assembled(tmp_path, slots, corpus, indexes, [ans_record(s["slot_id"])]).candidates
    assert item["review_status"] == "unreviewed"
    assert (
        item["reviewed_by"] is None and item["reviewed_at"] is None and item["review_note"] is None
    )
    assert item["translation_of"] is None


def test_provenance(tmp_path, slots, corpus, indexes):
    agent = tmp_path / "agent.md"
    agent.write_text("role file", encoding="utf-8")
    s = pick(slots, qtype="single")
    res = assembled(
        tmp_path,
        slots,
        corpus,
        indexes,
        [ans_record(s["slot_id"], note="check this", competing_candidates=["limitation periods"])],
        agent_file=agent,
        plan_sha256="c" * 64,
        drafter="question-writer",
        model="opus",
    )
    prov = res.candidates[0]["provenance"]
    import hashlib

    assert prov["agent_file_sha256"] == hashlib.sha256(b"role file").hexdigest()
    assert prov["plan_sha256"] == "c" * 64
    assert prov["drafted_on"] == "2026-10-16"
    assert (prov["drafter"], prov["model"]) == ("question-writer", "opus")
    assert prov["slot_id"] == s["slot_id"]
    assert (prov["critic_verdict"], prov["critic_rounds"], prov["critic_notes"]) == (None, 0, None)
    assert prov["writer_note"] == "check this"
    assert prov["competing_candidates"] == ["limitation periods"]
    assert prov["brief_sha256"] is None


def test_brief_sha256_comes_from_the_brief_index(tmp_path, slots, corpus, indexes):
    s = pick(slots, qtype="single")
    res = assembled(
        tmp_path,
        slots,
        corpus,
        indexes,
        [ans_record(s["slot_id"])],
        brief_index={s["slot_id"]: {"file": "220-01.md", "sha256": "d" * 64}},
    )
    assert res.candidates[0]["provenance"]["brief_sha256"] == "d" * 64


def test_skip_is_dropped_and_logged(tmp_path, slots, corpus, indexes):
    a, b = [s for s in slots if s["qtype"] == "single"][:2]
    res = assembled(
        tmp_path,
        slots,
        corpus,
        indexes,
        [
            ans_record(a["slot_id"]),
            {"slot_id": b["slot_id"], "skip": True, "skip_reason": "too vague"},
        ],
    )
    assert [c["slot_id"] for c in res.candidates] == [a["slot_id"]]
    assert res.skipped == [{"slot_id": b["slot_id"], "skip_reason": "too vague"}]
    assert res.missing_slots and b["slot_id"] not in res.missing_slots  # skipped is not "missing"


def test_slots_without_a_draft_are_reported_as_missing_not_as_errors(
    tmp_path, slots, corpus, indexes
):
    s = pick(slots, qtype="single")
    res = assembled(tmp_path, slots, corpus, indexes, [ans_record(s["slot_id"])])
    assert len(res.missing_slots) == len(slots) - 1


def test_candidates_are_sorted_by_slot_id(tmp_path, slots, corpus, indexes):
    picks = [s for s in slots if s["qtype"] == "single"][:5]
    recs = [ans_record(s["slot_id"]) for s in reversed(picks)]
    res = assembled(tmp_path, slots, corpus, indexes, recs)
    assert [c["slot_id"] for c in res.candidates] == sorted(s["slot_id"] for s in picks)


def test_candidate_keys_are_the_item_schema_plus_slot_id(tmp_path, slots, corpus, indexes):
    s = pick(slots, qtype="single")
    (item,) = assembled(tmp_path, slots, corpus, indexes, [ans_record(s["slot_id"])]).candidates
    from fedlex_bench.benchmark.validate import ITEM_KEYS

    assert set(item) == ITEM_KEYS | {"slot_id"}


def test_bad_drafted_on_is_an_error(tmp_path, slots, corpus, indexes):
    s = pick(slots, qtype="single")
    write(tmp_path, "w.jsonl", [ans_record(s["slot_id"])])
    loaded = drafts.load_drafts(tmp_path, slots)
    with pytest.raises(ValueError, match="drafted_on"):
        assemble.assemble(slots, loaded.records, corpus, indexes=indexes, drafted_on="16.10.2026")


# ------------------------------------------------------------------ the command


def run_assemble(tmp_path, slots, corpus, extra=()):
    slots_path = tmp_path / "slots.jsonl"
    plan.write_slots(slots, slots_path)
    out = tmp_path / "out" / "candidates.jsonl"
    code = assemble.run(
        [
            "--slots",
            str(slots_path),
            "--drafts",
            str(tmp_path / "drafts"),
            "--out",
            str(out),
            "--drafted-on",
            "2026-10-16",
            *extra,
        ],
        corpus=corpus,
    )
    return code, out


def test_command_writes_candidates_and_returns_zero(tmp_path, slots, corpus, capsys):
    (tmp_path / "drafts").mkdir()
    s = pick(slots, qtype="single")
    write(tmp_path / "drafts", "w.jsonl", [ans_record(s["slot_id"])])
    code, out = run_assemble(tmp_path, slots, corpus)
    assert code == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert list(json.loads(lines[0])) == sorted(json.loads(lines[0]))  # keys sorted
    assert b"\r" not in out.read_bytes()
    assert "1 candidates" in capsys.readouterr().out


def test_command_with_problems_prints_them_and_writes_nothing(tmp_path, slots, corpus, capsys):
    (tmp_path / "drafts").mkdir()
    s = pick(slots, qtype="single")
    write(tmp_path / "drafts", "w.jsonl", [ans_record(s["slot_id"]), ans_record("s-9999"), "{bad"])
    code, out = run_assemble(tmp_path, slots, corpus)
    assert code == 1 and not out.exists()
    text = capsys.readouterr().out
    assert "2 problem" in text and "s-9999" in text and "w.jsonl:3" in text


def test_command_allow_partial_writes_the_valid_records(tmp_path, slots, corpus, capsys):
    (tmp_path / "drafts").mkdir()
    s = pick(slots, qtype="single")
    write(tmp_path / "drafts", "w.jsonl", [ans_record(s["slot_id"]), ans_record("s-9999")])
    code, out = run_assemble(tmp_path, slots, corpus, extra=["--allow-partial"])
    assert code == 1 and out.exists()  # still non-zero: there were problems
    assert len(out.read_text(encoding="utf-8").splitlines()) == 1
