"""Draft records -> candidates (``benchmark assemble``). Strict parsing, ids copied by code."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark_fixtures import make_corpus
from fedlex_bench.benchmark import assemble, drafts, plan
from fedlex_bench.benchmark.bm25 import build_indexes, compute_ranks
from fedlex_bench.benchmark.brief import BriefIndex
from fedlex_bench.benchmark.common import canonical_sha256, slot_hash


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


def test_answerable_record_may_carry_trap_reason_null_and_empty_note(tmp_path, slots):
    s = pick(slots, qtype="single")
    rec = ans_record(s["slot_id"], trap_reason=None, note="")
    assert problems_for(tmp_path, slots, rec) == []
    assert (
        problems_for(tmp_path, slots, ans_record(s["slot_id"], trap_reason=None, note=None)) == []
    )
    c = pick(slots, qtype="cross_ref")
    assert problems_for(tmp_path, slots, cross_record(c["slot_id"], trap_reason=None)) == []


def test_answerable_record_with_a_trap_reason_string_is_rejected(tmp_path, slots):
    s = pick(slots, qtype="single")
    for value in ("a reason", "", " "):
        probs = problems_for(tmp_path, slots, ans_record(s["slot_id"], trap_reason=value))
        assert len(probs) == 1 and "trap_reason must be null" in probs[0], value


def test_skip_record_with_null_fields_is_accepted(tmp_path, slots):
    s = pick(slots, qtype="single")
    rec = {
        "slot_id": s["slot_id"],
        "skip": True,
        "skip_reason": "vague",
        "question": None,
        "trap_reason": None,
        "gold_facts": None,
        "difficulty": None,
        "note": "",
    }
    assert problems_for(tmp_path, slots, rec) == []


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


PLAN_SHA = "e" * 64


def index_for(slots, plan_sha256=PLAN_SHA, sha="d" * 64) -> BriefIndex:
    """A brief index as ``brief`` writes it: every slot with its brief file and slot hash."""
    entries = {
        s["slot_id"]: {"file": "x.md", "sha256": sha, "slot_sha256": slot_hash(s)} for s in slots
    }
    return BriefIndex(plan_sha256=plan_sha256, slots=entries)


def assembled(tmp_path, slots, corpus, indexes, records, **kw):
    write(tmp_path, "w.jsonl", records)
    loaded = drafts.load_drafts(tmp_path, slots)
    assert loaded.problems == []
    kw.setdefault("brief_index", index_for(slots))
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
    number = s["gold_article_ids"][0].partition(":")[2]
    q = f"Frage zu inhalt{number} und wortlaut"  # shares a token with the gold article
    (item,) = assembled(
        tmp_path, slots, corpus, indexes, [ans_record(s["slot_id"], question=q)]
    ).candidates
    ranks, overlap = compute_ranks(q, s["gold_article_ids"], "de", indexes)
    assert item["bm25_rank"] == ranks and item["lexical_overlap"] == overlap == ranks["de"]
    assert ranks["de"] == 1 and set(item["bm25_rank"]) == {"de", "fr", "it"}


def test_a_gold_article_without_a_shared_token_has_null_ranks(tmp_path, slots, corpus, indexes):
    s = pick(slots, qtype="single", query_lang="fr")
    (item,) = assembled(
        tmp_path, slots, corpus, indexes, [ans_record(s["slot_id"], question="zzzz qqqq xxxx?")]
    ).candidates
    assert item["bm25_rank"] == {"de": None, "fr": None, "it": None}
    assert item["lexical_overlap"] is None


def test_english_items_keep_the_three_ranks_but_have_no_lexical_overlap(
    tmp_path, slots, corpus, indexes
):
    en = pick(slots, qtype="single", query_lang="en")
    number = en["gold_article_ids"][0].partition(":")[2]
    q = f"What about inhalt{number}?"
    (item,) = assembled(
        tmp_path, slots, corpus, indexes, [ans_record(en["slot_id"], question=q)]
    ).candidates
    assert item["lexical_overlap"] is None
    assert all(item["bm25_rank"][lang] == 1 for lang in ("de", "fr", "it"))


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
        drafter="question-writer",
        model="opus",
    )
    prov = res.candidates[0]["provenance"]
    import hashlib

    assert prov["agent_file_sha256"] == hashlib.sha256(b"role file").hexdigest()
    assert prov["plan_sha256"] == PLAN_SHA  # from the brief index, not from any current file
    assert prov["drafted_on"] == "2026-10-16"
    assert (prov["drafter"], prov["model"]) == ("question-writer", "opus")
    assert prov["slot_id"] == s["slot_id"]
    assert (prov["critic_verdict"], prov["critic_rounds"], prov["critic_notes"]) == (None, 0, None)
    assert prov["writer_note"] == "check this"
    assert prov["competing_candidates"] == ["limitation periods"]
    assert prov["brief_sha256"] == "d" * 64


def test_brief_sha256_comes_from_the_brief_index(tmp_path, slots, corpus, indexes):
    s = pick(slots, qtype="single")
    res = assembled(
        tmp_path,
        slots,
        corpus,
        indexes,
        [ans_record(s["slot_id"])],
        brief_index=index_for(slots, plan_sha256="f" * 64, sha="9" * 64),
    )
    prov = res.candidates[0]["provenance"]
    assert prov["brief_sha256"] == "9" * 64 and prov["plan_sha256"] == "f" * 64


def test_a_slot_that_changed_since_the_brief_was_written_is_refused(
    tmp_path, slots, corpus, indexes
):
    s = pick(slots, qtype="single")
    changed = [
        {**x, "gold_article_ids": ["220:999"]} if x["slot_id"] == s["slot_id"] else x for x in slots
    ]
    res = assembled(
        tmp_path, slots, corpus, indexes, [ans_record(s["slot_id"])], brief_index=index_for(changed)
    )
    assert res.candidates == []
    assert len(res.problems) == 1 and s["slot_id"] in res.problems[0]
    assert "differs" in res.problems[0] and "brief" in res.problems[0]


def test_the_pilot_flag_does_not_count_as_a_change(tmp_path, slots, corpus, indexes):
    s = pick(slots, qtype="single")
    flipped = [{**x, "pilot": not x["pilot"]} if x["slot_id"] == s["slot_id"] else x for x in slots]
    res = assembled(
        tmp_path, slots, corpus, indexes, [ans_record(s["slot_id"])], brief_index=index_for(flipped)
    )
    assert len(res.candidates) == 1 and res.problems == []


def test_a_slot_without_a_brief_is_refused(tmp_path, slots, corpus, indexes):
    s = pick(slots, qtype="single")
    index = index_for([x for x in slots if x["slot_id"] != s["slot_id"]])
    res = assembled(tmp_path, slots, corpus, indexes, [ans_record(s["slot_id"])], brief_index=index)
    assert res.candidates == [] and "no brief" in res.problems[0]


def test_candidate_sha256_covers_question_and_gold_facts(tmp_path, slots, corpus, indexes):
    s = pick(slots, qtype="single")
    (a,) = assembled(tmp_path, slots, corpus, indexes, [ans_record(s["slot_id"])]).candidates
    expected = canonical_sha256({"question": a["question"], "gold_facts": a["gold_facts"]})
    assert a["candidate_sha256"] == expected
    (b,) = assembled(
        tmp_path, slots, corpus, indexes, [ans_record(s["slot_id"], question="Eine andere Frage?")]
    ).candidates
    assert b["candidate_sha256"] != a["candidate_sha256"]
    facts = [{"text": "A different fact.", "article": "A"}]
    (c,) = assembled(
        tmp_path, slots, corpus, indexes, [ans_record(s["slot_id"], gold_facts=facts)]
    ).candidates
    assert c["candidate_sha256"] not in {a["candidate_sha256"], b["candidate_sha256"]}
    # difficulty and the writer note are not part of what the critic judges
    (d,) = assembled(
        tmp_path, slots, corpus, indexes, [ans_record(s["slot_id"], difficulty="hard", note="x")]
    ).candidates
    assert d["candidate_sha256"] == a["candidate_sha256"]


def test_trap_candidate_sha256(tmp_path, slots, corpus, indexes):
    s = pick(slots, qtype="trap")
    (a,) = assembled(tmp_path, slots, corpus, indexes, [trap_record(s["slot_id"])]).candidates
    (b,) = assembled(
        tmp_path, slots, corpus, indexes, [trap_record(s["slot_id"], question="Another trap?")]
    ).candidates
    assert a["candidate_sha256"] != b["candidate_sha256"]


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

    assert set(item) == ITEM_KEYS | {"slot_id", "candidate_sha256"}


def test_bad_drafted_on_is_an_error(tmp_path, slots, corpus, indexes):
    s = pick(slots, qtype="single")
    write(tmp_path, "w.jsonl", [ans_record(s["slot_id"])])
    loaded = drafts.load_drafts(tmp_path, slots)
    with pytest.raises(ValueError, match="drafted_on"):
        assemble.assemble(
            slots,
            loaded.records,
            corpus,
            indexes=indexes,
            drafted_on="16.10.2026",
            brief_index=index_for(slots),
        )


# ------------------------------------------------------------------ the command


def write_index(path, slots, plan_sha256=PLAN_SHA):
    index = index_for(slots, plan_sha256)
    path.write_bytes(
        (json.dumps({"plan_sha256": index.plan_sha256, "slots": index.slots}) + "\n").encode()
    )
    return path


def run_assemble(tmp_path, slots, corpus, extra=(), index=True):
    slots_path = tmp_path / "slots.jsonl"
    plan.write_slots(slots, slots_path)
    out = tmp_path / "out" / "candidates.jsonl"
    args = [
        "--slots",
        str(slots_path),
        "--drafts",
        str(tmp_path / "drafts"),
        "--out",
        str(out),
        "--drafted-on",
        "2026-10-16",
    ]
    if index:
        full = [x for x in slots if not x["pilot"]]
        pilot = [x for x in slots if x["pilot"]]
        a = write_index(tmp_path / "index.json", full)
        b = write_index(tmp_path / "index-pilot.json", pilot)
        args += ["--brief-index", str(a), str(b)]
    code = assemble.run([*args, *extra], corpus=corpus)
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


# ------------------------------------------------------------------ what a writer must not type


@pytest.mark.parametrize(
    "text",
    [
        "Was sagt Artikel 5 dazu?",
        "Que dit l'article 4 ?",
        "Cosa dice l'articolo 7?",
        "What does Art. 12 say?",
        "See ARTICLES 3 and 4",
        "Und art.5 des Gesetzes?",
        "Was regelt SR 220?",
        "Was regelt sr281.1?",
    ],
)
def test_article_numbers_and_sr_numbers_are_rejected_in_questions_and_facts(tmp_path, slots, text):
    s = pick(slots, qtype="single")
    probs = problems_for(tmp_path, slots, ans_record(s["slot_id"], question=text))
    assert len(probs) == 1 and "question" in probs[0] and "writers never type" in probs[0], text
    facts = [{"text": text, "article": "A"}]
    probs = problems_for(tmp_path, slots, ans_record(s["slot_id"], gold_facts=facts))
    assert len(probs) == 1 and "gold_facts[0]" in probs[0] and "article" in probs[0], text
    trap = pick(slots, qtype="trap")
    assert problems_for(tmp_path, slots, trap_record(trap["slot_id"], question=text))


def test_ordinary_words_that_look_like_article_mentions_are_fine(tmp_path, slots):
    s = pick(slots, qtype="single")
    for text in (
        "Which article of the contract applies to a sale?",
        "Wie viele Wochen dauert die Frist?",
        "Un start 5 minuti dopo",
    ):
        assert problems_for(tmp_path, slots, ans_record(s["slot_id"], question=text)) == [], text


def test_a_skip_record_must_have_empty_other_fields(tmp_path, slots):
    s = pick(slots, qtype="single")
    base = {"slot_id": s["slot_id"], "skip": True, "skip_reason": "vague"}
    assert problems_for(tmp_path, slots, {**base, "question": None, "gold_facts": []}) == []
    assert problems_for(tmp_path, slots, {**base, "note": "free text is fine"}) == []
    for key, value in (
        ("question", "Eine Frage?"),
        ("gold_facts", [{"text": "x", "article": "A"}]),
        ("difficulty", "easy"),
        ("trap_reason", "because"),
        ("competing_candidates", ["x"]),
    ):
        probs = problems_for(tmp_path, slots, {**base, key: value})
        assert len(probs) == 1 and key in probs[0] and "skip" in probs[0], key


def test_drafts_are_split_at_line_feeds_only(tmp_path, slots):
    s = pick(slots, qtype="single")
    rec = ans_record(s["slot_id"], question="Erste Zeile zweite Zeile\u0085dritte?")
    (tmp_path / "w.jsonl").write_bytes((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
    loaded = drafts.load_drafts(tmp_path, slots)
    assert loaded.problems == []
    assert " " in loaded.records[s["slot_id"]]["question"]


def test_command_merges_the_full_and_the_pilot_index(tmp_path, slots, corpus, capsys):
    (tmp_path / "drafts").mkdir()
    pilot_slot = next(x for x in slots if x["pilot"] and x["qtype"] == "single")
    normal = next(x for x in slots if not x["pilot"] and x["qtype"] == "single")
    write(
        tmp_path / "drafts",
        "w.jsonl",
        [ans_record(pilot_slot["slot_id"]), ans_record(normal["slot_id"])],
    )
    code, out = run_assemble(tmp_path, slots, corpus)
    assert code == 0 and len(out.read_text(encoding="utf-8").splitlines()) == 2


def test_command_needs_a_brief_index(tmp_path, slots, corpus, capsys, monkeypatch):
    # the default index files are those of the repository: point them at nothing
    monkeypatch.setattr(assemble, "INDEX_PATH", tmp_path / "none-index.json")
    monkeypatch.setattr(assemble, "INDEX_PILOT_PATH", tmp_path / "none-pilot.json")
    (tmp_path / "drafts").mkdir()
    s = pick(slots, qtype="single")
    write(tmp_path / "drafts", "w.jsonl", [ans_record(s["slot_id"])])
    code, out = run_assemble(tmp_path, slots, corpus, index=False)
    assert code == 1 and not out.exists()
    assert "brief index" in capsys.readouterr().out


def test_command_refuses_a_slot_that_differs_from_its_brief(tmp_path, slots, corpus, capsys):
    (tmp_path / "drafts").mkdir()
    s = pick(slots, qtype="single")
    write(tmp_path / "drafts", "w.jsonl", [ans_record(s["slot_id"])])
    other_plan = [
        {**x, "gold_article_ids": ["220:998"]} if x["slot_id"] == s["slot_id"] else x for x in slots
    ]
    slots_path = tmp_path / "slots.jsonl"
    plan.write_slots(slots, slots_path)
    idx = write_index(tmp_path / "index.json", other_plan)
    out = tmp_path / "out.jsonl"
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
            "--brief-index",
            str(idx),
        ],
        corpus=corpus,
    )
    assert code == 1 and not out.exists()
    assert s["slot_id"] in capsys.readouterr().out


def test_command_refuses_indexes_from_different_plans(tmp_path, slots, corpus, capsys):
    (tmp_path / "drafts").mkdir()
    s = pick(slots, qtype="single")
    write(tmp_path / "drafts", "w.jsonl", [ans_record(s["slot_id"])])
    slots_path = tmp_path / "slots.jsonl"
    plan.write_slots(slots, slots_path)
    full = write_index(tmp_path / "a.json", [x for x in slots if not x["pilot"]])
    pilot = write_index(tmp_path / "b.json", [x for x in slots if x["pilot"]], plan_sha256="1" * 64)
    code = assemble.run(
        [
            "--slots",
            str(slots_path),
            "--drafts",
            str(tmp_path / "drafts"),
            "--out",
            str(tmp_path / "o.jsonl"),
            "--drafted-on",
            "2026-10-16",
            "--brief-index",
            str(full),
            str(pilot),
        ],
        corpus=corpus,
    )
    assert code == 1 and "different plans" in capsys.readouterr().out
