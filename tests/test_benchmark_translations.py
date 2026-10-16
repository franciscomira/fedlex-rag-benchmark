"""``translations-plan`` / ``translations-apply``: parents, plan, strict drafts, items."""

from __future__ import annotations

import copy
import json
from collections import Counter

import pytest

from benchmark_fixtures import make_corpus, make_valid_items, words
from fedlex_bench.benchmark import baseline, cli, translations, validate
from fedlex_bench.benchmark.bm25 import build_indexes, compute_ranks
from fedlex_bench.benchmark.common import QUERY_LANGS, read_jsonl, write_jsonl
from fedlex_bench.benchmark.translations import TranslationError


@pytest.fixture(scope="module")
def corpus():
    return make_corpus()


@pytest.fixture(scope="module")
def indexes(corpus):
    return build_indexes(corpus)


@pytest.fixture(scope="module")
def facts(corpus):
    return validate.corpus_facts(corpus)


@pytest.fixture(scope="module")
def parquet(tmp_path_factory, corpus):
    path = tmp_path_factory.mktemp("corpus") / "articles.parquet"
    corpus.to_parquet(path, index=False)
    return path


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


def load_all(folder):
    return read_jsonl(folder / "dev.jsonl") + read_jsonl(folder / "test.jsonl")


def make_plan(items, n=24):
    parents = translations.choose_parents(items, n)
    return translations.plan_rows(parents)


def drafts_for(plan, tag="d"):
    return {
        (r["parent_id"], r["target_lang"]): words(f"{tag}/{r['parent_id']}/{r['target_lang']}")
        + "?"
        for r in plan
    }


def write_drafts(folder, drafts):
    rows = [
        {"parent_id": p, "target_lang": lang, "question": q}
        for (p, lang), q in sorted(drafts.items())
    ]
    write_jsonl(folder / "out-1.jsonl", rows[:30])
    write_jsonl(folder / "out-2.jsonl", rows[30:])


# ------------------------------------------------------------------ parents


def test_language_quota_for_24_parents_is_two_two_one_one_in_dev():
    q = translations.per_language_quota(24)
    assert {lang: q[lang]["dev"] for lang in QUERY_LANGS} == {"en": 2, "de": 2, "fr": 1, "it": 1}
    assert all(q[lang]["dev"] + q[lang]["test"] == 6 for lang in QUERY_LANGS)
    assert sum(v["dev"] for v in q.values()) == 6 and sum(v["test"] for v in q.values()) == 18


def test_n_must_be_a_multiple_of_four():
    with pytest.raises(TranslationError):
        translations.per_language_quota(10)


def test_parents_are_six_per_language_six_dev_eighteen_test(items):
    parents = translations.choose_parents(items)
    assert len(parents) == 24
    assert Counter(p["query_lang"] for p in parents) == {lang: 6 for lang in QUERY_LANGS}
    assert Counter(p["split"] for p in parents) == {"dev": 6, "test": 18}
    assert all(p["qtype"] == "single" and not p["is_trap"] for p in parents)
    assert [p["id"] for p in parents] == sorted(p["id"] for p in parents)


def test_parent_choice_is_deterministic_and_ignores_input_order(items):
    first = translations.choose_parents(items)
    again = translations.choose_parents(list(reversed(items)))
    assert [p["id"] for p in first] == [p["id"] for p in again]


def test_excluded_parents(items):
    base = {p["id"] for p in translations.choose_parents(items)}
    victim = sorted(base)[0]
    flagged = copy.deepcopy(items)
    next(i for i in flagged if i["id"] == victim)["provenance"]["expert_flag"] = "wrong"
    assert victim not in {p["id"] for p in translations.choose_parents(flagged)}

    rejected = copy.deepcopy(items)
    next(i for i in rejected if i["id"] == victim).update(
        review_status="rejected", reviewed_by="x", reviewed_at="2026-10-20"
    )
    assert victim not in {p["id"] for p in translations.choose_parents(rejected)}

    assert victim not in {
        p["id"] for p in translations.choose_parents(items, machine_rejected={victim})
    }

    with_translation = copy.deepcopy(items)
    parent = next(i for i in with_translation if i["id"] == victim)
    child = copy.deepcopy(parent)
    child.update(id=f"{victim}-xx", translation_of=victim)
    with_translation.append(child)
    chosen = {p["id"] for p in translations.choose_parents(with_translation)}
    assert child["id"] not in chosen


def small_item(n, lang="de", split="test", act="220", status="unreviewed", qtype="single"):
    return {
        "id": f"v1-{n:04d}",
        "qtype": qtype,
        "is_trap": qtype == "trap",
        "query_lang": lang,
        "split": split,
        "source_act": act,
        "review_status": status,
        "translation_of": None,
        "provenance": {},
        "question": f"q{n}",
    }


def test_preference_is_sr_220_then_human_reviewed_but_neither_is_required():
    pool = [
        small_item(1, act="210", status="accepted"),
        small_item(2, act="220", status="unreviewed"),
        small_item(3, act="220", status="edited"),
        small_item(4, act="210", status="unreviewed"),
    ]
    # one language/split cell only: n=4 gives 1 dev + 1 test parent per language slot; build a
    # pool that has exactly the cells needed and look at the test cell of "de"
    filler = []
    n = 10
    for lang in QUERY_LANGS:
        for split, count in (("dev", 2), ("test", 5)):
            if lang == "de" and split == "test":
                continue
            for _ in range(count):
                filler.append(small_item(n, lang=lang, split=split))
                n += 1
    chosen = translations.choose_parents(pool + filler, 4)
    de_test = [p for p in chosen if p["query_lang"] == "de" and p["split"] == "test"]
    assert [p["id"] for p in de_test] == ["v1-0003"]  # SR 220 and human reviewed wins
    pool2 = [small_item(1, act="210"), small_item(2, act="220", status="accepted")]
    chosen2 = translations.choose_parents(pool2 + filler, 4)
    assert [p["id"] for p in chosen2 if p["query_lang"] == "de" and p["split"] == "test"] == [
        "v1-0002"
    ]
    # nothing reviewed and no SR 220 at all: still chosen
    chosen3 = translations.choose_parents([small_item(1, act="210")] + filler, 4)
    assert "v1-0001" in {p["id"] for p in chosen3}


def test_not_enough_candidates_is_an_error(items):
    few = [i for i in items if i["query_lang"] != "it"]
    with pytest.raises(TranslationError, match="query_lang=it"):
        translations.choose_parents(few)


# ------------------------------------------------------------------ plan files


def test_plan_has_72_sorted_rows_with_the_three_other_languages(items):
    plan = make_plan(items)
    assert len(plan) == 72
    assert plan == sorted(plan, key=lambda r: (r["parent_id"], r["target_lang"]))
    by_parent = {}
    for r in plan:
        assert r["target_lang"] != r["parent_query_lang"]
        by_parent.setdefault(r["parent_id"], set()).add(r["target_lang"])
    assert len(by_parent) == 24
    assert all(len(v) == 3 for v in by_parent.values())
    assert set(plan[0]) == {"parent_id", "target_lang", "parent_question", "parent_query_lang"}


def test_translator_inputs_are_chunks_of_twelve_without_gold(items):
    chunks = translations.input_chunks(make_plan(items))
    assert [len(c) for c in chunks] == [12] * 6
    for row in chunks[0]:
        assert set(row) == {"id", "source_lang", "target_lang", "question"}
        assert row["id"].endswith("-" + row["target_lang"])


def test_write_plan_is_idempotent_refuses_a_different_plan_and_force_overwrites(items, tmp_path):
    out = tmp_path / "translations"
    plan = make_plan(items)
    translations.write_plan(plan, out)
    first = (out / "plan.jsonl").read_bytes()
    translations.write_plan(plan, out)
    assert (out / "plan.jsonl").read_bytes() == first
    assert sorted(p.name for p in out.glob("in-*.jsonl")) == [f"in-{k}.jsonl" for k in range(1, 7)]
    with pytest.raises(TranslationError, match="--force"):
        translations.write_plan(plan[:-15], out)
    assert (out / "plan.jsonl").read_bytes() == first
    translations.write_plan(plan[:-15], out, force=True)
    assert len(read_jsonl(out / "plan.jsonl")) == 57
    assert not (out / "in-6.jsonl").exists()  # stale chunk removed


def test_write_plan_dry_run_writes_nothing(items, tmp_path):
    out = tmp_path / "translations"
    translations.write_plan(make_plan(items), out, dry_run=True)
    assert not out.exists()


def test_plan_command(selected, tmp_path, capsys):
    out = tmp_path / "translations"
    argv = ["translations-plan", "--dir", str(selected), "--out-dir", str(out)]
    argv += ["--machine-dir", str(tmp_path / "none")]
    assert cli.main(argv) == 0
    assert "wrote 72 plan rows from 24 parents" in capsys.readouterr().out
    assert len(read_jsonl(out / "plan.jsonl")) == 72
    assert cli.main(argv + ["--n", "8"]) == 1  # a different plan needs --force
    assert cli.main(argv + ["--n", "8", "--force"]) == 0
    assert len(read_jsonl(out / "plan.jsonl")) == 24


def test_machine_rejects_are_read_from_the_verdict_files(tmp_path):
    folder = tmp_path / "machine"
    write_jsonl(
        folder / "claude-judge-1.jsonl",
        [
            {"id": "v1-0001", "verdict": "approve", "issues": [], "reason": "ok"},
            {"id": "v1-0002", "verdict": "reject", "issues": [], "reason": "bad"},
        ],
    )
    assert translations.machine_rejected_ids(folder) == {"v1-0002"}
    assert translations.machine_rejected_ids(tmp_path / "missing") == set()


# ------------------------------------------------------------------ drafts


@pytest.fixture
def plan(items):
    return make_plan(items)


def test_drafts_must_answer_every_plan_row_exactly_once(plan, tmp_path):
    drafts = drafts_for(plan)
    write_drafts(tmp_path, drafts)
    assert translations.load_drafts(tmp_path, plan) == drafts


def test_missing_duplicate_unknown_and_extra_keys_are_rejected(plan, tmp_path):
    drafts = drafts_for(plan)
    rows = [{"parent_id": p, "target_lang": lang, "question": q} for (p, lang), q in drafts.items()]
    cases = {
        "missing": rows[:-1],
        "duplicate": [*rows, rows[0]],
        "unknown": [*rows, {"parent_id": "v1-9999", "target_lang": "fr", "question": "x"}],
        "extra key": [dict(rows[0], gold_article_ids=["220:1"]), *rows[1:]],
        "empty": [dict(rows[0], question="  "), *rows[1:]],
        "too long": [dict(rows[0], question="x" * 401), *rows[1:]],
    }
    for name, bad in cases.items():
        folder = tmp_path / name.replace(" ", "_")
        write_jsonl(folder / "o.jsonl", bad)
        with pytest.raises(TranslationError):
            translations.load_drafts(folder, plan)


def test_invalid_json_and_missing_folder_are_rejected(plan, tmp_path):
    (tmp_path / "o.jsonl").write_text("{not json\n", encoding="utf-8")
    with pytest.raises(TranslationError, match="invalid JSON"):
        translations.load_drafts(tmp_path, plan)
    with pytest.raises(TranslationError, match="does not exist"):
        translations.load_drafts(tmp_path / "nope", plan)


def test_id_may_stand_in_for_parent_and_language(plan, tmp_path):
    drafts = drafts_for(plan)
    rows = [
        {"id": f"{p}-{lang}", "question": q, "target_lang": lang} for (p, lang), q in drafts.items()
    ]
    write_jsonl(tmp_path / "o.jsonl", rows)
    assert translations.load_drafts(tmp_path, plan) == drafts
    rows[0]["id"] = "v1-0001-xx"
    write_jsonl(tmp_path / "o.jsonl", rows)
    with pytest.raises(TranslationError):
        translations.load_drafts(tmp_path, plan)


# ------------------------------------------------------------------ apply


def apply(plan, drafts, folder, indexes, **kw):
    return translations.apply_translations(
        plan,
        drafts,
        folder,
        indexes,
        plan_sha256="e" * 64,
        translated_on="2026-10-20",
        model_note="test alias",
        **kw,
    )


def test_apply_appends_72_translations_after_the_originals(selected, plan, indexes):
    before = load_all(selected)
    result = apply(plan, drafts_for(plan), selected, indexes)
    assert len(result["added"]) == 72 and result["unchanged"] == []
    for split in ("dev", "test"):
        rows = read_jsonl(selected / f"{split}.jsonl")
        originals = [r for r in rows if not r["translation_of"]]
        trans = [r for r in rows if r["translation_of"]]
        assert rows == originals + trans  # translations come last
        assert [r["id"] for r in trans] == sorted(r["id"] for r in trans)
        assert originals == [b for b in before if b["split"] == split]  # untouched
    assert len([r for r in load_all(selected) if r["translation_of"]]) == 72


def test_translation_items_copy_the_parent_and_recompute_ranks(selected, plan, indexes):
    drafts = drafts_for(plan)
    apply(plan, drafts, selected, indexes)
    rows = {r["id"]: r for r in load_all(selected)}
    row = plan[0]
    t = rows[f"{row['parent_id']}-{row['target_lang']}"]
    p = rows[row["parent_id"]]
    for key in (
        "qtype", "split", "gold_article_ids", "gold_facts", "source_act", "source_lang",
        "difficulty", "is_trap",
    ):  # fmt: skip
        assert t[key] == p[key]
    assert t["query_lang"] == row["target_lang"] != p["query_lang"]
    assert t["question"] == drafts[(row["parent_id"], row["target_lang"])]
    assert t["translation_of"] == p["id"]
    ranks, overlap = compute_ranks(t["question"], t["gold_article_ids"], t["query_lang"], indexes)
    assert t["bm25_rank"] == ranks and t["lexical_overlap"] == overlap
    assert t["review_status"] == "unreviewed"
    assert t["reviewed_by"] is None and t["reviewed_at"] is None and t["review_note"] is None
    prov = t["provenance"]
    assert prov["drafter"] == "translator" and prov["model"] == "test alias"
    assert prov["translated_on"] == prov["drafted_on"] == "2026-10-20"
    assert prov["candidate_sha256"] == p["provenance"].get("candidate_sha256")
    assert prov["plan_sha256"] == "e" * 64


def test_the_result_validates_and_complete_set_checks_ignore_translations(
    selected, plan, indexes, corpus, facts
):
    apply(plan, drafts_for(plan), selected, indexes)
    files = {s: read_jsonl(selected / f"{s}.jsonl") for s in ("dev", "test")}
    report = validate.validate_items(files, corpus, check_ranks=True, indexes=indexes, facts=facts)
    assert report.errors == [], [str(e) for e in report.errors]


def test_apply_twice_changes_nothing_and_a_changed_question_is_an_error(selected, plan, indexes):
    drafts = drafts_for(plan)
    apply(plan, drafts, selected, indexes)
    snapshot = {s: (selected / f"{s}.jsonl").read_bytes() for s in ("dev", "test")}
    again = apply(plan, drafts, selected, indexes)
    assert again["added"] == [] and len(again["unchanged"]) == 72
    assert {s: (selected / f"{s}.jsonl").read_bytes() for s in ("dev", "test")} == snapshot
    other = dict(drafts)
    other[next(iter(other))] = "something else?"
    with pytest.raises(TranslationError, match="different question"):
        apply(plan, other, selected, indexes)


def test_dry_run_writes_nothing(selected, plan, indexes):
    snapshot = {s: (selected / f"{s}.jsonl").read_bytes() for s in ("dev", "test")}
    result = apply(plan, drafts_for(plan), selected, indexes, dry_run=True)
    assert len(result["added"]) == 72
    assert {s: (selected / f"{s}.jsonl").read_bytes() for s in ("dev", "test")} == snapshot


def test_a_parent_whose_question_changed_blocks_the_apply(selected, plan, indexes):
    rows = read_jsonl(selected / "test.jsonl") + read_jsonl(selected / "dev.jsonl")
    pid = plan[0]["parent_id"]
    for r in rows:
        if r["id"] == pid:
            r["question"] += " (edited)"
    put(selected, rows)
    with pytest.raises(TranslationError, match="changed since the plan"):
        apply(plan, drafts_for(plan), selected, indexes)


def test_replace_files_aborts_if_a_file_changed_since_it_was_read(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_bytes(b"old\n")
    b.write_bytes(b"old\n")
    b.write_bytes(b"changed by the review app\n")
    with pytest.raises(TranslationError, match="changed on disk"):
        translations.replace_files({a: (b"old\n", b"new\n"), b: (b"old\n", b"new\n")})
    assert a.read_bytes() == b"old\n"  # nothing was written
    translations.replace_files({a: (b"old\n", b"new\n")})
    assert a.read_bytes() == b"new\n"
    new = tmp_path / "sub" / "c.jsonl"
    translations.replace_files({new: (None, b"x\n")})  # absent file, absent expected
    assert new.read_bytes() == b"x\n"
    with pytest.raises(TranslationError):
        translations.replace_files({new: (None, b"y\n")})


def test_apply_command(selected, plan, parquet, tmp_path, capsys):
    plan_path = tmp_path / "plan.jsonl"
    write_jsonl(plan_path, plan)
    drafts_dir = tmp_path / "drafts"
    write_drafts(drafts_dir, drafts_for(plan))
    argv = [
        "translations-apply", "--plan", str(plan_path), "--drafts", str(drafts_dir),
        "--dir", str(selected), "--parquet", str(parquet), "--date", "2026-10-20",
    ]  # fmt: skip
    assert cli.main([*argv, "--dry-run"]) == 0
    assert "would add 72" in capsys.readouterr().out
    assert len(load_all(selected)) == 170
    assert cli.main(argv) == 0
    assert "added 72 translation(s), 0 already present" in capsys.readouterr().out
    assert len(load_all(selected)) == 170 + 72
    assert cli.main(argv) == 0
    assert "added 0 translation(s), 72 already present" in capsys.readouterr().out
    assert cli.main([*argv[:-2], "--drafts", str(tmp_path / "nope")]) == 1


# ------------------------------------------------------------------ validator and baseline


def _two_items(items):
    parent = next(i for i in items if i["qtype"] == "single" and not i["is_trap"])
    child = copy.deepcopy(parent)
    lang = "fr" if parent["query_lang"] != "fr" else "de"
    child.update(
        id=f"{parent['id']}-{lang}", translation_of=parent["id"], query_lang=lang,
        question=words("child") + "?",
    )  # fmt: skip
    return parent, child


def run_validate(files, corpus, facts, indexes):
    return validate.validate_items(
        files, corpus, check_ranks=False, indexes=indexes, facts=facts, partial=True
    )


def test_validator_accepts_translation_ids_and_rejects_bad_ones(items, corpus, facts, indexes):
    parent, child = _two_items(items)
    files = {"dev": [], "test": []}
    files[parent["split"]] = [parent, child]
    assert "id_format" not in run_validate(files, corpus, facts, indexes).codes
    child["id"] = f"{parent['id']}-xx"
    assert "id_format" in run_validate(files, corpus, facts, indexes).codes
    child["id"] = f"{parent['id']}-{'en' if child['query_lang'] != 'en' else 'de'}"
    assert "id_format" in run_validate(files, corpus, facts, indexes).codes  # lang mismatch
    orig = copy.deepcopy(parent)
    orig["translation_of"] = None
    bad = copy.deepcopy(orig)
    bad["id"] = f"{orig['id']}-fr"  # an original may not use the translation id form
    files[parent["split"]] = [orig, bad]
    assert "id_format" in run_validate(files, corpus, facts, indexes).codes


def test_translations_must_come_after_the_originals(items, corpus, facts, indexes):
    parent, child = _two_items(items)
    other = next(
        i for i in items
        if i["split"] == parent["split"] and i["id"] > parent["id"] and not i["is_trap"]
    )  # fmt: skip
    ok = {"dev": [], "test": []}
    ok[parent["split"]] = [parent, other, child]
    assert "id_order" not in run_validate(ok, corpus, facts, indexes).codes
    bad = {"dev": [], "test": []}
    bad[parent["split"]] = [parent, child, other]
    assert "id_order" in run_validate(bad, corpus, facts, indexes).codes


def test_baseline_leaves_translations_out_unless_asked(items, indexes):
    parent, child = _two_items(items)
    parent["split"] = child["split"] = "dev"
    default = baseline.summarise([parent, child], baseline.build_all_indexes(make_corpus()))
    assert default["n_items"] == 1 and default["n_translations"] == 1
    assert default["translations_included"] is False
    with_t = baseline.summarise(
        [parent, child], baseline.build_all_indexes(make_corpus()), include_translations=True
    )
    assert with_t["n_items"] == 2 and with_t["n_translations"] == 1
    assert with_t["translations_included"] is True
    assert json.dumps(default["recall_at_10"])  # serialisable


# ------------------------------------------------------------------ parents that became unusable


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (lambda it: it["provenance"].update(expert_flag=True), "expert-flagged"),
        (lambda it: it.update(review_status="rejected"), "human-rejected"),
    ],
)
def test_apply_skips_translations_of_a_flagged_or_rejected_parent(
    items, selected, plan, indexes, change, reason, capsys
):
    parent_id = plan[0]["parent_id"]
    parent = next(i for i in items if i["id"] == parent_id)
    assert translations.parent_block_reason(parent) is None
    change(parent)
    assert translations.parent_block_reason(parent) == reason
    put(selected, items)
    result = apply(plan, drafts_for(plan), selected, indexes)
    gone = {f"{parent_id}-{r['target_lang']}" for r in plan if r["parent_id"] == parent_id}
    assert len(gone) == 3 and set(result["skipped"]) == gone
    assert len(result["added"]) == 72 - 3 and not gone & set(result["added"])
    written = {r["id"] for r in load_all(selected)}
    assert not gone & written


def test_apply_command_reports_skipped_translations(
    items, selected, plan, parquet, tmp_path, capsys
):
    parent = next(i for i in items if i["id"] == plan[0]["parent_id"])
    parent["provenance"]["expert_flag"] = True
    put(selected, items)
    plan_path = tmp_path / "plan.jsonl"
    write_jsonl(plan_path, plan)
    drafts = tmp_path / "drafts"
    write_drafts(drafts, drafts_for(plan))
    code = cli.main(
        [
            "translations-apply",
            "--plan",
            str(plan_path),
            "--drafts",
            str(drafts),
            "--dir",
            str(selected),
            "--parquet",
            str(parquet),
            "--date",
            "2026-10-20",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0 and out.count("skipped") == 3 and "added 69 translation(s)" in out
