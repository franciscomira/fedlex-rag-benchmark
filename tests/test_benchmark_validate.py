"""``benchmark validate``: one violation per test on a small synthetic benchmark."""

from __future__ import annotations

import copy
import json

import pytest

from benchmark_fixtures import make_corpus, make_valid_items, words
from fedlex_bench.benchmark import validate
from fedlex_bench.benchmark.bm25 import build_indexes, compute_ranks
from fedlex_bench.benchmark.common import ACT_QUOTA, split_of


@pytest.fixture(scope="module")
def corpus():
    return make_corpus()


@pytest.fixture(scope="module")
def indexes(corpus):
    return build_indexes(corpus)


@pytest.fixture(scope="module")
def valid_items(corpus):
    return make_valid_items(corpus)


@pytest.fixture
def items(valid_items):
    return copy.deepcopy(valid_items)


def files_of(items):
    return {
        "dev": [i for i in items if i["split"] == "dev"],
        "test": [i for i in items if i["split"] == "test"],
    }


@pytest.fixture(scope="module")
def facts(corpus):
    return validate.corpus_facts(corpus)


@pytest.fixture(autouse=True)
def _reuse_corpus_facts(facts, monkeypatch):
    """``benchmark_eligible`` takes about half a second: compute it once for the module."""
    real = validate.validate_items

    def with_facts(files, df, **kw):
        kw.setdefault("facts", facts)
        return real(files, df, **kw)

    monkeypatch.setattr(validate, "validate_items", with_facts)


def check(items, corpus, indexes, **kw):
    kw.setdefault("check_ranks", False)
    return validate.validate_items(files_of(items), corpus, indexes=indexes, **kw)


def answerable(items):
    return [i for i in items if not i["is_trap"]]


def traps(items):
    return [i for i in items if i["is_trap"]]


def singles(items, act=None):
    return [
        i
        for i in answerable(items)
        if i["qtype"] == "single" and (act is None or i["source_act"] == act)
    ]


def crosses(items):
    return [i for i in answerable(items) if i["qtype"] == "cross_ref"]


def with_rank(items):
    """An answerable item whose German BM25 rank exists (a shared token with its article)."""
    return next(i for i in answerable(items) if i["bm25_rank"]["de"] is not None)


def only(report, code):
    assert report.codes == {code}, [str(e) for e in report.errors]


# ------------------------------------------------------------------ the happy path


def test_a_valid_benchmark_has_no_errors_and_ranks_match(items, corpus, indexes):
    report = check(items, corpus, indexes, check_ranks=True)
    assert report.errors == [], [str(e) for e in report.errors]
    assert len(items) == 170 and len(answerable(items)) == 150


def test_a_valid_benchmark_passes_in_freeze_mode_once_reviewed(items, corpus, indexes):
    for it in items:
        it.update(review_status="accepted", reviewed_by="Francisco Mira", reviewed_at="2026-10-20")
    assert check(items, corpus, indexes, freeze=True).errors == []
    items[0].update(review_status="edited")
    assert check(items, corpus, indexes, freeze=True).errors == []


def test_every_check_code_is_documented():
    doc = validate.__doc__
    for code in (
        "schema",
        "gold_missing",
        "split_rule",
        "gold_reused",
        "freeze_status",
        "rank_mismatch",
    ):
        assert code in doc


# ------------------------------------------------------------------ schema and types


def test_schema_missing_key(items, corpus, indexes):
    del items[3]["difficulty"]
    report = check(items, corpus, indexes, partial=True)
    only(report, "schema")
    assert items[3]["id"] in str(report.errors[0])


def test_schema_extra_key(items, corpus, indexes):
    items[3]["gold_ids"] = []
    only(check(items, corpus, indexes, partial=True), "schema")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("split", "train"),
        ("qtype", "multi"),
        ("is_trap", "no"),
        ("query_lang", "es"),
        ("question", ""),
        ("source_lang", "en"),
        ("gold_article_ids", "220:1"),
        ("gold_facts", [{"text": "x"}]),
        ("gold_facts", ["x"]),
        ("difficulty", "trivial"),
        ("lexical_overlap", 1.5),
        ("lexical_overlap", True),
        ("bm25_rank", {"de": 1, "fr": 2}),
        ("bm25_rank", {"de": 1, "fr": 2, "it": 0}),
        ("trap_type", "misc"),
        ("translation_of", 7),
        ("provenance", "x"),
        ("review_status", "maybe"),
        ("reviewed_by", 5),
    ],
)
def test_schema_type_and_enum_errors(items, corpus, indexes, key, value):
    files = files_of(items)  # placed before the mutation: a bad split must not move the item
    items[3][key] = value
    report = validate.validate_items(files, corpus, partial=True, check_ranks=False)
    assert "schema" in report.codes, key
    assert any(items[3]["id"] in str(e) for e in report.errors)


# ------------------------------------------------------------------ consistency


def test_is_trap_must_match_qtype(items, corpus, indexes):
    items[3]["is_trap"] = True
    report = check(items, corpus, indexes, partial=True)
    only(report, "is_trap")


def test_ids_must_be_sorted_unique_and_well_formed(items, corpus, indexes):
    dev = files_of(items)["dev"]
    dev[0]["id"], dev[1]["id"] = dev[1]["id"], dev[0]["id"]
    assert "id_order" in check(items, corpus, indexes, partial=True).codes


def test_duplicate_ids(items, corpus, indexes):
    dev = files_of(items)["dev"]
    dev[1]["id"] = dev[0]["id"]
    codes = check(items, corpus, indexes, partial=True).codes
    assert "id_duplicate" in codes


def test_id_format(items, corpus, indexes):
    items[0]["id"] = "x-0001"
    assert "id_format" in check(items, corpus, indexes, partial=True).codes


def test_item_must_sit_in_the_file_of_its_split(items, corpus, indexes):
    files = files_of(items)
    moved = files["dev"].pop(0)
    files["test"].insert(0, moved)
    report = validate.validate_items(files, corpus, check_ranks=False, partial=True)
    assert "file_split" in report.codes


def test_gold_count_by_qtype(items, corpus, indexes):
    single = next(i for i in answerable(items))
    other = next(i for i in answerable(items) if i is not single)["gold_article_ids"][0]
    single["gold_article_ids"].append(other)
    assert "gold_count" in check(items, corpus, indexes, partial=True).codes
    single["gold_article_ids"] = []
    assert "gold_count" in check(items, corpus, indexes, partial=True).codes


def test_cross_ref_needs_two_distinct_gold_ids(items, corpus, indexes):
    it = answerable(items)[0]
    it["qtype"] = "cross_ref"
    assert "gold_count" in check(items, corpus, indexes, partial=True).codes


def test_gold_must_exist_in_all_three_languages(items, corpus, indexes):
    it = answerable(items)[0]
    it["gold_article_ids"] = ["220:99999"]
    it["gold_facts"][0]["article_id"] = "220:99999"
    assert "gold_missing" in check(items, corpus, indexes, partial=True).codes


def test_gold_must_be_benchmark_eligible(items, corpus, indexes):
    it = singles(items, "220")[0]
    for bad in ("220:r1", "220:s1", "220:u1/1"):  # repealed, too short, transitional
        it["gold_article_ids"] = [bad]
        it["gold_facts"][0]["article_id"] = bad
        report = check(items, corpus, indexes, partial=True)
        assert report.codes & {"gold_ineligible", "gold_missing"}, bad


def test_source_act_must_be_the_act_of_the_primary_gold(items, corpus, indexes):
    it = next(i for i in answerable(items) if i["source_act"] == "220")
    it["source_act"] = "210"
    report = check(items, corpus, indexes, partial=True)
    assert "source_act" in report.codes


def test_gold_fact_must_cite_a_gold_article(items, corpus, indexes):
    it = answerable(items)[0]
    it["gold_facts"][0]["article_id"] = "220:5"
    only(check(items, corpus, indexes, partial=True), "fact_article")


def test_gold_fact_count(items, corpus, indexes):
    it = answerable(items)[0]
    fact = dict(it["gold_facts"][0])
    it["gold_facts"] = []
    assert "fact_count" in check(items, corpus, indexes, partial=True).codes
    it["gold_facts"] = [dict(fact) for _ in range(4)]  # at most 3 for a single item
    assert "fact_count" in check(items, corpus, indexes, partial=True).codes
    it["gold_facts"] = [dict(fact) for _ in range(3)]
    assert "fact_count" not in check(items, corpus, indexes, partial=True).codes


# ------------------------------------------------------------------ split rule


def test_split_must_follow_the_hash_rule(items, corpus, indexes):
    it = answerable(items)[0]
    other = "test" if it["split"] == "dev" else "dev"
    files = files_of(items)
    files[it["split"]].remove(it)
    it["split"] = other
    files[other].append(it)
    files[other].sort(key=lambda x: x["id"])
    report = validate.validate_items(files, corpus, check_ranks=False, partial=True)
    assert "split_rule" in report.codes
    assert split_of(it["gold_article_ids"][0]) != other


def test_no_gold_in_both_splits(items, corpus, indexes):
    dev_it = next(i for i in answerable(items) if i["split"] == "dev")
    test_it = next(i for i in answerable(items) if i["split"] == "test")
    test_it["gold_article_ids"] = list(dev_it["gold_article_ids"])
    test_it["gold_facts"][0]["article_id"] = dev_it["gold_article_ids"][0]
    codes = check(items, corpus, indexes, partial=True).codes
    assert "gold_in_both_splits" in codes and "gold_reused" in codes


def test_a_gold_article_may_be_used_by_one_item_only(items, corpus, indexes):
    a, b = [i for i in answerable(items) if i["split"] == "test"][:2]
    b["gold_article_ids"] = list(a["gold_article_ids"])
    b["gold_facts"][0]["article_id"] = a["gold_article_ids"][0]
    report = check(items, corpus, indexes, partial=True)
    assert "gold_reused" in report.codes and "gold_in_both_splits" not in report.codes


# ------------------------------------------------------------------ translations


def make_translation(items, original, new_id="v1-9001", lang=None):
    t = copy.deepcopy(original)
    t["id"] = new_id
    t["translation_of"] = original["id"]
    t["query_lang"] = lang or ("fr" if original["query_lang"] != "fr" else "de")
    t["question"] = words(f"translation/{new_id}") + "?"
    items.append(t)
    return t


def test_translations_may_share_gold_with_their_original(items, corpus, indexes):
    original = next(i for i in answerable(items) if i["split"] == "test" and i["qtype"] == "single")
    make_translation(items, original, "v1-9001")
    make_translation(
        items, original, "v1-9002", lang="it" if original["query_lang"] != "it" else "de"
    )
    report = check(items, corpus, indexes)
    assert report.errors == [], [str(e) for e in report.errors]


def test_translations_do_not_count_in_balance_and_quotas(items, corpus, indexes):
    originals = [i for i in answerable(items) if i["split"] == "test" and i["query_lang"] == "de"][
        :8
    ]
    for n, original in enumerate(originals):
        make_translation(items, original, f"v1-{9000 + n}", lang="fr")
    assert check(items, corpus, indexes).errors == []


def test_translation_of_an_unknown_item(items, corpus, indexes):
    original = next(i for i in answerable(items) if i["split"] == "test")
    t = make_translation(items, original)
    t["translation_of"] = "v1-5555"
    assert "translation" in check(items, corpus, indexes).codes


def test_translation_must_keep_gold_split_and_change_language(items, corpus, indexes):
    originals = [i for i in answerable(items) if i["split"] == "test"]
    t = make_translation(items, originals[0])
    t["gold_article_ids"] = list(originals[1]["gold_article_ids"])
    t["gold_facts"][0]["article_id"] = originals[1]["gold_article_ids"][0]
    codes = check(items, corpus, indexes).codes
    assert "translation" in codes and "gold_reused" in codes
    items.remove(t)
    t2 = make_translation(items, originals[0])
    t2["query_lang"] = originals[0]["query_lang"]
    assert "translation" in check(items, corpus, indexes).codes


def test_no_translation_of_a_translation(items, corpus, indexes):
    original = next(i for i in answerable(items) if i["split"] == "test")
    t1 = make_translation(items, original, "v1-9001")
    make_translation(items, t1, "v1-9002", lang="it" if t1["query_lang"] != "it" else "de")
    assert "translation" in check(items, corpus, indexes).codes


# ------------------------------------------------------------------ balance and quotas


def test_language_balance(items, corpus, indexes):
    en = [i for i in answerable(items) if i["query_lang"] == "en"]
    for it in en[:12]:
        it["query_lang"] = "de"
    report = check(items, corpus, indexes)
    assert "lang_balance" in report.codes
    assert any("query_lang=en" in str(e) for e in report.errors)


def test_act_quota_within_one(items, corpus, indexes):
    victims = [i for i in answerable(items) if i["source_act"] == "241"][:2]
    for v in victims:
        items.remove(v)
    report = check(items, corpus, indexes)
    assert "act_quota" in report.codes and any("act=241" in str(e) for e in report.errors)
    items.append(victims[0])
    items.sort(key=lambda i: i["id"])
    assert "act_quota" not in check(items, corpus, indexes).codes  # one short is within tolerance


def test_act_outside_the_table(items, corpus, indexes):
    answerable(items)[0]["source_act"] = "999"
    report = check(items, corpus, indexes)
    assert "act_quota" in report.codes


def test_partial_skips_quota_and_balance_only(items, corpus, indexes):
    for it in [i for i in answerable(items) if i["source_act"] == "220"][:30]:
        items.remove(it)
    assert "act_quota" in check(items, corpus, indexes).codes
    assert "act_quota" not in check(items, corpus, indexes, partial=True).codes
    items[0]["difficulty"] = "bad"
    assert "schema" in check(items, corpus, indexes, partial=True).codes


# ------------------------------------------------------------------ questions


def test_exact_duplicate_questions_after_normalisation(items, corpus, indexes):
    a, b = [i for i in answerable(items) if i["query_lang"] == "de"][:2]
    b["question"] = a["question"].upper().replace("?", " ?!")
    assert "question_duplicate" in check(items, corpus, indexes).codes


def test_near_duplicate_questions_within_a_language(items, corpus, indexes):
    a, b = [i for i in answerable(items) if i["query_lang"] == "fr"][:2]
    b["question"] = a["question"].replace("?", " ok?")
    codes = check(items, corpus, indexes).codes
    assert "question_near_duplicate" in codes and "question_duplicate" not in codes


def test_similar_text_in_different_languages_is_fine(items, corpus, indexes):
    a = next(i for i in answerable(items) if i["query_lang"] == "fr")
    b = next(i for i in answerable(items) if i["query_lang"] == "de")
    b["question"] = a["question"] + " x"
    assert not check(items, corpus, indexes).codes & {
        "question_near_duplicate",
        "question_duplicate",
    }


def test_question_longer_than_400_characters(items, corpus, indexes):
    answerable(items)[0]["question"] = "wort " * 81
    only(check(items, corpus, indexes), "question_length")
    answerable(items)[0]["question"] = ("x" * 399) + "?"
    assert check(items, corpus, indexes).errors == []


# ------------------------------------------------------------------ traps


def test_trap_must_have_no_gold_and_a_reason(items, corpus, indexes):
    t = traps(items)[0]
    t["gold_article_ids"] = ["220:1"]
    assert "trap_fields" in check(items, corpus, indexes).codes
    t["gold_article_ids"] = []
    t["trap_reason"] = " "
    assert "trap_fields" in check(items, corpus, indexes).codes
    t["trap_reason"] = None
    assert "trap_fields" in check(items, corpus, indexes).codes


def test_trap_must_not_carry_answerable_fields(items, corpus, indexes):
    for key, value in (
        ("source_lang", "de"),
        ("source_act", "220"),
        ("difficulty", "easy"),
        ("lexical_overlap", 3),
    ):
        t = copy.deepcopy(traps(items)[0])
        t[key] = value
        probe = [i for i in items if i["id"] != t["id"]] + [t]
        probe.sort(key=lambda i: i["id"])
        assert "trap_fields" in check(probe, corpus, indexes).codes, key


def test_answerable_item_must_not_have_trap_fields(items, corpus, indexes):
    answerable(items)[0]["trap_reason"] = "x"
    assert "answerable_fields" in check(items, corpus, indexes).codes


def test_answerable_needs_source_lang_difficulty_and_ranks(items, corpus, indexes):
    answerable(items)[0]["source_lang"] = None
    answerable(items)[1]["difficulty"] = None
    answerable(items)[2]["bm25_rank"] = None
    report = check(items, corpus, indexes)
    assert sum(1 for e in report.errors if e.check == "answerable_fields") >= 3


# ------------------------------------------------------------------ provenance, review, freeze


def test_provenance_requires_its_keys_and_types(items, corpus, indexes):
    del items[0]["provenance"]["plan_sha256"]
    items[1]["provenance"]["drafted_on"] = "yesterday"
    items[2]["provenance"]["agent_file_sha256"] = "xyz"
    items[3]["provenance"]["critic_rounds"] = -1
    report = check(items, corpus, indexes)
    assert [e.item_id for e in report.errors if e.check == "provenance"].count(items[0]["id"]) == 1
    assert sum(1 for e in report.errors if e.check == "provenance") == 4


def test_missing_agent_file_hash_is_a_warning(items, corpus, indexes):
    items[0]["provenance"]["agent_file_sha256"] = None
    report = check(items, corpus, indexes)
    assert report.errors == [] and "warn_agent_file" in report.warning_codes


def test_unreviewed_items_carry_no_reviewer_fields(items, corpus, indexes):
    items[0]["reviewed_by"] = "Francisco Mira"
    only(check(items, corpus, indexes), "review_fields")


def test_reviewed_items_need_reviewer_and_date(items, corpus, indexes):
    items[0]["review_status"] = "accepted"
    only(check(items, corpus, indexes), "review_fields")


def test_freeze_requires_accepted_or_edited(items, corpus, indexes):
    for it in items:
        it.update(review_status="accepted", reviewed_by="Francisco Mira", reviewed_at="2026-10-20")
    items[0].update(review_status="unreviewed", reviewed_by=None, reviewed_at=None)
    items[1].update(review_status="rejected")
    report = check(items, corpus, indexes, freeze=True)
    assert [e.check for e in report.errors] == ["freeze_status", "freeze_status"]
    assert check(items, corpus, indexes).errors == []  # without --freeze this is allowed


@pytest.mark.parametrize(
    "reviewer", ["claude-opus-5", "question-critic", "Agent 7", "GPT", "AI", "writer"]
)
def test_freeze_rejects_reviewers_that_look_like_agents(items, corpus, indexes, reviewer):
    for it in items:
        it.update(review_status="accepted", reviewed_by="Francisco Mira", reviewed_at="2026-10-20")
    items[0]["reviewed_by"] = reviewer
    report = check(items, corpus, indexes, freeze=True)
    assert [e.check for e in report.errors] == ["freeze_reviewer"]
    assert items[0]["id"] in str(report.errors[0])


def test_freeze_accepts_human_names(items, corpus, indexes):
    for name in ("Francisco Mira", "F. Mira", "fmira", "Maria Gonzalez"):
        for it in items:
            it.update(review_status="accepted", reviewed_by=name, reviewed_at="2026-10-20")
        assert check(items, corpus, indexes, freeze=True).errors == [], name


# ------------------------------------------------------------------ ranks and warnings


def test_stored_ranks_must_match_the_recomputed_ones(items, corpus, indexes):
    it = next(i for i in answerable(items) if i["query_lang"] != "en" and i["bm25_rank"]["de"])
    it["bm25_rank"]["de"] += 1
    if it["query_lang"] == "de":
        it["lexical_overlap"] += 1  # keep the item consistent: only the recomputation can tell
    report = check(items, corpus, indexes, check_ranks=True)
    only(report, "rank_mismatch")
    assert it["id"] in str(report.errors[0])


def test_a_null_rank_must_stay_null_and_a_rank_must_not_be_null(items, corpus, indexes):
    it = next(i for i in answerable(items) if i["bm25_rank"]["fr"] is None)
    it["bm25_rank"]["fr"] = 4
    only(check(items, corpus, indexes, check_ranks=True), "rank_mismatch")
    it["bm25_rank"]["fr"] = None
    other = with_rank(items)
    other["bm25_rank"]["it"] = None
    only(check(items, corpus, indexes, check_ranks=True), "rank_mismatch")


def test_lexical_overlap_is_the_own_language_rank_and_null_for_english(items, corpus, indexes):
    en = next(i for i in answerable(items) if i["query_lang"] == "en")
    assert en["lexical_overlap"] is None
    en["lexical_overlap"] = 3
    assert "answerable_fields" in check(items, corpus, indexes).codes
    en["lexical_overlap"] = None
    de = next(i for i in answerable(items) if i["query_lang"] == "de")
    de["lexical_overlap"] = 9 if de["bm25_rank"]["de"] != 9 else 8
    assert "answerable_fields" in check(items, corpus, indexes).codes


def test_null_ranks_are_valid_schema(items, corpus, indexes):
    assert any(i["bm25_rank"]["de"] is None for i in answerable(items))
    assert any(i["bm25_rank"]["de"] is not None for i in answerable(items))
    assert check(items, corpus, indexes, check_ranks=True).errors == []


def test_rank_check_can_be_skipped_for_speed(items, corpus, indexes):
    it = with_rank(items)
    it["bm25_rank"]["de"] += 1
    assert "rank_mismatch" not in check(items, corpus, indexes, check_ranks=False).codes
    assert "rank_mismatch" in check(items, corpus, indexes, check_ranks=True).codes


def test_freeze_always_recomputes_the_ranks(items, corpus, indexes):
    for it in items:
        it.update(review_status="accepted", reviewed_by="Francisco Mira", reviewed_at="2026-10-20")
    with_rank(items)["bm25_rank"]["de"] += 1
    report = check(items, corpus, indexes, freeze=True, check_ranks=False)
    assert "rank_mismatch" in report.codes


def test_warning_when_same_language_rank_one_is_too_common(items, corpus, indexes):
    own = [i for i in answerable(items) if i["query_lang"] != "en"]
    for it in own[: int(len(own) * 0.5)]:
        it["bm25_rank"][it["query_lang"]] = 1
        it["lexical_overlap"] = 1
    report = check(items, corpus, indexes)
    assert "warn_same_lang_rank1" in report.warning_codes and report.errors == []


def test_no_rank_one_warning_below_the_limit(items, corpus, indexes):
    for it in answerable(items):
        for lang in it["bm25_rank"]:
            it["bm25_rank"][lang] = 7
        it["lexical_overlap"] = None if it["query_lang"] == "en" else 7
    assert "warn_same_lang_rank1" not in check(items, corpus, indexes).warning_codes


def test_warning_for_small_language_pair_cells(items, corpus, indexes):
    assert not [w for w in check(items, corpus, indexes).warnings if w.check == "warn_pair_cell"]
    cell = [i for i in answerable(items) if i["query_lang"] == "de" and i["source_lang"] == "it"]
    for it in cell[3:]:
        it["source_lang"] = "fr"  # leave 3 items in the (de, it) cell
    report = check(items, corpus, indexes)
    cells = [w for w in report.warnings if w.check == "warn_pair_cell"]
    assert [w.item_id for w in cells] == ["query=de source=it"]
    assert "fewer than 5" in cells[0].message


def test_warning_when_the_dev_language_counts_are_uneven(items, corpus, indexes):
    dev = [i for i in answerable(items) if i["split"] == "dev"]
    for n, it in enumerate(dev):  # an even spread first: 10 per language
        it["query_lang"] = ("en", "de", "fr", "it")[n % 4]
    assert "warn_dev_lang" not in check(items, corpus, indexes).warning_codes
    for it in dev[:6]:
        it["query_lang"] = "en"
    report = check(items, corpus, indexes)
    assert "warn_dev_lang" in report.warning_codes
    assert any("en=" in w.message for w in report.warnings if w.check == "warn_dev_lang")


# ------------------------------------------------------------------ completeness of the set


def test_total_counts_of_answerable_items_and_traps(items, corpus, indexes):
    items.remove(singles(items, "220")[0])
    report = check(items, corpus, indexes)
    assert "total_count" in report.codes
    assert any("149" in str(e) for e in report.errors)
    assert "total_count" not in check(items, corpus, indexes, partial=True).codes


def test_trap_total_and_trap_languages(items, corpus, indexes):
    removed = traps(items)[0]
    items.remove(removed)
    codes = check(items, corpus, indexes).codes
    assert {"total_count", "trap_lang_count"} <= codes
    # right total, wrong language mix: the copy has another language than the removed trap
    extra = copy.deepcopy(next(t for t in traps(items) if t["query_lang"] != removed["query_lang"]))
    extra["id"] = "v1-9999"
    extra["question"] = words("another trap") + "?"
    items.append(extra)
    codes = check(items, corpus, indexes).codes
    assert "trap_lang_count" in codes and "total_count" not in codes


def test_dev_test_counts_of_the_answerable_items(items, corpus, indexes):
    # one dev item less, one test item more: the total stays 150
    dev_item = next(i for i in singles(items, "220") if i["split"] == "dev")
    items.remove(dev_item)
    facts = validate.corpus_facts(corpus)
    used = {g for i in items for g in i["gold_article_ids"]}
    free = next(
        a
        for a in sorted(facts.pool)
        if a.startswith("220:") and split_of(a) == "test" and a not in used
    )
    extra = copy.deepcopy(singles(items, "220")[0])
    extra.update(id="v1-9998", split="test", gold_article_ids=[free], question=words("extra") + "?")
    extra["gold_facts"] = [{"text": "A fact.", "article_id": free}]
    items.append(extra)
    codes = check(items, corpus, indexes).codes
    assert "split_count" in codes and "total_count" not in codes


def test_cross_ref_count_within_one(items, corpus, indexes):
    def demote(item):
        item["qtype"] = "single"
        item["gold_article_ids"] = item["gold_article_ids"][:1]
        item["gold_facts"] = item["gold_facts"][:1]

    crosses_ = crosses(items)
    assert len(crosses_) == 22
    demote(crosses_[0])
    assert "cross_ref_count" not in check(items, corpus, indexes).codes  # 21: within tolerance
    demote(crosses_[1])
    assert "cross_ref_count" in check(items, corpus, indexes).codes  # 20
    assert "cross_ref_count" not in check(items, corpus, indexes, partial=True).codes


def test_a_cross_ref_secondary_must_be_in_the_same_act(items, corpus, indexes):
    cross = crosses(items)[0]
    primary_act = cross["source_act"]
    other = next(
        a
        for a in sorted(validate.corpus_facts(corpus).pool)
        if not a.startswith(primary_act + ":") and split_of(a) == cross["split"]
    )
    cross["gold_article_ids"][1] = other
    cross["gold_facts"][1]["article_id"] = other
    report = check(items, corpus, indexes, partial=True)
    assert "cross_ref_act" in report.codes


def test_cross_ref_secondary_must_pass_every_gold_check(items, corpus, indexes):
    cross = crosses(items)[0]
    cross["gold_article_ids"][1] = f"{cross['source_act']}:r1"  # repealed: not eligible
    cross["gold_facts"][1]["article_id"] = f"{cross['source_act']}:r1"
    codes = check(items, corpus, indexes, partial=True).codes
    assert codes & {"gold_ineligible", "gold_missing"}
    cross["gold_article_ids"][1] = cross["gold_article_ids"][0]  # the same article twice
    assert "gold_count" in check(items, corpus, indexes, partial=True).codes


def test_a_cross_ref_secondary_may_not_be_used_by_another_item(items, corpus, indexes):
    a, b = crosses(items)[:2]
    b["gold_article_ids"][1] = a["gold_article_ids"][1]
    b["gold_facts"][1]["article_id"] = a["gold_article_ids"][1]
    report = check(items, corpus, indexes, partial=True)
    assert "gold_reused" in report.codes


def test_translation_parent_must_be_single_and_translations_are_ranked(items, corpus, indexes):
    cross = crosses(items)[0]
    t = make_translation(items, cross, "v1-9001")
    report = check(items, corpus, indexes)
    assert "translation" in report.codes and any("single" in str(e) for e in report.errors)
    items.remove(t)
    single = next(i for i in singles(items) if i["split"] == "test")
    t = make_translation(items, single, "v1-9002")
    ranks, overlap = compute_ranks(t["question"], t["gold_article_ids"], t["query_lang"], indexes)
    t["bm25_rank"], t["lexical_overlap"] = ranks, overlap
    assert check(items, corpus, indexes, check_ranks=True).errors == []
    t["bm25_rank"] = {"de": 2, "fr": 2, "it": 2}  # a wrong rank of a translation is found too
    t["lexical_overlap"] = None if t["query_lang"] == "en" else 2
    assert "rank_mismatch" in check(items, corpus, indexes, check_ranks=True).codes


def test_a_question_with_a_unicode_line_separator_is_read_correctly(tmp_path, items, corpus):
    victim = singles(items)[0]
    ranks, overlap = compute_ranks(
        victim["question"].rstrip("?") + "\u2028 mit Zeilentrenner?",
        victim["gold_article_ids"],
        victim["query_lang"],
        build_indexes(corpus),
    )
    victim["question"] = victim["question"].rstrip("?") + "\u2028 mit Zeilentrenner?"
    victim["bm25_rank"], victim["lexical_overlap"] = ranks, overlap
    write_dir(tmp_path, items)
    report = validate.validate_dir(tmp_path, corpus, check_ranks=True)
    assert report.errors == [], [str(e) for e in report.errors]


# ------------------------------------------------------------------ files and command


def write_dir(directory, items):
    directory.mkdir(parents=True, exist_ok=True)
    for split, rows in files_of(items).items():
        text = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)
        (directory / f"{split}.jsonl").write_bytes(text.encode("utf-8"))


def test_validate_dir_reads_both_files(tmp_path, items, corpus):
    write_dir(tmp_path, items)
    assert validate.validate_dir(tmp_path, corpus, check_ranks=False).errors == []


def test_missing_file_and_bad_json_and_line_endings(tmp_path, items, corpus):
    write_dir(tmp_path, items)
    (tmp_path / "test.jsonl").unlink()
    assert "files" in validate.validate_dir(tmp_path, corpus, check_ranks=False, partial=True).codes
    write_dir(tmp_path, items)
    (tmp_path / "dev.jsonl").write_text("{broken\n", encoding="utf-8")
    assert "json" in validate.validate_dir(tmp_path, corpus, check_ranks=False, partial=True).codes
    write_dir(tmp_path, items)
    raw = (tmp_path / "dev.jsonl").read_bytes().replace(b"\n", b"\r\n")
    (tmp_path / "dev.jsonl").write_bytes(raw)
    assert "files" in validate.validate_dir(tmp_path, corpus, check_ranks=False, partial=True).codes


def test_command_exit_codes_and_output(tmp_path, items, corpus, capsys):
    write_dir(tmp_path, items)
    assert validate.run(["--dir", str(tmp_path), "--skip-rank-check"], corpus=corpus) == 0
    out = capsys.readouterr().out
    assert "0 error(s)" in out and "WARNING" in out  # warnings do not change the exit code

    broken = copy.deepcopy(items)
    victim = answerable(broken)[0]
    victim["gold_facts"][0]["article_id"] = "220:5"
    write_dir(tmp_path / "bad", broken)
    assert validate.run(["--dir", str(tmp_path / "bad"), "--skip-rank-check"], corpus=corpus) == 1
    out = capsys.readouterr().out
    assert victim["id"] in out and "fact_article" in out and "ERROR" in out


def test_command_freeze_flag(tmp_path, items, corpus, capsys):
    write_dir(tmp_path, items)
    assert validate.run(["--dir", str(tmp_path), "--freeze"], corpus=corpus) == 1
    assert "freeze_status" in capsys.readouterr().out
    for it in items:
        it.update(review_status="accepted", reviewed_by="Francisco Mira", reviewed_at="2026-10-20")
    write_dir(tmp_path / "ok", items)
    assert validate.run(["--dir", str(tmp_path / "ok"), "--freeze"], corpus=corpus) == 0


def test_command_rejects_skip_rank_check_with_freeze(tmp_path, items, corpus, capsys):
    write_dir(tmp_path, items)
    with pytest.raises(SystemExit) as exc:
        validate.run(["--dir", str(tmp_path), "--freeze", "--skip-rank-check"], corpus=corpus)
    assert exc.value.code == 2
    assert "always recomputes" in capsys.readouterr().err


def test_command_partial_flag(tmp_path, items, corpus):
    short = [i for i in items if i["source_act"] != "220"]
    write_dir(tmp_path, short)
    assert validate.run(["--dir", str(tmp_path), "--skip-rank-check"], corpus=corpus) == 1
    assert (
        validate.run(["--dir", str(tmp_path), "--skip-rank-check", "--partial"], corpus=corpus) == 0
    )


def test_quota_table_is_the_spec_table():
    assert sum(ACT_QUOTA.values()) == 150 and ACT_QUOTA["220"] == 90


def test_recomputing_ranks_in_fixture_matches_compute_ranks(items, corpus, indexes):
    it = answerable(items)[0]
    ranks, overlap = compute_ranks(
        it["question"], it["gold_article_ids"], it["query_lang"], indexes
    )
    assert it["bm25_rank"] == ranks and it["lexical_overlap"] == overlap
