"""The draft plan (``benchmark plan``): quotas, splits, languages, cross_ref, traps, pilot."""

from __future__ import annotations

import hashlib
from collections import Counter

import pytest

from benchmark_fixtures import SIZES, make_corpus
from fedlex_bench.benchmark import plan
from fedlex_bench.benchmark.common import (
    ACT_QUOTA,
    DEV_ANSWERABLE,
    LANG_FINAL,
    TEST_ANSWERABLE,
    TRAP_TYPES,
    draft_count,
    largest_remainder,
    split_of,
)
from fedlex_bench.eligibility import benchmark_eligible

ACTS = list(ACT_QUOTA)


@pytest.fixture(scope="module")
def corpus():
    return make_corpus()


@pytest.fixture(scope="module")
def result(corpus):
    return plan.build_plan(corpus)


@pytest.fixture(scope="module")
def slots(result):
    return result.slots


def answerable(slots):
    return [s for s in slots if s["qtype"] != "trap"]


def traps(slots):
    return [s for s in slots if s["qtype"] == "trap"]


def final_prefix(slots):
    """The first ``quota`` slots of every answerable cell, in slot order: what `select` keeps if
    the critic keeps everything."""
    quotas = plan.final_quotas()
    seen: Counter = Counter()
    out = []
    for s in answerable(slots):
        cell = (s["act"], s["split"])
        if seen[cell] < quotas[cell]:
            out.append(s)
        seen[cell] += 1
    return out


# ------------------------------------------------------------------ arithmetic and quotas


def test_largest_remainder_is_exact_and_breaks_ties_by_order():
    assert largest_remainder(10, {"a": 1, "b": 1, "c": 1}) == {"a": 4, "b": 3, "c": 3}
    assert largest_remainder(10, {"a": 1, "b": 1, "c": 1}, order=["c", "b", "a"]) == {
        "a": 3,
        "b": 3,
        "c": 4,
    }
    assert sum(largest_remainder(150, {"a": 90, "b": 14, "c": 46}).values()) == 150
    with pytest.raises(ValueError):
        largest_remainder(5, {"a": 0})


def test_draft_count_is_ceil_of_one_and_a_half_times_the_quota():
    assert [draft_count(q) for q in (1, 2, 3, 4, 5, 90, 24)] == [2, 3, 5, 6, 8, 135, 36]


def test_final_quotas_follow_the_spec_table_and_the_dev_test_totals():
    q = plan.final_quotas()
    for act, quota in ACT_QUOTA.items():
        assert q[(act, "dev")] + q[(act, "test")] == quota
    assert sum(v for (a, s), v in q.items() if s == "dev") == DEV_ANSWERABLE == 40
    assert sum(v for (a, s), v in q.items() if s == "test") == TEST_ANSWERABLE == 110
    assert {a: q[(a, "dev")] for a in ACTS} == {
        "210": 4,
        "220": 24,
        "221.214.1": 1,
        "221.229.1": 2,
        "235.1": 2,
        "241": 1,
        "272": 3,
        "281.1": 3,
    }


def test_cross_ref_targets_sum_to_22_and_fit_in_the_cells():
    q, c = plan.final_quotas(), plan.cross_ref_targets()
    assert sum(c.values()) == 22
    assert all(0 <= c[cell] <= q[cell] for cell in q)
    by_act = Counter()
    for (act, _), n in c.items():
        by_act[act] += n
    assert dict(by_act) == {
        "210": 2,
        "220": 14,
        "221.214.1": 1,
        "221.229.1": 1,
        "235.1": 1,
        "241": 1,
        "272": 1,
        "281.1": 1,
    }


# ------------------------------------------------------------------ the plan


def test_slot_counts_per_cell_are_one_and_a_half_times_the_quota(slots):
    counts = Counter((s["act"], s["split"]) for s in answerable(slots))
    for cell, q in plan.final_quotas().items():
        assert counts[cell] == draft_count(q), cell
    assert len(answerable(slots)) == 228
    assert len(traps(slots)) == 30
    assert len(slots) == 258


def test_slot_ids_are_stable_sequential_and_in_file_order(slots):
    assert [s["slot_id"] for s in slots] == [f"s-{i:04d}" for i in range(1, 259)]
    # answerable slots come first, grouped by cell in natural act order, dev before test
    cells = [(s["act"], s["split"]) for s in answerable(slots)]
    assert cells == sorted(cells, key=lambda c: (ACTS.index(c[0]), c[1]))
    assert all(s["qtype"] == "trap" for s in slots[228:])


def test_slot_fields(slots):
    keys = {
        "act",
        "gold_article_ids",
        "pilot",
        "qtype",
        "query_lang",
        "slot_id",
        "source_lang",
        "split",
        "trap_type",
    }
    for s in slots:
        assert set(s) == keys
        assert s["qtype"] in {"single", "cross_ref", "trap"}
        assert s["query_lang"] in {"en", "de", "fr", "it"}
        assert s["split"] in {"dev", "test"}
        assert isinstance(s["pilot"], bool)


def test_gold_ids_are_eligible_use_the_split_rule_and_are_used_once(slots, corpus):
    pool = set(benchmark_eligible(corpus)["article_id"])
    seen: Counter = Counter()
    for s in answerable(slots):
        expected = 2 if s["qtype"] == "cross_ref" else 1
        assert len(s["gold_article_ids"]) == expected
        for gold in s["gold_article_ids"]:
            seen[gold] += 1
            assert gold in pool
            assert split_of(gold) == s["split"]
            assert gold.partition(":")[0] == s["act"]
        assert s["trap_type"] is None
    assert max(seen.values()) == 1, [a for a, n in seen.items() if n > 1]
    for never in ("r1", "s1", "u1/1"):
        assert not any(a.endswith(f":{never}") for a in seen)


def test_the_plan_is_deterministic_and_independent_of_row_order(corpus, slots):
    shuffled = corpus.sample(frac=1.0, random_state=3).reset_index(drop=True)
    assert plan.build_plan(shuffled).slots == slots
    assert plan.build_plan(corpus).slots == slots


def test_a_different_seed_changes_the_sample(corpus, slots):
    other = plan.build_plan(corpus, seed=1).slots
    assert [s["gold_article_ids"] for s in other] != [s["gold_article_ids"] for s in slots]
    assert [s["split"] for s in other] == [s["split"] for s in slots]  # the split is not random


# ------------------------------------------------------------------ languages


def test_final_prefix_has_the_spec_language_balance(slots):
    langs = Counter(s["query_lang"] for s in final_prefix(slots))
    assert dict(langs) == LANG_FINAL == {"en": 38, "de": 37, "fr": 38, "it": 37}
    assert len(final_prefix(slots)) == 150


def test_languages_are_balanced_within_the_big_acts(slots):
    per_act = {}
    for s in final_prefix(slots):
        per_act.setdefault(s["act"], Counter())[s["query_lang"]] += 1
    for act in ("220", "210"):
        counts = [per_act[act][lang] for lang in ("en", "de", "fr", "it")]
        assert max(counts) - min(counts) <= 2, (act, counts)
    for act, counts in per_act.items():
        assert sum(counts.values()) == ACT_QUOTA[act]


def test_draft_surplus_is_balanced_too(slots):
    langs = Counter(s["query_lang"] for s in answerable(slots))
    assert max(langs.values()) - min(langs.values()) <= 3
    assert sum(langs.values()) == 228


def test_source_language_rules(slots):
    fin = final_prefix(slots)
    for lang in ("en", "de", "fr", "it"):
        mine = [s for s in fin if s["query_lang"] == lang]
        if lang == "en":
            c = Counter(s["source_lang"] for s in mine)
            assert set(c) == {"de", "fr", "it"} and max(c.values()) - min(c.values()) <= 1
        else:
            same = [s for s in mine if s["source_lang"] == lang]
            assert 8 <= len(same) <= 10, (lang, len(same))  # about 25%: the control group
            others = Counter(s["source_lang"] for s in mine if s["source_lang"] != lang)
            assert set(others) == {x for x in ("de", "fr", "it") if x != lang}
            assert abs(others.most_common()[0][1] - others.most_common()[1][1]) <= 1
    cross = sum(1 for s in fin if s["source_lang"] != s["query_lang"])
    assert 0.78 <= cross / len(fin) <= 0.84  # about 81% cross-lingual by construction
    assert all(s["source_lang"] in {"de", "fr", "it"} for s in answerable(slots))
    assert all(s["source_lang"] is None for s in traps(slots))


# ------------------------------------------------------------------ cross_ref


def test_cross_ref_slots_pair_a_referencing_article_with_the_one_it_cites(slots, corpus):
    text = {(r.article_id, r.lang): r.text for r in corpus.itertuples()}
    cross = [s for s in answerable(slots) if s["qtype"] == "cross_ref"]
    assert cross
    for s in cross:
        a, b = s["gold_article_ids"]
        number_b = b.partition(":")[2]
        for lang, marker in (("de", "Artikel"), ("fr", "l'article"), ("it", "l'articolo")):
            assert f"{marker} {number_b}." in text[(a, lang)]
        assert split_of(a) == split_of(b) == s["split"]


def test_cross_ref_final_prefix_matches_the_targets_when_pairs_exist(result, slots):
    targets = plan.cross_ref_targets()
    got = Counter((s["act"], s["split"]) for s in final_prefix(slots) if s["qtype"] == "cross_ref")
    for cell, n in targets.items():
        assert got[cell] == n, (cell, n, got[cell])
    assert sum(got.values()) == 22
    assert all(row["shortfall"] == 0 for row in result.cross_report)


def test_cross_ref_slots_are_spread_over_the_query_languages(slots):
    langs = Counter(s["query_lang"] for s in final_prefix(slots) if s["qtype"] == "cross_ref")
    assert sum(langs.values()) == 22
    assert set(langs) == {"en", "de", "fr", "it"}
    assert max(langs.values()) - min(langs.values()) <= 2, langs


def test_cross_ref_draft_count_is_one_and_a_half_times_the_target(result):
    for row in result.cross_report:
        assert row["draft_slots"] == min(
            draft_count(row["final_target"]), draft_count(plan.final_quotas()[row["cell"]])
        )


def test_without_references_everything_is_single_and_the_shortfall_is_reported():
    res = plan.build_plan(make_corpus(references=False))
    assert {s["qtype"] for s in answerable(res.slots)} == {"single"}
    assert len(answerable(res.slots)) == 228
    assert sum(row["shortfall"] for row in res.cross_report) == 22
    assert "shortfall" in plan.summary_text(res).lower()


# ------------------------------------------------------------------ traps


def test_trap_slots(slots):
    t = traps(slots)
    assert len(t) == 30
    for s in t:
        assert s["gold_article_ids"] == [] and s["act"] is None and s["source_lang"] is None
        assert s["trap_type"] in TRAP_TYPES and s["pilot"] in (True, False)


def test_trap_split_is_the_hash_rank_of_the_slot_id(slots):
    t = traps(slots)
    ranked = sorted(
        t,
        key=lambda s: hashlib.sha256(("fedlex-bench-v1-trap:" + s["slot_id"]).encode()).hexdigest(),
    )
    dev_ids = {s["slot_id"] for s in ranked[:8]}
    assert {s["slot_id"] for s in t if s["split"] == "dev"} == dev_ids
    assert Counter(s["split"] for s in t) == {"dev": 8, "test": 22}


def test_trap_final_prefix_has_five_per_language_and_5_dev_15_test(slots):
    quotas = {"dev": 5, "test": 15}
    seen: Counter = Counter()
    fin = []
    for s in traps(slots):
        if seen[s["split"]] < quotas[s["split"]]:
            fin.append(s)
        seen[s["split"]] += 1
    assert Counter(s["split"] for s in fin) == quotas
    assert Counter(s["query_lang"] for s in fin) == {"en": 5, "de": 5, "fr": 5, "it": 5}
    all_langs = Counter(s["query_lang"] for s in traps(slots))
    assert all(7 <= n <= 8 for n in all_langs.values()) and sum(all_langs.values()) == 30


def test_trap_types_rotate_and_are_not_tied_to_the_language(slots):
    t = traps(slots)
    types = Counter(s["trap_type"] for s in t)
    assert max(types.values()) - min(types.values()) <= 2
    pairs = Counter((s["query_lang"], s["trap_type"]) for s in t)
    assert len(pairs) > 8  # each language sees several types


# ------------------------------------------------------------------ pilot


def test_exactly_ten_pilot_slots_with_the_required_mix(slots):
    pilot = [s for s in slots if s["pilot"]]
    assert len(pilot) == 10
    ans, tr = answerable(pilot), traps(pilot)
    assert len(ans) == 8 and len(tr) == 2
    assert sum(1 for s in ans if s["qtype"] == "cross_ref") == 1
    assert sum(1 for s in ans if s["act"] == "220") >= 4
    assert len({s["act"] for s in ans}) >= 3
    assert len({s["query_lang"] for s in ans}) >= 3
    assert any(s["source_lang"] == s["query_lang"] for s in ans)
    assert any(s["source_lang"] != s["query_lang"] for s in ans)
    assert {s["split"] for s in pilot} == {"dev"}  # nothing from the test split is seen early
    assert len({s["trap_type"] for s in tr}) == 2 and len({s["query_lang"] for s in tr}) == 2


def test_pilot_choice_is_deterministic(corpus, slots):
    again = plan.build_plan(corpus).slots
    assert [s["slot_id"] for s in again if s["pilot"]] == [
        s["slot_id"] for s in slots if s["pilot"]
    ]


# ------------------------------------------------------------------ pool checks and output


def test_the_plan_stops_when_an_act_pool_is_below_twice_its_draft_need():
    small = make_corpus(sizes={**SIZES, "241": 12})
    with pytest.raises(plan.PlanError, match="241"):
        plan.build_plan(small)


def test_the_plan_stops_when_a_cell_has_too_few_articles():
    tiny = make_corpus(sizes={**SIZES, "221.214.1": 4})
    with pytest.raises(plan.PlanError):
        plan.build_plan(tiny)


def test_write_slots_is_idempotent_and_canonical(tmp_path, slots):
    p = tmp_path / "plan" / "slots.jsonl"
    h1 = plan.write_slots(slots, p)
    h2 = plan.write_slots(slots, p)
    assert h1 == h2 == hashlib.sha256(p.read_bytes()).hexdigest()
    raw = p.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    lines = raw.decode("utf-8").splitlines()
    assert len(lines) == 258
    assert lines[0].startswith('{"act": "210"')  # keys sorted
    assert plan.read_slots(p) == slots


def test_summary_text_reports_counts_by_cell_and_language(result):
    text = plan.summary_text(result)
    assert "act" in text and "query_lang" in text and "source_lang" in text
    assert "258" in text and "228" in text
    assert "220" in text and "trap" in text
    for lang in ("en", "de", "fr", "it"):
        assert lang in text
