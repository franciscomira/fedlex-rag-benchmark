"""``benchmark brief``: what the writers read. They must never see an article id or number."""

from __future__ import annotations

import hashlib
import json
import re

import pytest

from benchmark_fixtures import make_corpus
from fedlex_bench.benchmark import brief, plan
from fedlex_bench.benchmark.common import TRAP_TYPES, slot_hash, slots_sha256
from fedlex_bench.benchmark.refs import mentions_article_number, mentions_sr


class FakeStructure:
    """Structure provider: a fixed act name and a path derived from the article number."""

    def act_name(self, sr, lang):
        return f"Testgesetz {sr} ({lang})"

    def path(self, article_id, lang):
        number = article_id.partition(":")[2]
        if number.endswith("7"):
            return None  # an article with no division above it
        return [f"1. Teil: Allgemeines ({lang})", f"2. Titel: Kauf ab {number[:1]}"]


@pytest.fixture(scope="module")
def corpus():
    return make_corpus()


@pytest.fixture(scope="module")
def slots(corpus):
    return plan.build_plan(corpus).slots


@pytest.fixture(scope="module")
def structure():
    return FakeStructure()


def first(slots, **match):
    return next(s for s in slots if all(s[k] == v for k, v in match.items()))


def not_pilot(slots, **match):
    return next(s for s in slots if not s["pilot"] and all(s[k] == v for k, v in match.items()))


# ------------------------------------------------------------------ grouping


def test_groups_split_220_into_files_of_about_25_and_keep_other_acts_together(slots):
    groups = brief.group_slots(slots)
    names = list(groups)
    normal = [s for s in slots if not s["pilot"]]
    n220 = sum(len(v) for k, v in groups.items() if k.startswith("220-"))
    assert n220 == sum(1 for s in normal if s["act"] == "220")
    assert [k for k in names if k.startswith("220-")] == [f"220-{i:02d}" for i in range(1, 6)]
    others = [k for k in names if k.startswith("other-")]
    assert sum(len(groups[k]) for k in others) == sum(
        1 for s in normal if s["act"] not in (None, "220")
    )
    assert sum(len(groups[k]) for k in names if k.startswith("traps")) == sum(
        1 for s in normal if s["act"] is None
    )
    for k, members in groups.items():
        assert 15 <= len(members) <= 35, (k, len(members))
    for members in groups.values():
        ids = [s["slot_id"] for s in members]
        assert ids == sorted(ids)


def test_pilot_slots_are_left_out_of_the_full_run_groups(slots):
    groups = brief.group_slots(slots)
    flat = [s["slot_id"] for members in groups.values() for s in members]
    assert sorted(flat) == [s["slot_id"] for s in slots if not s["pilot"]]
    assert len(flat) == len(slots) - 10
    assert not {s["slot_id"] for s in slots if s["pilot"]} & set(flat)


def test_220_group_is_a_pure_220_group_and_other_groups_have_no_220(slots):
    groups = brief.group_slots(slots)
    for k, members in groups.items():
        acts = {s["act"] for s in members}
        if k.startswith("220-"):
            assert acts == {"220"}
        elif k.startswith("other-"):
            assert "220" not in acts and None not in acts
        else:
            assert acts == {None}


def test_pilot_is_a_single_group_with_the_ten_pilot_slots(slots):
    groups = brief.group_slots(slots, pilot=True)
    assert list(groups) == ["pilot"]
    assert [s["slot_id"] for s in groups["pilot"]] == [s["slot_id"] for s in slots if s["pilot"]]
    assert len(groups["pilot"]) == 10


def test_chunking_is_even():
    assert [len(c) for c in brief.chunk_evenly(list(range(135)))] == [27] * 5
    assert [len(c) for c in brief.chunk_evenly(list(range(93)))] == [24, 23, 23, 23]
    assert [len(c) for c in brief.chunk_evenly(list(range(10)))] == [10]
    assert brief.chunk_evenly([]) == []


# ------------------------------------------------------------------ content


def render(slots, corpus, structure=None, **match):
    slot = first(slots, **match)
    return slot, brief.render_group("t", [slot], corpus, structure)


def test_single_slot_shows_labels_languages_title_and_the_source_language_text(
    slots, corpus, structure
):
    slot, text = render(slots, corpus, structure, qtype="single", query_lang="de", source_lang="fr")
    assert f"## Slot {slot['slot_id']}" in text
    assert "qtype: single" in text
    assert "query_lang: de" in text and "German" in text
    assert "source_lang: fr" in text and "French" in text
    shown = text.split("Slots in this brief")[1]
    assert "### Article A" in shown and "### Article B" not in shown
    number = slot["gold_article_ids"][0].partition(":")[2]
    assert f"fr wortlaut {slot['act']} nummer {number} " in text  # the French text
    assert "de wortlaut" not in text and "it wortlaut" not in text
    assert f"Titel {number}" in text
    assert "Block heading:" in text


def test_act_name_and_structure_are_shown_in_the_source_language(slots, corpus, structure):
    slot, text = render(slots, corpus, structure, qtype="single", source_lang="it")
    number = slot["gold_article_ids"][0].partition(":")[2]
    assert f"Act: Testgesetz {slot['act']} (it)" in text
    if number.endswith("7"):
        assert "Structure: not available" in text
    else:
        assert f"Structure: 1. Teil: Allgemeines (it) > 2. Titel: Kauf ab {number[:1]}" in text


def test_missing_structure_is_stated_never_invented(slots, corpus):
    slot, text = render(slots, corpus, None, qtype="single")  # no structure provider
    assert "Act: not available" in text and "Structure: not available" in text
    assert "Teil" not in text and "Titel:" not in text.replace("Title:", "")


def test_structure_is_per_article_for_cross_ref(slots, corpus, structure):
    slot = first(slots, qtype="cross_ref")
    text = brief.render_group("t", [slot], corpus, structure)
    assert text.count("Act: Testgesetz") == 2 and text.count("Structure:") == 2


def test_the_brief_contains_no_article_id(slots, corpus, structure):
    groups = brief.group_slots(slots)
    groups.update(brief.group_slots(slots, pilot=True))
    for name, members in groups.items():
        text = brief.render_group(name, members, corpus, structure)
        for s in members:
            for gold in s["gold_article_ids"]:
                assert gold not in text, (name, gold)
        assert "articles.parquet" not in text and "art_" not in text
        assert not re.search(r"\b\d{1,3}(?:\.\d+){0,2}:\d+[a-z]*\b", text), name


def test_the_brief_shows_no_article_number_for_any_slot(slots, corpus, structure):
    """No "Art. 5", "Artikel 5", "article 5", "articolo 5" or "SR 220" in what a writer reads,
    for every slot of the plan (the corpus quotes such references in its texts)."""
    body = re.compile(r"(?m)^### Article [AB]$")
    for name, members in {**brief.group_slots(slots), **brief.group_slots(slots, True)}.items():
        text = brief.render_group(name, members, corpus, structure)
        rules_end = text.index("Slots in this brief:")
        shown = text[rules_end:]
        assert body.search(shown) or name.startswith("traps"), name
        assert not mentions_article_number(shown), (name, mentions_article_number(shown))
        assert not mentions_sr(shown), name


def test_cross_ref_slot_shows_both_articles_and_names_the_other_one_by_its_label(
    slots, corpus, structure
):
    slot = first(slots, qtype="cross_ref", source_lang="it")
    text = brief.render_group("t", [slot], corpus, structure).split("Slots in this brief")[1]
    a, b = slot["gold_article_ids"]
    assert "### Article A" in text and "### Article B" in text
    for aid in (a, b):
        number = aid.partition(":")[2]
        assert f"it wortlaut {slot['act']} nummer {number} " in text
    assert text.index("### Article A") < text.index("### Article B")
    assert "needs both" in text
    # A's text cites B: the number is replaced by the label (the fixture cites article i-1)
    article_a = text[text.index("### Article A") : text.index("### Article B")]
    assert "Vedi" in article_a
    assert "[Article B]" in article_a
    assert re.search(r"\d", article_a.split("Vedi")[-1]) is None


def test_references_to_other_articles_are_replaced_by_a_placeholder(slots, corpus, structure):
    slot = not_pilot(slots, qtype="single", source_lang="de")
    number = int(slot["gold_article_ids"][0].partition(":")[2])
    text = brief.render_group("t", [slot], corpus, structure)
    if number % 2 == 0 and number > 2:  # the fixture's even articles cite article number-1
        assert "Siehe [another article]." in text
        assert f"Artikel {number - 1}" not in text
    assert "Artikel" not in text.split("Slots in this brief")[1].replace("[another article]", "")


def test_trap_slot_shows_type_instruction_and_no_article(slots, corpus, structure):
    for trap_type in TRAP_TYPES:
        slot, text = render(slots, corpus, structure, qtype="trap", trap_type=trap_type)
        assert f"trap_type: {trap_type}" in text
        assert "Article A" not in text and "Act:" not in text.split("Slots in this brief")[1]
        assert "trap_reason" in text
        assert "source_lang" not in text or "source_lang: -" in text
        assert slot["slot_id"] in text


def test_trap_instructions_differ_by_type(slots, corpus, structure):
    texts = {t: render(slots, corpus, structure, qtype="trap", trap_type=t)[1] for t in TRAP_TYPES}
    assert len(set(texts.values())) == 4
    assert "repealed" in texts["repealed"].lower()
    assert "case law" in texts["non_statute"].lower()


def test_writer_rules_are_stated_once_at_the_top(slots, corpus, structure):
    text = brief.render_group("220-01", brief.group_slots(slots)["220-01"][:2], corpus, structure)
    assert text.startswith("# Brief 220-01")
    assert text.count("slot_id") >= 2
    assert '"skip"' in text and "gold_facts" in text and "difficulty" in text
    assert "[Article B]" in text.split("Slots in this brief")[0]  # the rules explain placeholders


def test_missing_article_text_is_an_error(slots, corpus):
    slot = first(slots, qtype="single")
    gone = corpus[
        ~(
            (corpus["article_id"] == slot["gold_article_ids"][0])
            & (corpus["lang"] == slot["source_lang"])
        )
    ]
    with pytest.raises(KeyError, match=slot["gold_article_ids"][0]):
        brief.render_group("t", [slot], gone)


# ------------------------------------------------------------------ files and index


def test_write_briefs_creates_one_file_per_group_with_stable_hashes(
    tmp_path, slots, corpus, structure
):
    sums = brief.write_briefs(slots, corpus, tmp_path, structure=structure)
    files = sorted(p.name for p in tmp_path.glob("*.md"))
    assert files == sorted(sums)
    assert "traps-01.md" in files and "220-01.md" in files and "other-01.md" in files
    for name, sha in sums.items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == sha
    again = brief.write_briefs(slots, corpus, tmp_path, structure=structure)
    assert again == sums  # idempotent


def test_pilot_brief_is_one_file_named_pilot(tmp_path, slots, corpus, structure):
    sums = brief.write_briefs(slots, corpus, tmp_path, pilot=True, structure=structure)
    assert list(sums) == ["pilot.md"]
    text = (tmp_path / "pilot.md").read_text(encoding="utf-8")
    assert text.count("## Slot ") == 10


def test_index_records_the_plan_hash_and_a_hash_of_every_slot(tmp_path, slots, corpus, structure):
    index_dir = tmp_path / "plan"
    sums = brief.write_briefs(
        slots, corpus, tmp_path / "briefs", structure=structure, index_dir=index_dir
    )
    assert not list((tmp_path / "briefs").glob("*.json"))  # the index is not in the ignored folder
    index = json.loads((index_dir / "index.json").read_text(encoding="utf-8"))
    assert index["plan_sha256"] == slots_sha256(slots)
    normal = [s for s in slots if not s["pilot"]]
    assert set(index["slots"]) == {s["slot_id"] for s in normal}
    by_id = {s["slot_id"]: s for s in slots}
    for slot_id, entry in index["slots"].items():
        assert sums[entry["file"]] == entry["sha256"]
        assert entry["slot_sha256"] == slot_hash(by_id[slot_id])  # gold ids are part of it


def test_the_slot_hash_changes_with_the_gold_ids(tmp_path, slots, corpus, structure):
    brief.write_briefs(slots, corpus, tmp_path, structure=structure)
    index = brief.read_index(tmp_path / "index.json")
    s = not_pilot(slots, qtype="single")
    changed = {**s, "gold_article_ids": ["220:999"]}
    assert index.slots[s["slot_id"]]["slot_sha256"] == slot_hash(s) != slot_hash(changed)


def test_pilot_index_has_only_the_pilot_slots_and_the_merge_covers_everything(
    tmp_path, slots, corpus, structure
):
    brief.write_briefs(slots, corpus, tmp_path, structure=structure)
    pilot_sums = brief.write_briefs(slots, corpus, tmp_path, pilot=True, structure=structure)
    full = brief.read_index(tmp_path / "index.json")
    pilot = brief.read_index(tmp_path / "index-pilot.json")
    assert len(pilot.slots) == 10
    assert all(
        v["file"] == "pilot.md" and v["sha256"] == pilot_sums["pilot.md"]
        for v in pilot.slots.values()
    )
    merged, problems = brief.merge_indexes([full, pilot])
    assert problems == [] and set(merged.slots) == {s["slot_id"] for s in slots}
    assert merged.plan_sha256 == full.plan_sha256 == pilot.plan_sha256


def test_merge_reports_indexes_from_different_plans_and_overlaps():
    a = brief.BriefIndex("1" * 64, {"s-0001": {}})
    b = brief.BriefIndex("2" * 64, {"s-0001": {}})
    merged, problems = brief.merge_indexes([a, b])
    assert any("different plans" in p for p in problems)
    assert any("s-0001" in p and "two" in p for p in problems)


def test_briefs_use_lf_and_utf8(tmp_path, slots, corpus, structure):
    brief.write_briefs(slots, corpus, tmp_path, pilot=True, structure=structure)
    raw = (tmp_path / "pilot.md").read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    raw.decode("utf-8")
