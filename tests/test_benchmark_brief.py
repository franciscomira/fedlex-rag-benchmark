"""``benchmark brief``: what the writers read. They must never see an article id."""

from __future__ import annotations

import hashlib
import json

import pytest

from benchmark_fixtures import make_corpus
from fedlex_bench.benchmark import brief, plan
from fedlex_bench.benchmark.common import TRAP_TYPES


@pytest.fixture(scope="module")
def corpus():
    return make_corpus()


@pytest.fixture(scope="module")
def slots(corpus):
    return plan.build_plan(corpus).slots


def first(slots, **match):
    return next(s for s in slots if all(s[k] == v for k, v in match.items()))


# ------------------------------------------------------------------ grouping


def test_groups_split_220_into_files_of_about_25_and_keep_other_acts_together(slots):
    groups = brief.group_slots(slots)
    names = list(groups)
    n220 = sum(len(v) for k, v in groups.items() if k.startswith("220-"))
    assert n220 == sum(1 for s in slots if s["act"] == "220") == 135
    assert [k for k in names if k.startswith("220-")] == [f"220-{i:02d}" for i in range(1, 6)]
    others = [k for k in names if k.startswith("other-")]
    assert sum(len(groups[k]) for k in others) == 228 - 135
    assert sum(len(groups[k]) for k in names if k.startswith("traps")) == 30
    for k, members in groups.items():
        assert 15 <= len(members) <= 35, (k, len(members))
    # every slot is in exactly one group, in slot order
    flat = [s["slot_id"] for members in groups.values() for s in members]
    assert sorted(flat) == [s["slot_id"] for s in slots]
    for members in groups.values():
        ids = [s["slot_id"] for s in members]
        assert ids == sorted(ids)


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


def render(slots, corpus, **match):
    slot = first(slots, **match)
    return slot, brief.render_group("t", [slot], corpus)


def test_single_slot_shows_labels_languages_title_and_the_source_language_text(slots, corpus):
    slot, text = render(slots, corpus, qtype="single", query_lang="de", source_lang="fr")
    assert f"## Slot {slot['slot_id']}" in text
    assert "qtype: single" in text
    assert "query_lang: de" in text and "German" in text
    assert "source_lang: fr" in text and "French" in text
    assert "Article A" in text and "Article B" not in text
    number = slot["gold_article_ids"][0].partition(":")[2]
    assert f"fr wortlaut {slot['act']} nummer {number} " in text  # the French text
    assert "de wortlaut" not in text and "it wortlaut" not in text
    assert f"Titel {number}" in text
    assert "Block heading:" in text


def test_the_brief_contains_no_article_id(slots, corpus):
    groups = brief.group_slots(slots)
    for name, members in groups.items():
        text = brief.render_group(name, members, corpus)
        for s in members:
            for gold in s["gold_article_ids"]:
                assert gold not in text, (name, gold)
        assert "articles.parquet" not in text and "art_" not in text


def test_cross_ref_slot_shows_both_articles_in_the_source_language(slots, corpus):
    slot, text = render(slots, corpus, qtype="cross_ref", source_lang="it")
    a, b = slot["gold_article_ids"]
    assert "Article A" in text and "Article B" in text
    for aid in (a, b):
        number = aid.partition(":")[2]
        assert f"it wortlaut {slot['act']} nummer {number} " in text
    assert text.index("Article A") < text.index("Article B")
    assert "needs both" in text


def test_trap_slot_shows_type_instruction_and_no_article(slots, corpus):
    for trap_type in TRAP_TYPES:
        slot, text = render(slots, corpus, qtype="trap", trap_type=trap_type)
        assert f"trap_type: {trap_type}" in text
        assert "Article A" not in text
        assert "trap_reason" in text  # the instruction tells the writer what to return
        assert "source_lang" not in text or "source_lang: -" in text
        assert slot["slot_id"] in text


def test_trap_instructions_differ_by_type(slots, corpus):
    texts = {t: render(slots, corpus, qtype="trap", trap_type=t)[1] for t in TRAP_TYPES}
    assert len(set(texts.values())) == 4
    assert "repealed" in texts["repealed"].lower()
    assert "case law" in texts["non_statute"].lower()


def test_writer_rules_are_stated_once_at_the_top(slots, corpus):
    text = brief.render_group("220-01", brief.group_slots(slots)["220-01"][:2], corpus)
    assert text.startswith("# Brief 220-01")
    assert text.count("slot_id") >= 2
    assert '"skip"' in text and "gold_facts" in text and "difficulty" in text


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


# ------------------------------------------------------------------ files


def test_write_briefs_creates_one_file_per_group_with_stable_hashes(tmp_path, slots, corpus):
    sums = brief.write_briefs(slots, corpus, tmp_path)
    files = sorted(p.name for p in tmp_path.glob("*.md"))
    assert files == sorted(sums)
    assert "traps-01.md" in files and "220-01.md" in files and "other-01.md" in files
    for name, sha in sums.items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == sha
    again = brief.write_briefs(slots, corpus, tmp_path)
    assert again == sums  # idempotent


def test_pilot_brief_is_one_file_named_pilot(tmp_path, slots, corpus):
    sums = brief.write_briefs(slots, corpus, tmp_path, pilot=True)
    assert list(sums) == ["pilot.md"]
    text = (tmp_path / "pilot.md").read_text(encoding="utf-8")
    assert text.count("## Slot ") == 10


def test_index_maps_every_slot_to_its_brief_file_and_hash(tmp_path, slots, corpus):
    sums = brief.write_briefs(slots, corpus, tmp_path)
    index = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert set(index["slots"]) == {s["slot_id"] for s in slots}
    for entry in index["slots"].values():
        assert sums[entry["file"]] == entry["sha256"]
    pilot_sums = brief.write_briefs(slots, corpus, tmp_path, pilot=True)
    pilot_index = json.loads((tmp_path / "index-pilot.json").read_text(encoding="utf-8"))
    assert len(pilot_index["slots"]) == 10
    assert all(
        v["file"] == "pilot.md" and v["sha256"] == pilot_sums["pilot.md"]
        for v in pilot_index["slots"].values()
    )


def test_briefs_use_lf_and_utf8(tmp_path, slots, corpus):
    brief.write_briefs(slots, corpus, tmp_path, pilot=True)
    raw = (tmp_path / "pilot.md").read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    raw.decode("utf-8")
