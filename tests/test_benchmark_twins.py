"""Near-duplicate articles ("twins"): excluded as gold candidates at plan time."""

from __future__ import annotations

import random
from fractions import Fraction

import pytest

from fedlex_bench.benchmark import twins


def test_normalise_text_lowercases_and_collapses_everything_but_letters_and_digits():
    assert twins.normalise_text("  Der  Käufer,\nder Verkäufer (Art. 5)!  ") == (
        "der käufer der verkäufer art 5"
    )


def test_char_ngrams_and_jaccard():
    assert twins.char_ngrams("abcdef") == {"abcde", "bcdef"}
    assert twins.char_ngrams("abc") == {"abc"}  # shorter than 5: the text itself
    assert twins.jaccard({"a", "b"}, {"b", "c"}) == Fraction(1, 3)
    assert twins.jaccard(set(), set()) == 1


def test_identical_and_near_identical_articles_are_twins():
    base = (
        "Der Verkäufer ist verpflichtet, die Sache zu übergeben und das Eigentum zu verschaffen. "
        * 3
    )
    texts = {
        "220:1": base,
        "220:2": base,
        "220:3": base + "Dazu kommt ein kurzer Zusatz.",
        "220:4": "Völlig anderer Text über die Miete von Wohnräumen und die Kündigung des Mieters.",
    }
    found = twins.find_twins(texts)
    assert set(found) == {"220:1", "220:2", "220:3"}
    assert found["220:1"][0] == "220:2" and found["220:1"][1] == 1
    assert found["220:2"][0] == "220:1"
    assert found["220:3"][0] in {"220:1", "220:2"} and 0.7 <= found["220:3"][1] < 1


def test_the_twin_is_the_partner_with_the_highest_jaccard_then_the_lowest_id():
    base = (
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda my ny xi omikron pi " * 2
    )
    texts = {
        "220:5": base,
        "220:6": base + "extra words here",  # closer
        "220:7": base + "extra words here and a lot more words that were added to this one",
        "220:8": base,  # identical to 5: ties with nobody closer
    }
    found = twins.find_twins(texts)
    assert found["220:5"][0] == "220:8" and found["220:8"][0] == "220:5"
    assert found["220:6"][0] in {"220:5", "220:8"}  # tie between two identical partners: lowest id
    assert found["220:6"][0] == "220:5"


def test_the_threshold_is_inclusive_and_exact():
    texts = {
        "x:1": "abcdefghij",
        "x:2": "abcdefghik",
    }  # grams: 6 each, 5 shared of 7 -> 5/7 = 0.714
    assert twins.jaccard(
        twins.char_ngrams("abcdefghij"), twins.char_ngrams("abcdefghik")
    ) == Fraction(5, 7)
    assert set(twins.find_twins(texts, threshold=0.7)) == {"x:1", "x:2"}
    assert twins.find_twins(texts, threshold=0.72) == {}


def test_matches_brute_force_on_random_texts():
    rng = random.Random(7)
    words = [f"w{i}x" for i in range(60)]
    texts = {}
    base_texts = [" ".join(rng.choice(words) for _ in range(rng.randint(8, 30))) for _ in range(25)]
    for i, t in enumerate(base_texts):
        texts[f"220:{i + 1}"] = t
    # derived near-copies with a few words replaced
    for i, t in enumerate(base_texts[:12]):
        parts = t.split()
        for _ in range(rng.randint(0, 3)):
            parts[rng.randrange(len(parts))] = rng.choice(words)
        texts[f"220:{i + 100}"] = " ".join(parts)
    norm = {k: twins.char_ngrams(twins.normalise_text(v)) for k, v in texts.items()}
    for threshold in (0.5, 0.7, 0.9):
        expected = {}
        for a in norm:
            best = max(
                ((twins.jaccard(norm[a], norm[b]), b) for b in norm if b != a),
                key=lambda x: (x[0], -ord_key(x[1])),
            )
            if best[0] >= Fraction(str(threshold)):
                expected[a] = (best[1], float(best[0]))
        got = twins.find_twins(texts, threshold=threshold)
        assert {k: v[0] for k, v in got.items()} == {k: v[0] for k, v in expected.items()}, (
            threshold
        )
        for k, (_, j) in got.items():
            assert j == pytest.approx(expected[k][1])


def ord_key(article_id: str) -> int:
    return int(article_id.partition(":")[2])


def test_records_are_sorted_and_rounded():
    found = {"220:10": ("220:2", 0.123456), "220:2": ("220:10", 0.123456)}
    recs = twins.twin_records(found)
    assert [r["article_id"] for r in recs] == ["220:2", "220:10"]
    assert recs[0] == {"article_id": "220:2", "jaccard": 0.1235, "twin_id": "220:10"}


def test_no_twins_in_an_empty_or_single_corpus():
    assert twins.find_twins({}) == {}
    assert twins.find_twins({"220:1": "text"}) == {}
