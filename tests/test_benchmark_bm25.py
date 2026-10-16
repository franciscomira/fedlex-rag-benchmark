"""BM25 (own implementation, spec section 6) on toy corpora. No corpus needed."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from fedlex_bench.benchmark.bm25 import (
    Bm25Index,
    build_indexes,
    compute_ranks,
    lexical_overlap,
    tokenize,
)

# ------------------------------------------------------------------ tokenizer


def test_tokenize_lowercases_keeps_unicode_and_drops_single_characters():
    assert tokenize("Art. 257d: Die Verjährung, 5 Jahre; a b") == [
        "art",
        "257d",
        "die",
        "verjährung",
        "jahre",
    ]
    assert tokenize("L'article 12 du Code") == ["article", "12", "du", "code"]
    assert tokenize("") == []


def test_tokenize_does_no_stemming_and_no_stop_word_removal():
    assert tokenize("der Kaufvertrag und die Kaufverträge") == [
        "der",
        "kaufvertrag",
        "und",
        "die",
        "kaufverträge",
    ]


# ------------------------------------------------------------------ scoring formula


def test_score_matches_a_hand_computed_value():
    index = Bm25Index({"d1": "alpha beta", "d2": "alpha gamma gamma"})
    # N=2, avgdl=(2+3)/2=2.5; "beta" occurs in 1 doc: idf=ln(1+(2-1+0.5)/(1+0.5))=ln 2
    k1, b = 1.5, 0.75
    norm = 1 + k1 * (1 - b + b * 2 / 2.5)  # tf=1, dl=2
    expected = math.log(2) * (1 * (k1 + 1)) / norm
    scores = index.scores("beta")
    assert scores == {"d1": pytest.approx(expected)}
    assert round(scores["d1"], 4) == 0.7617


def test_idf_is_never_negative_even_for_a_term_in_every_document():
    index = Bm25Index({"d1": "alpha beta", "d2": "alpha gamma"})
    assert index.scores("alpha")["d1"] > 0


def test_repeated_query_terms_count_each_time():
    index = Bm25Index({"d1": "alpha beta", "d2": "gamma delta"})
    once, twice = index.scores("alpha")["d1"], index.scores("alpha alpha")["d1"]
    assert twice == pytest.approx(2 * once)


def test_parameters_are_k1_1_5_and_b_0_75():
    index = Bm25Index({"d1": "alpha"})
    assert (index.k1, index.b) == (1.5, 0.75)


# ------------------------------------------------------------------ ranks


@pytest.fixture
def toy() -> Bm25Index:
    return Bm25Index(
        {
            "a": "Der Kaufvertrag verpflichtet den Verkäufer",
            "b": "Der Mietvertrag verpflichtet den Vermieter",
            "c": "Die Verjährung beginnt mit der Fälligkeit",
        }
    )


def test_known_rank_case(toy):
    # "b" has both query terms, "a" only "verpflichtet", "c" none (score 0).
    assert toy.rank("verpflichtet Vermieter", ["b"]) == 1
    assert toy.rank("verpflichtet Vermieter", ["a"]) == 2
    assert toy.rank("verpflichtet Vermieter", ["c"]) == 3
    assert toy.top("verpflichtet Vermieter", 2) == ["b", "a"]


def test_rare_term_beats_common_term(toy):
    assert toy.rank("Verjährung Fälligkeit", ["c"]) == 1
    # no stemming: "vertrag" is not "kaufvertrag"
    assert toy.scores("vertrag") == {}
    assert toy.rank("vertrag", ["c"]) == 3  # all scores 0: article order


def test_best_of_several_gold_articles_is_used(toy):
    assert toy.rank("verpflichtet Vermieter", ["a", "b"]) == 1
    assert toy.rank("verpflichtet Vermieter", ["c", "a"]) == 2


def test_ties_are_broken_by_article_order_not_by_input_order():
    docs = {"220:10": "alpha beta", "220:2": "alpha beta", "220:2a": "alpha beta"}
    index = Bm25Index(docs)  # natural article order: 2 < 2a < 10
    assert [index.rank("alpha", [d]) for d in ("220:2", "220:2a", "220:10")] == [1, 2, 3]
    reordered = Bm25Index(dict(reversed(list(docs.items()))))
    assert [reordered.rank("alpha", [d]) for d in ("220:2", "220:2a", "220:10")] == [1, 2, 3]


def test_zero_score_documents_rank_after_matches_in_article_order():
    docs = {"d1": "alpha", "d2": "beta", "d3": "gamma", "d4": "alpha"}
    index = Bm25Index(docs)
    assert index.rank("alpha", ["d1"]) == 1
    assert index.rank("alpha", ["d4"]) == 2
    # no document matches: everything has score 0, order is the article order
    assert [index.rank("zzz", [d]) for d in ("d1", "d2", "d3", "d4")] == [1, 2, 3, 4]
    # d2 and d3 have score 0; d4 is a match, so only d2 sorts before d3 among the zeros
    assert index.rank("alpha", ["d2"]) == 3
    assert index.rank("alpha", ["d3"]) == 4


def test_rank_of_unknown_or_empty_gold_is_an_error(toy):
    with pytest.raises(KeyError):
        toy.rank("x", ["nope"])
    with pytest.raises(ValueError):
        toy.rank("x", [])


def test_query_without_tokens_gives_the_article_order(toy):
    assert toy.rank("", ["a"]) == 1
    assert toy.rank("?", ["c"]) == 3


def test_ranks_are_stable_across_runs():
    docs = {f"220:{i}": f"alpha beta{i % 3} gamma" for i in range(1, 40)}
    first = [Bm25Index(docs).rank("alpha beta1", [f"220:{i}"]) for i in range(1, 40)]
    second = [Bm25Index(docs).rank("alpha beta1", [f"220:{i}"]) for i in range(1, 40)]
    assert first == second
    assert sorted(first) == list(range(1, 40))  # a permutation: no two articles share a rank


# ------------------------------------------------------------------ indexes from a corpus frame


def _frame() -> pd.DataFrame:
    rows = []
    for lang, titles in {
        "de": ("Kauf", "Miete", "Verjährung"),
        "fr": ("Vente", "Bail", "Prescription"),
        "it": ("Vendita", "Locazione", "Prescrizione"),
    }.items():
        for i, title in enumerate(titles, 1):
            rows.append(
                {
                    "sr": "220",
                    "article_id": f"220:{i}",
                    "lang": lang,
                    "title": title,
                    "text": f"Testo {lang} numero {i} " + "x" * 10,
                    "kind": "article",
                    "is_repealed": False,
                    "is_historical": False,
                }
            )
    # a repealed article and a transitional one are not part of any index
    rows.append({**rows[0], "article_id": "220:9", "text": "", "is_repealed": True})
    rows.append({**rows[0], "article_id": "220:u1/1", "kind": "transitional"})
    return pd.DataFrame(rows)


def test_build_indexes_covers_retrieval_eligible_articles_only():
    indexes = build_indexes(_frame())
    assert set(indexes) == {"de", "fr", "it"}
    for index in indexes.values():
        assert sorted(index.ids) == ["220:1", "220:2", "220:3"]


def test_titles_are_part_of_the_indexed_text():
    indexes = build_indexes(_frame())
    assert indexes["de"].rank("Verjährung", ["220:3"]) == 1
    assert indexes["fr"].rank("Prescription", ["220:3"]) == 1
    assert indexes["it"].rank("Locazione", ["220:2"]) == 1
    assert indexes["it"].rank("Locazione", ["220:1"]) > 1


# ------------------------------------------------------------------ item ranks and lexical overlap


def test_compute_ranks_returns_one_rank_per_language_and_the_overlap():
    indexes = build_indexes(_frame())
    ranks, overlap = compute_ranks("Verjährung", ["220:3"], "de", indexes)
    assert set(ranks) == {"de", "fr", "it"} and ranks["de"] == 1
    # cross-lingual: the German word is unknown in the French and Italian indexes, the gold
    # article scores 0 there, and a zero score is "no rank" (null), not a rank at the end
    assert ranks["fr"] is None and ranks["it"] is None
    assert overlap == 1


def test_lexical_overlap_is_the_own_language_rank_and_null_for_english():
    ranks = {"de": 7, "fr": 2, "it": 30}
    assert lexical_overlap(ranks, "de") == 7
    assert lexical_overlap(ranks, "fr") == 2
    assert lexical_overlap(ranks, "it") == 30
    assert lexical_overlap(ranks, "en") is None  # n/a: an English query has no own index
    assert lexical_overlap({"de": None, "fr": 2, "it": 3}, "de") is None


def test_english_question_keeps_all_three_ranks_but_no_lexical_overlap():
    indexes = build_indexes(_frame())
    ranks, overlap = compute_ranks("Prescrizione", ["220:3"], "en", indexes)
    assert ranks == {"de": None, "fr": None, "it": 1}
    assert overlap is None


def test_a_gold_article_with_score_zero_has_no_rank_even_when_it_would_sort_first():
    index = Bm25Index({"a": "alpha", "b": "beta"})
    assert index.rank("zzz", ["a"]) == 1  # article order among the zero scores
    assert index.rank_scored("zzz", ["a"]) is None
    assert index.rank_scored("beta", ["b"]) == 1
    assert index.rank_scored("beta", ["a"]) is None


def test_rank_scored_uses_the_best_scoring_gold_article():
    index = Bm25Index({"a": "alpha", "b": "beta", "c": "beta gamma"})
    assert index.rank_scored("beta", ["a", "c"]) == index.rank("beta", ["c"])
    assert index.rank_scored("beta", ["a", "b"]) == 1
    assert index.rank_scored("zzz", ["a", "b"]) is None


def test_compute_ranks_cross_reference_uses_best_gold_article():
    indexes = build_indexes(_frame())
    ranks, _ = compute_ranks("Miete", ["220:1", "220:2"], "de", indexes)
    assert ranks["de"] == 1
