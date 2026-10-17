from __future__ import annotations

import pytest
from retrieval_fixtures import make_articles

from fedlex_bench.retrieval.chunking import (
    build_chunks,
    chunk_article,
    split_paragraphs,
    split_windows,
)


def words(n, start=0):
    return " ".join(f"w{i}" for i in range(start, start + n))


def test_article_chunk_keeps_id_and_prefixes_title():
    [c] = chunk_article("220:257d", "de", "Titel", "1 Text", "article")
    assert (c.article_id, c.chunk_id, c.lang) == ("220:257d", "220:257d", "de")
    assert c.text == "Titel\n1 Text"


def test_missing_title_gives_body_only():
    [c] = chunk_article("220:1", "de", "", "1 Text", "article")
    assert c.text == "1 Text"


def test_split_paragraphs_numbered_with_list_items():
    text = "1 Erster Absatz:\n1. a;\n2. b.\n2 Zweiter Absatz.\n2bis Dritter."
    assert split_paragraphs(text) == [
        "1 Erster Absatz:\n1. a;\n2. b.",
        "2 Zweiter Absatz.",
        "2bis Dritter.",
    ]


def test_split_paragraphs_unnumbered_is_one_chunk():
    assert split_paragraphs("Ein einziger Absatz\nmit Umbruch") == [
        "Ein einziger Absatz\nmit Umbruch"
    ]


def test_paragraph_chunks_ids_and_titles():
    chunks = chunk_article("220:1", "fr", "Loyer", "1 a\n2 b", "paragraph")
    assert [c.chunk_id for c in chunks] == ["220:1#p1", "220:1#p2"]
    assert {c.article_id for c in chunks} == {"220:1"}
    assert chunks[1].text == "Loyer\n2 b"


def test_split_windows_200_50():
    wins = split_windows(words(450), 200, 50)
    assert [w.split()[0] for w in wins] == ["w0", "w150", "w300"]
    assert [len(w.split()) for w in wins] == [200, 200, 150]
    assert wins[0].split()[-50:] == wins[1].split()[:50]  # overlap of 50 words


def test_split_windows_edges():
    assert len(split_windows(words(10), 200, 50)) == 1
    assert len(split_windows(words(200), 200, 50)) == 1
    two = split_windows(words(201), 200, 50)
    assert len(two) == 2 and two[1].split()[-1] == "w200"
    assert split_windows("", 200, 50) == []


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1), (10, 11)])
def test_split_windows_rejects_bad_params(size, overlap):
    with pytest.raises(ValueError):
        split_windows("a b c", size, overlap)


def test_window_chunk_ids():
    chunks = chunk_article("220:1", "it", "T", words(450), "window")
    assert [c.chunk_id for c in chunks] == ["220:1#w1", "220:1#w2", "220:1#w3"]
    assert all(c.text.startswith("T\n") for c in chunks)


def test_unknown_chunking_raises():
    with pytest.raises(ValueError, match="chunking"):
        chunk_article("220:1", "de", "T", "x", "sentence")


def test_build_chunks_only_eligible_sorted_and_deterministic():
    df = make_articles()
    a = build_chunks(df, "article")
    b = build_chunks(df.sample(frac=1.0, random_state=3), "article")
    assert a == b
    assert {c.article_id for c in a} == {"220:1", "220:2", "210:7"}
    assert len(a) == 9
    assert [c.article_id for c in a][:3] == ["210:7"] * 3  # natural act order
    assert [c.lang for c in a][:3] == ["de", "fr", "it"]


def test_build_chunks_language_filter_and_variants():
    df = make_articles()
    fr = build_chunks(df, "paragraph", langs=("fr",))
    assert {c.lang for c in fr} == {"fr"}
    assert [c.chunk_id for c in fr if c.article_id == "220:1"] == ["220:1#p1", "220:1#p2"]
