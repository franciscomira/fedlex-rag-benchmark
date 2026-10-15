"""Article-id parsing, number ordering and range-id resolution. No network access.

The corpus-level tests at the bottom run on ``data/processed/articles.parquet`` and are skipped
when it is missing (rebuild with ``uv run python -m fedlex_bench.chunk``).
"""

from __future__ import annotations

import pandas as pd
import pytest

from fedlex_bench.chunk import ARTICLES_PARQUET
from fedlex_bench.ids import (
    ArticleIdError,
    IdResolver,
    expand_range,
    is_range_id,
    normalize_article_id,
    number_key,
    parse_article_id,
    range_covers,
    resolve_id,
    split_number,
)

# ------------------------------------------------------------------ parsing


@pytest.mark.parametrize(
    ("article_id", "expected"),
    [
        ("220:257d", ("220", "257d")),
        ("220:1", ("220", "1")),
        ("221.229.1:95c", ("221.229.1", "95c")),
        ("235.1:6bis", ("235.1", "6bis")),
        ("220:6abis", ("220", "6abis")),
        ("210:135-149", ("210", "135-149")),
        ("220:697l-697m", ("220", "697l-697m")),
        ("220:u12/2-4", ("220", "u12/2-4")),
        ("220:u12/3", ("220", "u12/3")),
        ("220:u1", ("220", "u1")),
        ("220:12ater", ("220", "12ater")),  # letter + Latin ordinal
        ("220:5aquater", ("220", "5aquater")),
        ("220:5gbis", ("220", "5gbis")),
        ("220:5quinquies", ("220", "5quinquies")),
        ("220:5-5bis", ("220", "5-5bis")),
    ],
)
def test_parse_article_id(article_id, expected):
    assert parse_article_id(article_id) == expected


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "220",
        "220:",
        ":5",
        "abc:5",
        "220:5 d",  # not normalised: whitespace
        "220:257D",  # not normalised: upper case
        "220:Art. 5",
        "220:135-",
        "220:-149",
        "220:135-149-150",
        "220:149-135",  # descending range
        "220:5-5",  # empty range
        "220:u1/3-u2/4",  # range across blocks
        "220:u/3",
        "220:5/3",
        "220:5bis-",
        # suffix grammar: one letter, an optional Latin ordinal, or a letter plus an ordinal
        "220:135ff",
        "220:140xyz",
        "220:5aa",
        "220:5abc",
        "220:5bisbis",
        "220:5bisx",
        "220:5xbisx",
        "220:5biss",
        "220:5foo",
        "220:135ff-149",
        "220:135-149ff",
    ],
)
def test_parse_article_id_rejects_malformed(bad):
    with pytest.raises(ArticleIdError):
        parse_article_id(bad)


def test_article_id_error_is_a_value_error():
    assert issubclass(ArticleIdError, ValueError)


def test_split_number():
    s = split_number("135-149")
    assert (s.block, s.start, s.end) == ("", "135", "149")
    s = split_number("u12/2-4")
    assert (s.block, s.start, s.end) == ("u12", "2", "4")
    s = split_number("257d")
    assert (s.block, s.start, s.end) == ("", "257d", None)
    s = split_number("u3")
    assert (s.block, s.start, s.end) == ("u3", "", None)


def test_is_range_id():
    assert is_range_id("210:135-149")
    assert is_range_id("220:u12/2-4")
    assert not is_range_id("220:257d")
    assert not is_range_id("220:u12/3")
    assert not is_range_id("garbage")  # unparsable is not a range id


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("220:257 d", "220:257d"),
        ("220:Art. 257 d", "220:257d"),
        ("220:257D", "220:257d"),
        ("220:8 bis", "220:8bis"),
        ("220:6a bis", "220:6abis"),
        ("210:135–149", "210:135-149"),  # en dash
        ("210:Art. 135 à 149", "210:135-149"),
        ("210:135 a 149", "210:135-149"),
        ("220:697l-697m", "220:697l-697m"),  # already canonical
        ("220:u12/2-4", "220:u12/2-4"),
        (" 220 : 257d ", "220:257d"),
    ],
)
def test_normalize_article_id(raw, expected):
    assert normalize_article_id(raw) == expected
    assert normalize_article_id(expected) == expected  # idempotent


def test_a_conjunction_is_not_an_article_id():
    # "Art. 50 und 51" cites two articles: it is neither one id nor the range 50-51.
    for raw in ("210:Art. 50 und 51", "210:50 et 51", "210:50 e 51", "210:50 und 52"):
        with pytest.raises(ArticleIdError):
            normalize_article_id(raw)
        assert resolve_id(raw, {"210:50-51", "210:50", "210:51"}) is None
    # a real range with a connector still normalises
    assert normalize_article_id("210:Art. 50 bis 51") == "210:50-51"


def test_normalize_article_id_rejects_unparsable():
    for bad in ("", "220", "220:", "220:foo bar baz!"):
        with pytest.raises(ArticleIdError):
            normalize_article_id(bad)


# ------------------------------------------------------------------ ordering


def test_number_key_orders_suffixes():
    ordered = [
        "6",
        "6a",
        "6abis",
        "6b",
        "6bis",
        "6ter",
        "6quater",
        "6quinquies",
        "7",
        "10",
        "257",
        "257a",
        "257d",
        "257e",
        "257bis",
        "258",
        "697l",
        "697m",
        "698",
    ]
    assert sorted(reversed(ordered), key=number_key) == ordered


def test_number_key_is_numeric_not_lexicographic():
    assert number_key("9") < number_key("10") < number_key("100")


def test_letters_sort_before_latin_multiplicatives():
    # Fixed convention (shared with chunk.natural_key): 274g < 274bis.
    assert number_key("274g") < number_key("274bis")


# ------------------------------------------------------------------ range_covers / expand_range


def test_range_covers_integer_range():
    assert range_covers("210:135-149", "210:135")
    assert range_covers("210:135-149", "210:140")
    assert range_covers("210:135-149", "210:149")
    assert not range_covers("210:135-149", "210:134")
    assert not range_covers("210:135-149", "210:150")
    assert not range_covers("210:135-149", "220:140")  # other act


def test_range_covers_numeric_not_lexicographic():
    # "9" > "135" lexicographically, but 9 < 135.
    assert not range_covers("210:135-149", "210:9")
    assert range_covers("210:95-110", "210:100")


def test_range_covers_letter_suffix_ranges():
    assert range_covers("220:257a-257e", "220:257d")
    assert range_covers("220:257a-257e", "220:257a")
    assert range_covers("220:257a-257e", "220:257e")
    assert not range_covers("220:257a-257e", "220:257")  # plain 257 sorts before 257a
    assert not range_covers("220:257a-257e", "220:257f")
    assert not range_covers("220:257a-257e", "220:258")
    assert range_covers("220:697l-697m", "220:697l")
    assert range_covers("220:697l-697m", "220:697m")
    assert not range_covers("220:697l-697m", "220:697k")
    assert not range_covers("220:697l-697m", "220:697n")


def test_range_covers_bare_number_to_letter():
    # "274-274g" starts at the bare article 274 and ends at 274g.
    assert range_covers("220:274-274g", "220:274")
    assert range_covers("220:274-274g", "220:274c")
    assert range_covers("220:274-274g", "220:274g")
    assert not range_covers("220:274-274g", "220:274h")
    assert not range_covers("220:274-274g", "220:275")
    assert not range_covers("220:274-274g", "220:273")
    # Latin multiplicatives sort after all single letters, so 274bis is outside.
    assert not range_covers("220:274-274g", "220:274bis")


def test_range_covers_does_not_guess_suffixed_numbers_inside_integer_ranges():
    # 140a sorts between 140 and 141, but "Art. 135-149" does not say whether an inserted 140a
    # was meant to be part of it: no guess, so integer ranges cover bare integers only.
    for x in ("140a", "140z", "140bis", "135a", "149a", "149bis"):
        assert not range_covers("210:135-149", f"210:{x}"), x
    assert range_covers("210:135-149", "210:136")


def test_range_covers_never_covers_malformed_suffixes():
    # Invalid ids are errors for range_covers, and resolve_id turns them into None.
    with pytest.raises(ArticleIdError):
        range_covers("210:135-149", "210:140xyz")


def test_range_covers_never_matches_ranges_or_other_blocks():
    assert not range_covers("210:135-149", "210:136-140")  # a range is not an article
    assert not range_covers("210:135-149", "210:u1/140")  # transitional namespace
    assert not range_covers("210:135-149", "210:u1")
    assert not range_covers("210:135", "210:135")  # first argument must be a range id


def test_range_covers_transitional_block():
    assert range_covers("220:u12/2-4", "220:u12/3")
    assert not range_covers("220:u12/2-4", "220:u13/3")  # other block
    assert not range_covers("220:u12/2-4", "220:3")  # main text, same number
    assert not range_covers("220:u12/2-4", "220:u12/5")


def test_expand_range_integers():
    out = expand_range("210:135-149")
    assert out == [f"210:{n}" for n in range(135, 150)]


def test_expand_range_letters():
    assert expand_range("220:226a-226d") == [f"220:226{c}" for c in "abcd"]
    assert expand_range("220:697l-697m") == ["220:697l", "220:697m"]
    assert expand_range("210:28d-28f") == ["210:28d", "210:28e", "210:28f"]
    assert expand_range("221.229.1:101a-101c") == [f"221.229.1:101{c}" for c in "abc"]


def test_expand_range_bare_to_letter_includes_bare_number():
    assert expand_range("220:274-274g") == ["220:274", *[f"220:274{c}" for c in "abcdefg"]]


def test_expand_range_transitional():
    assert expand_range("220:u12/2-4") == ["220:u12/2", "220:u12/3", "220:u12/4"]


@pytest.mark.parametrize("bad", ["220:257d", "220:u1", "220:5a-7", "220:5-7bis", "220:5bis-6"])
def test_expand_range_refuses_what_it_cannot_enumerate(bad):
    with pytest.raises(ArticleIdError):
        expand_range(bad)


@pytest.mark.parametrize(
    "range_id",
    ["210:135-149", "220:226a-226d", "220:274-274g", "220:u12/2-4", "220:697l-697m"],
)
def test_expanded_members_are_covered(range_id):
    for member in expand_range(range_id):
        assert range_covers(range_id, member)


# ------------------------------------------------------------------ resolve_id


KNOWN = {
    "210:134",
    "210:135-149",
    "210:150",
    "220:226a-226d",
    "220:226e",
    "220:257a-257e",
    "220:274-274g",
    "220:u12/2-4",
    "220:u12",
    "220:u13/3",
    "220:1",
    "220:257",
    "220:257f",
}


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("210:140", "210:135-149"),  # the motivating case
        ("210:135", "210:135-149"),
        ("210:149", "210:135-149"),
        ("210:134", "210:134"),  # exact id wins
        ("210:150", "210:150"),
        ("210:151", None),
        ("210:140a", None),  # suffixed number inside an integer range: not guessed
        ("210:140z", None),
        ("210:140bis", None),
        ("210:135ff", None),  # malformed suffix
        ("210:140xyz", None),
        ("210:133", None),
        ("220:226b", "220:226a-226d"),
        ("220:226d", "220:226a-226d"),
        ("220:226e", "220:226e"),
        ("220:226f", None),
        ("220:226", None),  # bare 226 is not in 226a-226d
        ("220:257d", "220:257a-257e"),  # letter inside a letter range
        ("220:257", "220:257"),  # exact id exists
        ("220:257f", "220:257f"),
        ("220:257g", None),
        ("220:274", "220:274-274g"),
        ("220:274c", "220:274-274g"),
        ("220:274h", None),
        ("220:274bis", None),
        ("220:274z", None),
        ("220:u12/3", "220:u12/2-4"),
        ("220:u12/5", None),
        ("220:u13/3", "220:u13/3"),
        ("220:u13/2", None),
        ("220:u12", "220:u12"),  # block-only id: exact match only
        ("220:u14", None),
        ("220:3", None),  # main-text 3 is not the transitional range
        ("235.1:140", None),  # act without ranges / unknown act
    ],
)
def test_resolve_id(query, expected):
    assert resolve_id(query, KNOWN) == expected


def test_resolve_id_normalises_the_query():
    assert resolve_id("210:Art. 140", KNOWN) == "210:135-149"
    assert resolve_id("220:257 D", KNOWN) == "220:257a-257e"
    assert resolve_id("220:274 c", KNOWN) == "220:274-274g"


def test_resolve_id_unparsable_query_is_none():
    for bad in ("", "220", "220:", "garbage", "220:???", "220:149-135"):
        assert resolve_id(bad, KNOWN) is None


def test_resolve_id_range_query_needs_exact_match():
    assert resolve_id("210:135-149", KNOWN) == "210:135-149"
    # A sub-range or an overlapping range is not resolved to the containing range: never guess.
    assert resolve_id("210:136-140", KNOWN) is None
    assert resolve_id("210:130-140", KNOWN) is None


def test_resolve_id_ambiguous_overlap_is_none():
    known = {"210:5-10", "210:8-12"}
    assert resolve_id("210:6", known) == "210:5-10"
    assert resolve_id("210:11", known) == "210:8-12"
    assert resolve_id("210:9", known) is None  # covered by two ranges: ambiguous
    assert resolve_id("210:8", known) is None


def test_resolve_id_exact_beats_ambiguity():
    known = {"210:5-10", "210:8-12", "210:9"}
    assert resolve_id("210:9", known) == "210:9"


def test_resolve_id_rejects_a_bare_string_for_known_ids():
    # A str is itself an iterable of characters: "210:135-149" would silently become a set of
    # single characters and resolve nothing.
    with pytest.raises(TypeError, match="not a single string"):
        resolve_id("210:140", "210:135-149")
    with pytest.raises(TypeError, match="not a single string"):
        IdResolver("210:135-149")
    with pytest.raises(TypeError, match="not a single string"):
        resolve_id("210:140", b"210:135-149")


def test_resolve_id_accepts_any_iterable_and_empty_known():
    assert resolve_id("210:140", ["210:135-149"]) == "210:135-149"
    assert resolve_id("210:140", pd.Series(["210:135-149", "210:150"])) == "210:135-149"
    assert resolve_id("210:140", iter(["210:135-149"])) == "210:135-149"
    assert resolve_id("210:140", set()) is None


def test_resolve_id_does_not_cross_acts():
    assert resolve_id("220:140", {"210:135-149"}) is None
    assert resolve_id("210.1:140", {"210:135-149"}) is None


def test_id_resolver_reports_unparsable_known_range_ids():
    r = IdResolver(["210:135-149", "210:149-135", "210:7-"])
    assert r.resolve("210:140") == "210:135-149"
    assert r.skipped == ["210:149-135", "210:7-"]


def test_id_resolver_matches_resolve_id():
    r = IdResolver(KNOWN)
    for q in ("210:140", "220:274c", "220:u12/3", "220:9999", "bad"):
        assert r.resolve(q) == resolve_id(q, KNOWN)


# ------------------------------------------------------------------ corpus-level checks

corpus = pytest.mark.skipif(
    not ARTICLES_PARQUET.exists(),
    reason="data/processed/articles.parquet missing (run `uv run python -m fedlex_bench.chunk`)",
)


@pytest.fixture(scope="module")
def corpus_df() -> pd.DataFrame:
    return pd.read_parquet(ARTICLES_PARQUET)


@corpus
def test_corpus_every_id_parses_and_normalisation_is_idempotent(corpus_df):
    for aid in sorted(set(corpus_df["article_id"])):
        sr, number = parse_article_id(aid)
        assert sr == corpus_df.loc[corpus_df["article_id"] == aid, "sr"].iloc[0]
        assert normalize_article_id(aid) == aid


@corpus
def test_corpus_every_id_resolves_to_itself(corpus_df):
    known = set(corpus_df["article_id"])
    r = IdResolver(known)
    for aid in sorted(known):
        assert r.resolve(aid) == aid


@corpus
def test_corpus_range_ids_are_the_same_in_all_languages(corpus_df):
    by_lang = {
        lang: {a for a in g["article_id"] if "-" in a} for lang, g in corpus_df.groupby("lang")
    }
    assert len({frozenset(v) for v in by_lang.values()}) == 1, by_lang


@corpus
def test_corpus_every_range_id_is_covered(corpus_df):
    """Every range id in the corpus: parses, expands, resolves for all its members and for
    nothing just outside it, and overlaps no other range."""
    for lang, g in corpus_df.groupby("lang"):
        known = set(g["article_id"])
        resolver = IdResolver(known)
        assert resolver.skipped == [], (lang, resolver.skipped)
        ranges = sorted((a for a in known if is_range_id(a)), key=_range_sort_key)
        assert ranges, f"no range ids found for {lang}: the corpus-level check would be vacuous"
        for rid in ranges:
            sr, number = parse_article_id(rid)
            spec = split_number(number)
            assert number_key(spec.start) < number_key(spec.end), rid
            members = expand_range(rid)  # raises if the range shape is not enumerable
            assert len(members) >= 2, rid
            for member in members:
                # a range chunk and a chunk of its own for a member would be contradictory
                assert member not in known, (lang, rid, member)
                assert resolver.resolve(member) == rid, (lang, rid, member)
            # no other range claims any member
            for other in ranges:
                if other != rid:
                    assert not any(range_covers(other, m) for m in members), (rid, other)
            # just outside the range (integer neighbours) it must not resolve to this range
            if spec.start.isdigit() and spec.end.isdigit():
                for outside in (int(spec.start) - 1, int(spec.end) + 1):
                    prefix = f"{sr}:{spec.block + '/' if spec.block else ''}"
                    assert resolver.resolve(f"{prefix}{outside}") != rid, (rid, outside)


@corpus
def test_corpus_no_chunk_hides_inside_the_order_span_of_a_range(corpus_df):
    """Full-order check: no single-article chunk of the same act and block lies anywhere between
    the two ends of a range id in the natural order (``140a`` between 135 and 149 would be a
    chunk the range id silently shadows or fails to cover)."""
    for lang, g in corpus_df.groupby("lang"):
        known = set(g["article_id"])
        singles = [a for a in known if not is_range_id(a)]
        spans: dict[tuple[str, str], list[tuple[tuple, tuple, str]]] = {}
        for rid in (a for a in known if is_range_id(a)):
            sr, number = parse_article_id(rid)
            spec = split_number(number)
            entry = (number_key(spec.start), number_key(spec.end), rid)
            spans.setdefault((sr, spec.block), []).append(entry)
        assert spans
        for aid in sorted(singles):
            sr, number = parse_article_id(aid)
            spec = split_number(number)
            if not spec.start:
                continue  # block-only id
            key = number_key(spec.start)
            for lo, hi, rid in spans.get((sr, spec.block), ()):
                assert not (lo <= key <= hi), (lang, aid, "lies inside", rid)


@corpus
def test_corpus_range_chunks_are_repealed_so_resolution_yields_no_text(corpus_df):
    ranges = corpus_df[corpus_df["article_id"].map(is_range_id)]
    assert not ranges.empty
    assert ranges["is_repealed"].all()
    assert (ranges["text"].str.strip() == "").all()


def _range_sort_key(article_id: str) -> tuple:
    sr, number = parse_article_id(article_id)
    spec = split_number(number)
    return (sr, spec.block, number_key(spec.start))
