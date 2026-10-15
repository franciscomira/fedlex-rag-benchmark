"""Smoke tests on the real Fedlex snapshot. Skipped when ``data/raw`` has not been downloaded
(run ``uv run python -m fedlex_bench.chunk`` first). No network access."""

from __future__ import annotations

import re

import pytest

from fedlex_bench import ingest
from fedlex_bench.ingest import LANGS, RAW_DIR, file_path, load_manifest, parse_akn

MANIFEST = load_manifest(ingest.MANIFEST_PATH)


def _files_present() -> bool:
    if not MANIFEST:
        return False
    return all(file_path(MANIFEST[sr], lang).exists() for sr in MANIFEST for lang in LANGS)


pytestmark = pytest.mark.skipif(
    not RAW_DIR.exists() or not _files_present(),
    reason="data/raw not downloaded (run `uv run python -m fedlex_bench.chunk`)",
)


@pytest.fixture(scope="module")
def snapshot() -> dict[tuple[str, str], list]:
    return {
        (sr, lang): parse_akn(file_path(MANIFEST[sr], lang)) for sr in MANIFEST for lang in LANGS
    }


def _bases(number: str) -> set[int]:
    """Integer base numbers covered by an article number: '257d' -> {257}, '135-149' ->
    {135..149}, '226a-226d' -> {226}, '274-274g' -> {274}."""
    parts = number.split("-")
    nums = [int(m.group(0)) for p in parts if (m := re.match(r"\d+", p))]
    if len(nums) == 2:
        return set(range(nums[0], nums[1] + 1))
    return set(nums)


def test_raw_files_match_recorded_sha256() -> None:
    for sr, entry in MANIFEST.items():
        for lang in LANGS:
            f = entry["files"][lang]
            assert f.get("sha256"), f"SR {sr}/{lang}: no sha256 in manifest"
            assert ingest.sha256_of(file_path(entry, lang)) == f["sha256"]


def test_counts_are_equal_across_languages(snapshot) -> None:
    for sr in MANIFEST:
        counts = {lang: len(snapshot[(sr, lang)]) for lang in LANGS}
        assert len(set(counts.values())) == 1, (sr, counts)
        ids = {lang: [a.number for a in snapshot[(sr, lang)]] for lang in LANGS}
        assert ids["de"] == ids["fr"] == ids["it"], sr


@pytest.mark.parametrize(("sr", "last"), [("220", 1186), ("210", 977)])
def test_body_numbers_end_at_the_last_article_without_gaps(snapshot, sr, last) -> None:
    if sr not in MANIFEST:
        pytest.skip(f"SR {sr} not in manifest")
    for lang in LANGS:
        covered: set[int] = set()
        for a in snapshot[(sr, lang)]:
            if a.kind == "article":
                covered |= _bases(a.number)
        assert max(covered) == last, (sr, lang, max(covered))
        missing = sorted(set(range(1, last + 1)) - covered)
        assert not missing, f"SR {sr}/{lang}: base numbers without an article: {missing[:20]}"


def test_no_text_line_is_a_bare_paragraph_or_list_marker(snapshot) -> None:
    marker = re.compile(r"^(\d+[a-z]*|[a-z]|[ivx]+)\.?$")
    orphans = [
        (sr, lang, a.number, line)
        for (sr, lang), arts in snapshot.items()
        for a in arts
        for line in a.text.split("\n")
        if marker.match(line)
    ]
    assert not orphans, orphans[:10]


def test_flags_agree_across_languages(snapshot) -> None:
    for sr in MANIFEST:
        flags = {
            lang: [(a.number, a.is_repealed, a.is_historical) for a in snapshot[(sr, lang)]]
            for lang in LANGS
        }
        assert flags["de"] == flags["fr"] == flags["it"], sr


def test_zgb_historical_block_is_flagged_in_all_languages(snapshot) -> None:
    if "210" not in MANIFEST:
        pytest.skip("SR 210 not in manifest")
    for lang in LANGS:
        hist = [a for a in snapshot[("210", lang)] if a.is_historical]
        assert len(hist) == 74
        assert all(a.number.startswith("u2/") for a in hist)
