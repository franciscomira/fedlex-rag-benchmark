"""Parsing rules added in review round 1, plus checksummed/idempotent raw download. No network."""

from __future__ import annotations

import json

import httpx
import pytest

from akn_fixtures import ART_1, full_act, wrap
from fedlex_bench import ingest
from fedlex_bench.ingest import normalize_article_number, parse_akn, pick_manifestations


@pytest.fixture(scope="module")
def parsed() -> dict:
    return {a.eid: a for a in parse_akn(full_act())}


# ---------------------------------------------------------------- further parsing rules


def test_dash_variants_in_ranges() -> None:
    for dash in ("-", "‐", "‑", "‒", "–", "—", "−"):
        assert normalize_article_number(f"Art. 135{dash}149") == "135-149"
        assert normalize_article_number(f"Art. 135 {dash} 149") == "135-149"
    assert normalize_article_number("Art. 663dà663h") == "663d-663h"  # glued FR range


def test_paragraph_number_stays_with_nested_list_introduction(parsed: dict) -> None:
    # paragraph > num, content > blockList > listIntroduction: "1" must not be an orphan line
    assert parsed["art_20"].text.splitlines() == [
        "1 Der Kunde kann widerrufen, wenn:",
        "a. am Arbeitsplatz;",
        "b. am Telefon.",
        "2 Zweiter Absatz.",
    ]


def test_historical_block_is_flagged_and_keeps_block_heading(parsed: dict) -> None:
    hist = parsed["disp_u5/art_178"]
    assert hist.is_historical and not hist.is_repealed
    assert hist.block_heading.startswith("Wortlaut der früheren Bestimmungen")
    assert not parsed["disp_u1/art_1"].is_historical
    assert not parsed["art_1"].is_historical and parsed["art_1"].block_heading == ""


@pytest.mark.parametrize(
    "heading",
    [
        "Wortlaut der früheren Bestimmungen des sechsten Titels",
        "Teneur des anciennes dispositions du titre sixième",
        "Testo delle disposizioni previgenti del titolo sesto",
    ],
)
def test_historical_heading_detection_in_all_languages(heading: str) -> None:
    xml = wrap(
        f'<proviso eId="disp_u2"><heading>{heading}</heading>'
        '<article eId="disp_u2/art_1"><num><b>Art. 1</b></num><paragraph eId="p">'
        "<content><p>x</p></content></paragraph></article></proviso>"
    )
    assert parse_akn(xml)[0].is_historical


def test_block_heading_for_transitional_articles(parsed: dict) -> None:
    assert parsed["disp_u1/art_1"].block_heading == "Schlusstitel"
    assert parsed["disp_u12/art_2_4"].block_heading == "Schlussbestimmungen"


def test_amendment_intro_plus_elision_is_elided_like_bare_elision(parsed: dict) -> None:
    a = parsed["disp_u6/art_2"]
    assert a.is_repealed and a.text == "" and "Sono modificate" in a.repeal_note


def test_structural_fingerprint_counts(parsed: dict) -> None:
    assert (parsed["art_20"].n_paragraphs, parsed["art_20"].n_items) == (2, 2)
    assert (parsed["art_1"].n_paragraphs, parsed["art_1"].n_items) == (2, 0)
    assert parsed["art_257_d"].displayed_number == "257d"
    assert parsed["art_12_bis"].displayed_number == "12bis"


# ---------------------------------------------------------------- revisions and raw files


def test_revision_is_compared_numerically_not_lexicographically() -> None:
    rows = [{"lang": "de", "fmt": "xml", "url": f"u/de-xml-{n}.xml"} for n in (9, 10, 2)]
    assert pick_manifestations(rows)["de"]["url"] == "u/de-xml-10.xml"
    assert ingest.revision_of("a/b-fr-xml-12.xml") == 12
    assert ingest.revision_of("a/b.docx") == 0


def test_raw_path_includes_revision_or_url_hash(tmp_path) -> None:
    p3 = ingest.raw_path("220", "de", "2026-10-01", "xml", tmp_path, "https://x/f-xml-3.xml")
    p4 = ingest.raw_path("220", "de", "2026-10-01", "xml", tmp_path, "https://x/f-xml-4.xml")
    assert p3.name == "de_2026-10-01_r3.xml" and p4.name == "de_2026-10-01_r4.xml"
    ph = ingest.raw_path("220", "de", "2026-10-01", "xml", tmp_path, "https://x/no-revision.xml")
    assert ph.name.startswith("de_2026-10-01_h") and ph != p3


def test_validate_akn_rejects_html_error_pages_and_other_xml() -> None:
    ingest.validate_akn(full_act())
    with pytest.raises(ingest.RawDataError):
        ingest.validate_akn(b"<html><body>503 Service Unavailable</body></html>")
    with pytest.raises(ingest.RawDataError):
        ingest.validate_akn(b"<!DOCTYPE html><html><p>unclosed")
    with pytest.raises(ingest.RawDataError):
        ingest.validate_akn(b"<root xmlns='urn:other'><act/></root>")


# ---------------------------------------------------------------- idempotent, checksummed download

LANG3 = ("de", "fr", "it")


def _entry(rev: int = 3) -> dict:
    return {
        "sr": "220",
        "as_of": "2026-10-15",
        "consolidation": "c",
        "consolidation_date": "2026-10-01",
        "consolidation_end": None,
        "work": "w",
        "files": {lang: {"format": "xml", "url": f"u/{lang}-xml-{rev}.xml"} for lang in LANG3},
    }


class _FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


class _FakeClient:
    """Stands in for httpx.Client; serves ``payload`` for every GET and records the URLs."""

    calls: list[str] = []
    payload: bytes = b""

    def __init__(self, *_a, **_k) -> None:
        pass

    def get(self, url: str) -> _FakeResponse:
        _FakeClient.calls.append(url)
        return _FakeResponse(_FakeClient.payload)

    def close(self) -> None:
        pass


@pytest.fixture
def fake_net(monkeypatch):
    _FakeClient.calls = []
    _FakeClient.payload = full_act()
    monkeypatch.setattr(httpx, "Client", _FakeClient)
    return _FakeClient


def _write_manifest(tmp_path, rev: int = 3) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"220": _entry(rev)}), encoding="utf-8")


def test_ensure_raw_downloads_once_records_sha_and_is_idempotent(tmp_path, fake_net) -> None:
    raw = tmp_path / "raw"
    _write_manifest(tmp_path)
    m1 = ingest.ensure_raw(("220",), "2026-10-15", raw_dir=raw)
    assert len(fake_net.calls) == 3
    digest = ingest.sha256_of(full_act())
    assert all(f["sha256"] == digest for f in m1["220"]["files"].values())
    assert (raw / "220" / "de_2026-10-01_r3.xml").exists()
    on_disk = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk == m1  # the tracked manifest (next to raw/, not inside it) was updated
    m2 = ingest.ensure_raw(("220",), "2026-10-15", raw_dir=raw)
    assert len(fake_net.calls) == 3  # second run: fully offline
    assert m2 == m1


def test_ensure_raw_cached_run_never_creates_a_client(tmp_path, monkeypatch, fake_net) -> None:
    raw = tmp_path / "raw"
    _write_manifest(tmp_path)
    ingest.ensure_raw(("220",), "2026-10-15", raw_dir=raw)

    def boom(*_a, **_k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(httpx, "Client", boom)
    ingest.ensure_raw(("220",), "2026-10-15", raw_dir=raw)


def test_ensure_raw_rejects_modified_cache_by_sha(tmp_path, fake_net) -> None:
    raw = tmp_path / "raw"
    _write_manifest(tmp_path)
    ingest.ensure_raw(("220",), "2026-10-15", raw_dir=raw)
    target = raw / "220" / "fr_2026-10-01_r3.xml"
    target.write_bytes(target.read_bytes().replace(b"Abschluss", b"Abschluss!"))  # still valid XML
    with pytest.raises(ingest.RawDataError, match="sha256 mismatch"):
        ingest.ensure_raw(("220",), "2026-10-15", raw_dir=raw)


def test_ensure_raw_replaces_cached_html_error_page(tmp_path, fake_net) -> None:
    raw = tmp_path / "raw"
    _write_manifest(tmp_path)
    bad = raw / "220" / "de_2026-10-01_r3.xml"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"<html><body>Bad gateway</body></html>")
    ingest.ensure_raw(("220",), "2026-10-15", raw_dir=raw)
    assert bad.read_bytes() == full_act()  # re-downloaded
    assert sum("de-xml" in u for u in fake_net.calls) == 1


def test_ensure_raw_refuses_to_store_a_downloaded_html_page(tmp_path, fake_net) -> None:
    fake_net.payload = b"<html><body>Service unavailable</body></html>"
    _write_manifest(tmp_path)
    with pytest.raises(ingest.RawDataError):
        ingest.ensure_raw(("220",), "2026-10-15", raw_dir=tmp_path / "raw")
    assert not list((tmp_path / "raw").rglob("*.xml"))


def test_new_revision_gets_a_new_file_and_never_reuses_old_content(tmp_path, fake_net) -> None:
    raw = tmp_path / "raw"
    _write_manifest(tmp_path, 3)
    ingest.ensure_raw(("220",), "2026-10-15", raw_dir=raw)
    old = (raw / "220" / "de_2026-10-01_r3.xml").read_bytes()
    # the filestore publishes revision 4 with different content
    fake_net.payload = wrap(ART_1)
    _write_manifest(tmp_path, 4)
    m = ingest.ensure_raw(("220",), "2026-10-15", raw_dir=raw)
    assert (raw / "220" / "de_2026-10-01_r4.xml").read_bytes() == wrap(ART_1)
    assert (raw / "220" / "de_2026-10-01_r3.xml").read_bytes() == old  # untouched
    assert m["220"]["files"]["de"]["sha256"] == ingest.sha256_of(wrap(ART_1))
