"""Where an article sits in its act (division / title / chapter / section), from the raw XML."""

from __future__ import annotations

import pytest
from lxml import etree

from fedlex_bench import ingest
from fedlex_bench.benchmark import structure
from fedlex_bench.benchmark.structure import RawStructure, build_structure
from fedlex_bench.ingest import RawDataError

HEAD = (
    '<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0" '
    'xmlns:fedlex="http://fedlex.admin.ch/"><act name="publicLaw"><meta><identification>'
    '<FRBRWork><FRBRname xml:lang="fr" value="Loi fédérale de test du 1er janvier 2000"/>'
    '<FRBRname xml:lang="de" value="Testgesetz vom 1. Januar 2000"/></FRBRWork>'
    "</identification></meta><body>"
)
TAIL = "</body></act></akomaNtoso>"


def doc(inner: str) -> bytes:
    return (HEAD + inner + TAIL).encode("utf-8")


NESTED = doc(
    '<part eId="part_1"><num>Erster Teil:</num><heading>Allgemeines</heading>'
    '<title eId="part_1/tit_2"><num>2. Titel:</num><heading>Der Kauf</heading>'
    '<chapter eId="part_1/tit_2/chap_3"><num>3. Abschnitt:</num><heading>Pflichten</heading>'
    '<level eId="part_1/tit_2/chap_3/lvl_A" fedlex:role="marginal"><num>A.</num>'
    "<heading>Marginal</heading>"
    '<article eId="art_5"><num>Art. 5</num><paragraph eId="art_5/para"><content><p>Text.</p>'
    "</content></paragraph></article></level></chapter></title></part>"
    '<article eId="art_99"><num>Art. 99</num><paragraph eId="art_99/para"><content><p>Flat.</p>'
    "</content></paragraph></article>"
)


def test_path_lists_the_structural_ancestors_from_the_outside_in():
    s = build_structure(etree.fromstring(NESTED), "999", "de")
    assert s.paths["999:5"] == [
        "Erster Teil: Allgemeines",
        "2. Titel: Der Kauf",
        "3. Abschnitt: Pflichten",
    ]


def test_marginal_levels_are_not_structure():
    s = build_structure(etree.fromstring(NESTED), "999", "de")
    assert all("Marginal" not in part for part in s.paths["999:5"])


def test_an_article_outside_any_division_has_an_empty_path():
    s = build_structure(etree.fromstring(NESTED), "999", "de")
    assert s.paths["999:99"] == []


def test_the_act_name_is_the_frbr_name_of_the_language():
    root = etree.fromstring(NESTED)
    assert build_structure(root, "999", "de").act_name == "Testgesetz vom 1. Januar 2000"
    assert build_structure(root, "999", "fr").act_name == "Loi fédérale de test du 1er janvier 2000"
    assert build_structure(root, "999", "it").act_name is None  # no Italian name: not invented


def test_a_missing_heading_is_marked_not_invented():
    xml = doc(
        '<chapter eId="chap_1"><num>1. Kapitel:</num>'
        '<section eId="chap_1/sec_1"><num>1. Abschnitt:</num><heading>Begriffe</heading>'
        '<article eId="art_1"><num>Art. 1</num><paragraph eId="art_1/p"><content><p>T.</p>'
        "</content></paragraph></article></section></chapter>"
    )
    s = build_structure(etree.fromstring(xml), "999", "de")
    assert s.paths["999:1"] == ["1. Kapitel: [heading not available]", "1. Abschnitt: Begriffe"]


def test_transitional_blocks_and_conjunction_numbers():
    xml = doc(
        '<title eId="tit_1"><num>1. Titel:</num><heading>Titel</heading>'
        '<article eId="art_50_51"><num>Art. 50 und 51</num><paragraph eId="p"><content><p>x</p>'
        "</content></paragraph></article>"
        '<article eId="art_7_a"><num>Art. 7a</num><paragraph eId="q"><content><p>y</p>'
        "</content></paragraph></article></title>"
        '<transitional eId="disp_u1"><heading>Schluss</heading>'
        '<article eId="disp_u1/art_1"><num>Art. 1</num><paragraph eId="r"><content><p>z</p>'
        "</content></paragraph></article></transitional>"
    )
    s = build_structure(etree.fromstring(xml), "999", "de")
    assert s.paths["999:50-51"] == ["1. Titel: Titel"]
    assert s.paths["999:7a"] == ["1. Titel: Titel"]
    assert "999:u1/1" not in s.paths and "999:1" not in s.paths  # only main-text articles


# ------------------------------------------------------------------ the provider


def _entry(sr, lang, tmp_path, content: bytes, sha: str | None = None):
    entry = {
        "sr": sr,
        "consolidation_date": "2026-01-01",
        "files": {lang: {"format": "xml", "url": f"https://x/{lang}-xml-1.xml", "sha256": sha}},
    }
    path = ingest.file_path(entry, lang, tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    if sha is None:
        entry["files"][lang]["sha256"] = ingest.sha256_of(content)
    return entry


def test_provider_reads_the_raw_file_after_checking_its_hash(tmp_path):
    manifest = {"999": _entry("999", "de", tmp_path, NESTED)}
    prov = RawStructure(manifest, tmp_path)
    assert prov.path("999:5", "de") == [
        "Erster Teil: Allgemeines",
        "2. Titel: Der Kauf",
        "3. Abschnitt: Pflichten",
    ]
    assert prov.act_name("999", "de") == "Testgesetz vom 1. Januar 2000"
    assert prov.path("999:12345", "de") is None  # unknown article: not available
    assert prov.path("999:99", "de") is None  # in no division: not available
    assert prov.act_name("999", "it") is None


def test_provider_refuses_a_raw_file_that_does_not_match_the_manifest(tmp_path):
    manifest = {"999": _entry("999", "de", tmp_path, NESTED, sha="0" * 64)}
    with pytest.raises(RawDataError, match="sha256"):
        RawStructure(manifest, tmp_path).path("999:5", "de")


def test_provider_with_a_missing_act_or_file_says_not_available(tmp_path):
    prov = RawStructure({}, tmp_path)
    assert prov.path("999:5", "de") is None and prov.act_name("999", "de") is None
    entry = _entry("999", "de", tmp_path, NESTED)
    ingest.file_path(entry, "de", tmp_path).unlink()
    with pytest.raises(FileNotFoundError):
        RawStructure({"999": entry}, tmp_path).path("999:5", "de")


def test_module_exposes_the_structural_element_names():
    assert {"part", "title", "chapter", "section"} <= structure.STRUCTURAL


# ------------------------------------------------------------------ the real corpus

_MANIFEST = ingest.load_manifest()


def _raw_present() -> bool:
    return bool(_MANIFEST) and all(
        ingest.file_path(e, lang).exists()
        for e in _MANIFEST.values()
        for lang in ("de", "fr", "it")
    )


@pytest.mark.skipif(not _raw_present(), reason="data/raw not downloaded")
def test_real_articles_sit_in_the_expected_titles():
    prov = RawStructure()
    expect = {
        ("220:906", "de"): ("Genossenschaft", "Organisation der Genossenschaft"),
        ("220:906", "fr"): ("coopérative", "Organisation de la société"),
        ("220:906", "it"): ("cooperativa", "Organizzazione della società"),
        ("220:818", "de"): (
            "Gesellschaft mit beschränkter Haftung",
            "Organisation der Gesellschaft",
        ),
        ("220:818", "fr"): ("responsabilité limitée", "Organisation de la société"),
        ("220:818", "it"): ("garanzia limitata", "Organizzazione della società"),
    }
    for (aid, lang), (title_word, chapter) in expect.items():
        path = prov.path(aid, lang)
        assert path is not None, (aid, lang)
        assert title_word in path[1] and chapter in path[2], (aid, lang, path)
    assert "Obligationenrecht" in prov.act_name("220", "de")
    assert "obligations" in prov.act_name("220", "fr")
    assert "obbligazioni" in prov.act_name("220", "it")
    # a short act with chapters and sections only
    assert prov.path("235.1:5", "de")[0].endswith("Allgemeine Bestimmungen")


@pytest.mark.skipif(not _raw_present(), reason="data/raw not downloaded")
def test_every_act_has_a_name_in_every_language():
    prov = RawStructure()
    for sr in _MANIFEST:
        for lang in ("de", "fr", "it"):
            assert prov.act_name(sr, lang), (sr, lang)
