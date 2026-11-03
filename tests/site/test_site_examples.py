"""The recorded-examples page (site/examples.html)."""

from __future__ import annotations

import importlib.util
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

from site_fixtures import make_root

from fedlex_bench.serve.notices import NOT_LEGAL_ADVICE, NOT_OFFICIAL_DE, NOT_OFFICIAL_EN
from fedlex_bench.site.__main__ import write_site
from fedlex_bench.site.build import PROJECT_ROOT
from fedlex_bench.site.examples import (
    COMMANDS,
    article_label,
    mismatch_note,
    render_examples,
)

ANSWER_ODD = "Answer with <b>markup</b> & \"quotes\"\nand a second line, Art. 335c CO."


def example(i: int, status="answered", expected="answered", **kw) -> dict:
    base = {
        "lang": ("en", "de", "fr", "it")[i % 4], "question": f"Question number {i}?",
        "expected_status": expected, "status": status, "answer": f"Answer {i}. {ANSWER_ODD}",
        "citations": [f"220:{300 + i}c"] if status == "answered" else [],
        "pointers": ["210:39"] if status == "refused" else [],
        "top_ranked": [f"220:{j}" for j in range(5)], "n_invalid_ids": 0, "parse_failed": False,
        "cost_usd": 0.002 + i / 10000, "latency_s": 2.5 + i,
    }
    base.update(kw)
    return base


def data(**kw) -> dict:
    items = [example(i) for i in range(7)]
    items.append(example(7, status="refused", expected="answered"))
    items.append(example(8, status="refused", expected="refused"))
    items.append(example(9, status="answered", expected="refused"))
    d = {"recorded_at": "2031-01-02", "model": "model-xyz", "embedder_revision": "abc",
         "total_cost_usd": 0.123456, "examples": items}
    d.update(kw)
    return d


def build_examples(tmp_path: Path, d: dict) -> str:
    root, manifest = make_root(tmp_path)
    (root / "demo").mkdir()
    (root / "demo" / "examples.json").write_text(json.dumps(d), encoding="utf-8")
    write_site(root, tmp_path / "out", manifest)
    return (tmp_path / "out" / "examples.html").read_text(encoding="utf-8")


class Articles(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.items: list[dict] = []
        self.refs: list[tuple[str, str, str]] = []
        self.tags: set[str] = set()

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        self.tags.add(tag)
        if tag == "article":
            self.items.append({"status": a["data-status"], "expected": a["data-expected"],
                               "text": []})
        for name, value in attrs:
            if name in ("src", "href") and value:
                self.refs.append((tag, name, value))

    def handle_data(self, text):
        if self.items:
            self.items[-1]["text"].append(text)


def parse(page: str) -> Articles:
    p = Articles()
    p.feed(page)
    return p


def test_notices_title_and_no_live_demo_statement(tmp_path):
    page = build_examples(tmp_path, data())
    for text in (NOT_OFFICIAL_EN, NOT_OFFICIAL_DE, NOT_LEGAL_ADVICE):
        assert page.count(text) == 2
    assert page.index(NOT_OFFICIAL_EN) < page.index("<h1>")
    assert "<title>Recorded examples: the full pipeline</title>" in page
    # statement at the top, with date and model from the file, counts from the data
    assert "There is no live demo." in page
    assert "10 answers recorded once on 2031-01-02" in page
    assert "model-xyz" in page and "0.123456 USD" in page
    assert page.index("There is no live demo.") < page.index('id="example-1"')


def test_ten_examples_with_statuses_from_file(tmp_path):
    d = data()
    page = build_examples(tmp_path, d)
    arts = parse(page).items
    assert len(arts) == 10
    for art, ex in zip(arts, d["examples"], strict=True):
        assert art["status"] == ex["status"] and art["expected"] == ex["expected_status"]
        text = "".join(art["text"])
        shown = "Answered" if ex["status"] == "answered" else "Refused: not in the sources"
        assert shown in text and ex["question"] in text


def test_answer_text_exactly_as_recorded_and_escaped(tmp_path):
    page = build_examples(tmp_path, data())
    assert "<b>markup</b>" not in page
    from html import escape

    assert escape(ANSWER_ODD) in page  # the newline and the markup survive, escaped


def test_citations_parsed_from_ids_without_article_text_or_filestore_links(tmp_path):
    d = data()
    page = build_examples(tmp_path, d)
    assert "SR 220, article 300c (id 220:300c)" in page
    assert "SR 210, article 39 (id 210:39)" in page
    assert article_label("220:257d") == "SR 220, article 257d (id 220:257d)"
    assert "filestore" not in page
    refs = [r for r in parse(page).refs if r[0] == "a" and r[2].startswith("http")]
    assert [r[2] for r in refs] == ["https://www.fedlex.admin.ch"]


def test_notes_computed_from_expected_versus_actual(tmp_path):
    page = build_examples(tmp_path, data())
    arts = parse(page).items
    notes = ["".join(a["text"]) for a in arts]
    assert "Expected: answered. Recorded: refused." in notes[7]
    assert "a miss" in notes[7]
    assert "as intended" in notes[8]  # expected refused, refused
    assert "Expected: refused. Recorded: answered." in notes[9]
    assert "Expected:" not in notes[0]  # matching answered: no note
    assert page.count('class="example differs"') == 2
    assert "2 had a recorded status different from the expected one and are marked" in page
    assert mismatch_note("answered", "answered") == ""


def test_counts_and_costs_come_from_the_data(tmp_path):
    d = data()
    page = build_examples(tmp_path, d)
    answered = sum(x["status"] == "answered" for x in d["examples"])
    assert f"{answered} were answered and {10 - answered} refused" in page
    assert "5 top-ranked articles" in page
    d2 = data()
    d2["examples"][0]["top_ranked"] = ["220:1", "220:2"]
    d2["examples"][0]["n_invalid_ids"] = 3
    page2 = build_examples(tmp_path / "again", d2)
    assert "2 top-ranked articles" in page2 and "3 invalid ids rejected by code" in page2


def test_changed_file_changes_the_page(tmp_path):
    a = build_examples(tmp_path / "a", data())
    b = build_examples(tmp_path / "b", data(recorded_at="2032-05-06"))
    c = build_examples(tmp_path / "c", {**data(), "examples": data()["examples"][:9]})
    assert a != b and "2032-05-06" in b
    assert "9 answers recorded once" in c and 'id="example-10"' not in c


def test_try_it_yourself_commands(tmp_path):
    page = build_examples(tmp_path, data())
    assert "Try the search yourself" in page
    for command in COMMANDS:
        assert command in page
        assert command.startswith("uv run ")
        module = command.split(" -m ")[1].split()[0]
        assert importlib.util.find_spec(module) is not None, module
    assert "needs the corpus built locally" in page


def test_no_external_resources_and_theme(tmp_path):
    page = build_examples(tmp_path, data())
    p = parse(page)
    assert not p.tags & {"script", "iframe", "img", "video", "form", "object", "embed"}
    for tag, _name, value in p.refs:
        parsed = urlparse(value)
        if tag == "a":
            assert value.startswith("#") or value == "index.html" or (
                parsed.scheme == "https" and parsed.netloc == "www.fedlex.admin.ch"), value
        else:
            assert value == "style.css", value  # same stylesheet, light and dark


def test_rebuild_is_byte_identical(tmp_path):
    root, manifest = make_root(tmp_path)
    (root / "demo").mkdir()
    (root / "demo" / "examples.json").write_text(json.dumps(data()), encoding="utf-8")
    write_site(root, tmp_path / "a", manifest)
    write_site(root, tmp_path / "b", manifest)
    assert (tmp_path / "a" / "examples.html").read_bytes() == (
        tmp_path / "b" / "examples.html").read_bytes()


def test_no_examples_file_means_no_page(tmp_path):
    root, manifest = make_root(tmp_path)
    write_site(root, tmp_path / "out", manifest)
    assert not (tmp_path / "out" / "examples.html").exists()


def test_index_links_to_examples_page(tmp_path):
    root, manifest = make_root(tmp_path)
    write_site(root, tmp_path / "out", manifest)
    page = (tmp_path / "out" / "index.html").read_text(encoding="utf-8")
    assert page.count('href="examples.html"') == 2  # nav and links section
    assert "recorded examples (no live demo)" in page
    assert "demo: coming" not in page


def test_real_page_matches_the_real_file():
    path = PROJECT_ROOT / "demo" / "examples.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    page = render_examples(d)
    arts = parse(page).items
    assert len(arts) == len(d["examples"]) == 10
    assert [a["status"] for a in arts] == [x["status"] for x in d["examples"]]
    for x in d["examples"]:
        for i in x["citations"] + x["pointers"]:
            assert article_label(i) in page
        assert re.sub(r"\s+", " ", x["question"]) in re.sub(r"\s+", " ", page).replace(
            "&#x27;", "'")
