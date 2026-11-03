"""The results page: numbers equal the harness outputs, pending rows, notices, no external URLs."""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import pytest
from site_fixtures import FAKE_ANSWER_TEXT, fmt2, make_root, parse_cells, write_run

from fedlex_bench.eval.stats import paired_bootstrap
from fedlex_bench.serve.notices import NOT_LEGAL_ADVICE, NOT_OFFICIAL_DE, NOT_OFFICIAL_EN
from fedlex_bench.site.__main__ import main, write_site
from fedlex_bench.site.build import PROJECT_ROOT, build_report, clopper_pearson
from fedlex_bench.site.data import compare

ALLOWED_LINK_HOSTS = {"github.com", "huggingface.co", "www.fedlex.admin.ch"}


def build(tmp_path: Path, **kw) -> tuple[Path, str, list[str]]:
    root, manifest = make_root(tmp_path, **kw)
    out = tmp_path / "out"
    warnings = write_site(root, out, manifest)
    return root, (out / "index.html").read_text(encoding="utf-8"), warnings


def metrics(root: Path, run_id: str) -> dict:
    return json.loads((root / "eval/results" / run_id / "metrics.json").read_text("utf-8"))


def test_table_numbers_equal_metrics_json(tmp_path):
    root, page, _ = build(tmp_path)
    cells = parse_cells(page).cells
    runs = json.loads((root / "site/results_manifest.json").read_text("utf-8"))["runs"]
    checked = 0
    for key in ("T1", "T2", "Cp", "A3", "Bp"):
        m = metrics(root, runs[key]["id"])
        for metric in ("recall@1", "recall@10", "mrr", "cost_usd"):
            digits = 4 if metric == "cost_usd" else 2
            s = m["overall"][metric]
            expected = f"{s['mean']:.{digits}f} [{s['lo']:.{digits}f}, {s['hi']:.{digits}f}]"
            assert cells[(key, metric, None, None)] == expected
            checked += 1
        assert cells[(key, "recall@10", "en", None)] == fmt2(m["by_query_lang"]["en"]["recall@10"])
        p50, p95 = m["overall"]["latency_p50"], m["overall"]["latency_p95"]
        want = f"{p50:.1f} / {p95:.1f} s" if p50 >= 1 else f"{p50:.2f} / {p95:.2f} s"
        assert cells[(key, "latency", None, None)].startswith(want.split(" / ")[0])
    assert checked == 20
    # answer-level table (phase 2)
    for key in ("B2", "C2", "D2"):
        m = metrics(root, runs[key]["id"])
        for metric in ("fact_recall", "faithfulness", "cit_precision", "over_refusal"):
            assert cells[(key, metric, None, None)] == fmt2(m["overall"][metric])
        assert cells[(key, "traps", None, None)].startswith("1 of 2 [")
    # answer metrics are n/a for the retrieval-only system
    assert cells[("A2", "fact_recall", None, None)] == "n/a"


def test_headline_numbers_come_from_data(tmp_path):
    root, page, _ = build(tmp_path)
    t1 = metrics(root, "30_test_aaaaaa")["overall"]["recall@10"]
    assert fmt2(t1) in page.split('id="phase3"')[0]  # the first section, before the tables
    assert "On 12 answerable test questions" in page


def test_paired_difference_matches_stats_and_eval_compare(tmp_path, capsys, monkeypatch):
    root, page, _ = build(tmp_path)
    report = build_report(root, root / "site/results_manifest.json")
    a, b = report.runs["T1"], report.runs["A3"]
    rows_a = {r["id"]: r for r in a.items()}
    rows_b = {r["id"]: r for r in b.items()}
    ids = sorted(i for i in rows_a if rows_a[i]["recall@10"] is not None)
    d = paired_bootstrap([rows_a[i]["recall@10"] for i in ids],
                         [rows_b[i]["recall@10"] for i in ids],
                         [rows_a[i]["group"] for i in ids])
    expected = f"{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"
    cell = parse_cells(page).cells[("T1-A3", "recall@10", None, None)]
    assert cell.startswith(expected)
    claimable = d["lo"] > 0 or d["hi"] < 0
    assert cell.endswith("not claimable") != claimable
    # the same line `eval compare` prints
    import fedlex_bench.eval.__main__ as ev

    monkeypatch.setattr(ev, "RESULTS", root / "eval" / "results")
    ev.main(["compare", a.run_id, b.run_id, "--metrics", "recall@10"])
    out = capsys.readouterr().out
    assert expected in out


def test_comparison_refused_when_judge_differs(tmp_path):
    root, page, _ = build(tmp_path)
    report = build_report(root, root / "site/results_manifest.json")
    comp = compare(report.runs["B2"], report.runs["A2"])
    assert not comp.ready and "judge_model" in comp.error
    rows = parse_cells(page).pending_rows
    assert any(r.startswith("B2 minus A2") and "not comparable" in r for r in rows)


def test_not_claimable_list_matches_intervals(tmp_path):
    root, page, _ = build(tmp_path)
    report = build_report(root, root / "site/results_manifest.json")
    for comp in report.section_comparisons("phase3"):
        for metric in ("recall@10", "mrr"):
            d = comp.diffs[metric]
            cell = parse_cells(page).cells[(f"{comp.a.key}-{comp.b.key}", metric, None, None)]
            assert cell.endswith("not claimable") == (not d.claimable)
    assert "Not claimable" in page or "excludes 0" in page


def test_pending_rows_do_not_fail_and_are_marked(tmp_path):
    _, page, warnings = build(tmp_path)
    pend = parse_cells(page).pending_rows
    for key in ("P1", "P2", "P3"):
        assert any(r.startswith(key) and "pending" in r for r in pend)
    assert any(r.startswith("P1 minus T1") and r.endswith("(no run yet)") for r in pend)
    assert not warnings  # a null id is expected, not a warning
    assert "available once the runs exist" in page


def test_pending_run_gets_filled_when_id_is_added(tmp_path):
    root, manifest = make_root(tmp_path)
    write_run(root, "40_test_p1pppp", 0.78, config_hash="p1pppp")
    data = json.loads(manifest.read_text("utf-8"))
    data["runs"]["P1"]["id"] = "40_test_p1pppp"
    manifest.write_text(json.dumps(data), encoding="utf-8")
    out = tmp_path / "out"
    write_site(root, out, manifest)
    page = (out / "index.html").read_text("utf-8")
    cells = parse_cells(page)
    m = metrics(root, "40_test_p1pppp")
    assert cells.cells[("P1", "recall@10", None, None)] == fmt2(m["overall"]["recall@10"])
    assert not any(r.startswith("P1 ") for r in cells.pending_rows)
    assert ("P1-T1", "recall@10", None, None) in cells.cells  # the comparison is now computed
    assert "40_test_p1pppp" in page  # the reproduce command lists it


def test_missing_run_folder_is_pending_with_warning(tmp_path):
    root, manifest = make_root(tmp_path, p1_id="2099-01-01_test_nothere")
    out = tmp_path / "out"
    warnings = write_site(root, out, manifest)
    assert any("2099-01-01_test_nothere" in w for w in warnings)
    page = (out / "index.html").read_text("utf-8")
    assert any(r.startswith("P1") and "run folder not found" in r
               for r in parse_cells(page).pending_rows)
    with pytest.raises(SystemExit):
        main(["build", "--root", str(root), "--out", str(out), "--manifest", str(manifest),
              "--strict"])


def test_test_run_without_completed_line_is_not_shown(tmp_path):
    root, manifest = make_root(tmp_path)
    write_run(root, "40_test_p2pppp", 0.7, completed=False, config_hash="p2pppp")
    data = json.loads(manifest.read_text("utf-8"))
    data["runs"]["P2"]["id"] = "40_test_p2pppp"
    manifest.write_text(json.dumps(data), encoding="utf-8")
    report = build_report(root, manifest)
    assert not report.runs["P2"].ok
    assert "completed" in report.runs["P2"].reason


def test_frozen_name_and_hash_shown(tmp_path):
    _, page, _ = build(tmp_path)
    assert "frozen as T1, config aaaaaa" in page


def test_notices_verbatim_in_header_and_footer(tmp_path):
    _, page, _ = build(tmp_path)
    for text in (NOT_OFFICIAL_EN, NOT_OFFICIAL_DE, NOT_LEGAL_ADVICE):
        assert page.count(text) == 2, text  # top and bottom
    assert page.index(NOT_OFFICIAL_EN) < page.index("<h1>")  # visible at the top


class Refs(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.refs: list[tuple[str, str, str]] = []
        self.tags: set[str] = set()

    def handle_starttag(self, tag, attrs):
        self.tags.add(tag)
        for name, value in attrs:
            if name in ("src", "href", "srcset", "action", "poster", "data") and value:
                self.refs.append((tag, name, value))


def test_no_external_resources(tmp_path):
    _, page, _ = build(tmp_path)
    p = Refs()
    p.feed(page)
    assert not p.tags & {"script", "iframe", "img", "video", "audio", "form", "object", "embed"}
    for tag, name, value in p.refs:
        parsed = urlparse(value)
        if tag == "a" and name == "href":
            if parsed.scheme:
                assert parsed.scheme == "https" and parsed.netloc in ALLOWED_LINK_HOSTS, value
            else:
                assert value.startswith("#") or value == "examples.html", value
        else:  # link, and anything else, must be a local file
            assert not parsed.scheme and not parsed.netloc and not value.startswith("//"), value
    css = (tmp_path / "out" / "style.css").read_text("utf-8")
    assert "http" not in css and "@import" not in css


def test_demo_placeholder_text_and_configurable(tmp_path):
    root, manifest = make_root(tmp_path)
    write_site(root, tmp_path / "o1", manifest)
    page1 = (tmp_path / "o1/index.html").read_text("utf-8")
    assert '<a href="examples.html">recorded examples (no live demo)</a>' in page1
    assert "demo: coming" not in page1
    data = json.loads(manifest.read_text("utf-8"))
    data["links"]["demo"] = {"text": "Try the demo", "url": "https://github.com/x/y"}
    manifest.write_text(json.dumps(data), encoding="utf-8")
    write_site(root, tmp_path / "o2", manifest)
    assert '<a href="https://github.com/x/y">Try the demo</a>' in (
        (tmp_path / "o2/index.html").read_text("utf-8"))


def test_no_answer_or_article_text_on_page(tmp_path):
    root, page, _ = build(tmp_path)
    assert FAKE_ANSWER_TEXT not in page
    assert FAKE_ANSWER_TEXT not in (tmp_path / "out" / "results_data.json").read_text("utf-8")


def test_build_is_deterministic(tmp_path):
    root, manifest = make_root(tmp_path)
    write_site(root, tmp_path / "a", manifest)
    write_site(root, tmp_path / "b", manifest)
    for name in ("index.html", "style.css", "results_data.json"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_page_has_no_hand_typed_digits_in_prose():
    """Prose comes from templates; digits in it are only names such as T1 or Phase 3."""
    src = (PROJECT_ROOT / "src/fedlex_bench/site/render.py").read_text("utf-8")
    prose = re.findall(r'^\s+(?:f?"[^"]*"|f?\'[^\']*\')\s*$', src, flags=re.M)
    allowed = re.compile(r"(Phase \d|section \d|\bT[12]\b|\b[A-D]'|\bP[123]\b|\d\.\s|<h\d|"
                         r"\{[^}]*\}|<[^>]*>|&#\d+;|utf-8|data-|95%|@\d+|(?<![0-9.])0(?![0-9.])|0\.5|(?<![0-9.])1(?![0-9.])|int8)")
    offenders = []
    for line in prose:
        stripped = allowed.sub("", line)
        if re.search(r"\d", stripped):
            offenders.append(line.strip())
    assert not offenders, offenders


def test_clopper_pearson_matches_phase2_report():
    lo, hi = clopper_pearson(10, 15)
    assert (round(lo, 2), round(hi, 2)) == (0.38, 0.88)
    lo, hi = clopper_pearson(11, 15)
    assert (round(lo, 2), round(hi, 2)) == (0.45, 0.92)
    assert clopper_pearson(0, 5)[0] == 0.0 and clopper_pearson(5, 5)[1] == 1.0


def test_unknown_run_in_manifest_is_an_error(tmp_path):
    root, manifest = make_root(tmp_path)
    data = json.loads(manifest.read_text("utf-8"))
    data["sections"]["phase4"]["runs"].append("ZZ")
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SystemExit):
        build_report(root, manifest)
