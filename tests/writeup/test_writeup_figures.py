"""Figures and generated tables: numbers from the run data, accessible, deterministic."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from writeup_fixtures import make_project

from fedlex_bench.site.build import build_report
from fedlex_bench.site.data import fmt_seconds
from fedlex_bench.writeup.figures import (
    COST_KEYS,
    ENGLISH_KEYS,
    FAMILY,
    FIGURE_NAMES,
    POINT_LABEL,
    RECALL_KEYS,
    FigureError,
    make_figures,
    nice_ticks,
    place_label,
)
from fedlex_bench.writeup.tables import DASH, make_table, table_markdown

SVG = "{http://www.w3.org/2000/svg}"


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root, manifest = make_project(tmp_path_factory.mktemp("fig"))
    return root, build_report(root, manifest)


def metrics(root: Path, run_id: str) -> dict:
    return json.loads((root / "eval/results" / run_id / "metrics.json").read_text("utf-8"))


def test_three_figures_are_valid_accessible_svg(project):
    _, report = project
    figures = make_figures(report.runs)
    assert tuple(sorted(figures)) == tuple(sorted(FIGURE_NAMES))
    for name, fig in figures.items():
        tree = ET.fromstring(fig.standalone)
        assert tree.tag == f"{SVG}svg" and tree.get("role") == "img"
        title, desc = tree.find(f"{SVG}title"), tree.find(f"{SVG}desc")
        assert title is not None and title.text and desc is not None and desc.text
        assert tree.get("aria-labelledby") == f"{title.get('id')} {desc.get('id')}"
        assert title.get("id").startswith(name)
        assert tree.get("viewBox")
        # inline form has no namespace declaration (HTML5 inline SVG) and no background
        assert "xmlns" not in fig.inline and "<rect class=\"wf-bgrect\"" not in fig.inline
        assert "<rect class=\"wf-bgrect\"" in fig.standalone


def test_colours_follow_page_variables_in_light_and_dark(project):
    _, report = project
    svg = make_figures(report.runs)["fig_recall"].standalone
    assert "var(--text,#1c1b19)" in svg and "var(--text,#ece9e2)" in svg
    assert "prefers-color-scheme:dark" in svg
    assert re.search(r"fill:#|stroke:#", svg) is None  # series colours only via variables


def test_recall_figure_values_and_order(project):
    root, report = project
    fig = make_figures(report.runs)["fig_recall"]
    ids = {"A3": "30_test_a3aaaa", "Bp": "30_test_bpbbbb", "T1": "30_test_aaaaaa",
           "T2": "30_test_t2tttt", "Cp": "30_test_cpcccc", "P2": "40_test_p2pppp"}
    assert RECALL_KEYS == ["A3", "Bp", "T1", "T2", "Cp", "P2"]
    positions = []
    for key in RECALL_KEYS:
        s = metrics(root, ids[key])["overall"]["recall@10"]
        run = report.runs[key]
        text = (f"{run.label} {s['mean']:.2f}, 95% interval {s['lo']:.2f} to {s['hi']:.2f}")
        assert text in fig.desc
        assert f">{s['mean']:.2f}<" in fig.inline
        assert f"[{s['lo']:.2f}, {s['hi']:.2f}]" in fig.inline
        positions.append(fig.desc.index(text))
    assert positions == sorted(positions)


def test_whisker_geometry_matches_the_interval(project):
    root, report = project
    fig = make_figures(report.runs)["fig_recall"]
    s = metrics(root, "30_test_aaaaaa")["overall"]["recall@10"]
    x0, x1 = 14.0, 240.0
    lo, hi = x0 + (x1 - x0) * s["lo"], x0 + (x1 - x0) * s["hi"]
    assert f'x1="{lo:.1f}"' in fig.inline and f'x2="{hi:.1f}"' in fig.inline


def test_english_figure_uses_the_english_cell(project):
    root, report = project
    fig = make_figures(report.runs)["fig_english"]
    assert ENGLISH_KEYS == ["A3", "T1", "T2", "Cp"]
    for key, rid in (("A3", "30_test_a3aaaa"), ("T1", "30_test_aaaaaa")):
        s = metrics(root, rid)["by_query_lang"]["en"]["recall@10"]
        assert f"{report.runs[key].label} {s['mean']:.2f}, 95% interval" in fig.desc


def test_cost_latency_axes_and_values(project):
    root, report = project
    fig = make_figures(report.runs)["fig_cost_latency"]
    assert COST_KEYS == ["A3", "T1", "T2", "Cp", "P2", "P3"]
    assert "Horizontal axis: cost per question" in fig.desc
    assert "Vertical axis: median latency in seconds" in fig.desc
    assert "Cost per question (USD" in fig.inline
    m = metrics(root, "30_test_t2tttt")["overall"]
    assert (f"{report.runs['T2'].label} {m['cost_usd']['mean']:.4f} USD, median "
            f"{fmt_seconds(m['latency_p50'])} s") in fig.desc
    for key in COST_KEYS:  # every point carries a direct label
        assert f">{POINT_LABEL[key]}</text>" in fig.inline


def test_points_are_told_apart_by_shape_not_only_colour(project):
    _, report = project
    inline = make_figures(report.runs)["fig_cost_latency"].inline
    families = {FAMILY[k] for k in COST_KEYS}
    assert families == {"base", "llm", "tuned", "shipped"}
    assert "<circle" in inline and "<rect" in inline and inline.count("<polygon") >= 3


def test_figures_are_deterministic(project):
    _, report = project
    a, b = make_figures(report.runs), make_figures(report.runs)
    assert {k: v.standalone for k, v in a.items()} == {k: v.standalone for k, v in b.items()}


def test_a_pending_run_fails_the_figures(tmp_path):
    from site_fixtures import make_root

    root, manifest = make_root(tmp_path)
    with pytest.raises(FigureError, match="run 'P2' is pending"):
        make_figures(build_report(root, manifest).runs)


def test_label_placement_avoids_overlaps_and_stays_inside():
    placed = [(0.0, 90.0, 18.0, 108.0), (0.0, 95.0, 18.0, 113.0)]
    others = [(9.0, 99.0), (9.0, 104.0)]
    bounds = (4.0, 40.0, 380.0, 230.0)
    boxes = []
    for x, y in others:
        lx, ly, anchor, box, dy = place_label(x, y, 120.0, placed + boxes, bounds, others)
        assert bounds[0] <= box[0] and box[2] <= bounds[2] and bounds[1] <= box[1]
        assert all(not (box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3])
                   for b in placed + boxes)
        boxes.append(box)
    assert boxes[0][1] < boxes[1][1]  # the upper marker's label is the upper one


def test_nice_ticks():
    assert nice_ticks(0.0036) == ([0.0, 0.001, 0.002, 0.003], 3)
    ticks, decimals = nice_ticks(12.0)
    assert ticks[0] == 0 and ticks[-1] <= 12 and decimals == 0


# ---------------------------------------------------------------- tables


def test_phase3_table_equals_the_results_page_cells(project):
    root, report = project
    table = make_table("phase3", report.runs)
    assert [r.label for r in table.rows] == [report.runs[k].label
                                             for k in ("A3", "Bp", "T1", "T2", "Cp")]
    assert len(table.rows) == 5 and len(table.headers) == 6
    t1 = next(r for r in table.rows if r.label == report.runs["T1"].label)
    m = metrics(root, "30_test_aaaaaa")
    o = m["overall"]
    s = o["recall@10"]
    assert t1.cells[1] == f"{s['mean']:.2f} [{s['lo']:.2f}, {s['hi']:.2f}]"
    en = m["by_query_lang"]["en"]["recall@10"]
    assert t1.cells[3] == f"{en['mean']:.2f} [{en['lo']:.2f}, {en['hi']:.2f}]"
    c = o["cost_usd"]
    assert t1.cells[4] == f"{c['mean']:.4f} [{c['lo']:.4f}, {c['hi']:.4f}]"
    assert t1.cells[5] == f"{fmt_seconds(o['latency_p50'])} s"


def test_phase4_table_marks_what_a_retrieval_only_run_cannot_have(project):
    root, report = project
    table = make_table("phase4", report.runs)
    p1, p2 = table.rows[0], table.rows[1]
    assert p1.cells[1] == p1.cells[2] == p1.cells[3] == DASH
    assert "of" in p2.cells[3] and p2.cells[1] != DASH
    md = table_markdown(table)
    assert md.splitlines()[1] == "|---|" + "---:|" * 6


def test_by_language_table(project):
    root, report = project
    table = make_table("by_language", report.runs)
    assert table.headers == ("EN recall@10", "DE recall@10", "FR recall@10", "IT recall@10")
    s = metrics(root, "30_test_t2tttt")["by_query_lang"]["fr"]["recall@10"]
    row = next(r for r in table.rows if r.label == report.runs["T2"].label)
    assert row.cells[2] == f"{s['mean']:.2f} [{s['lo']:.2f}, {s['hi']:.2f}]"
