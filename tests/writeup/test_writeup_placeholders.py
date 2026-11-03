"""Placeholders: values equal the harness data; every problem fails the build with its line."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from site_fixtures import fmt2, make_root
from writeup_fixtures import make_project

from fedlex_bench.eval.stats import paired_bootstrap
from fedlex_bench.site.build import build_report
from fedlex_bench.writeup.build import build_post
from fedlex_bench.writeup.placeholders import BuildError


def metrics(root: Path, run_id: str) -> dict:
    return json.loads((root / "eval/results" / run_id / "metrics.json").read_text("utf-8"))


def fill(root: Path, manifest: Path, template: str) -> str:
    report = build_report(root, manifest)
    return build_post(report, "# T\n\n" + template, root).markdown


def body(markdown: str) -> str:
    """The filled text between the opening notices and the closing rule."""
    middle = markdown.split("\n\n> ", 1)[1].split("\n\n", 1)[1]
    return middle.rsplit("\n\n---\n\n", 1)[0].strip()


def expect_errors(root: Path, manifest: Path, template: str) -> list[str]:
    with pytest.raises(BuildError) as exc:
        fill(root, manifest, template)
    return exc.value.errors


@pytest.fixture()
def project(tmp_path):
    return make_project(tmp_path)


def test_v_and_m_equal_metrics_json(project):
    root, manifest = project
    t1 = metrics(root, "30_test_aaaaaa")["overall"]
    out = body(fill(root, manifest, "{{v T1 recall@10}} | {{m T1 recall@10}} | {{v T1 mrr}}"))
    assert out == f"{fmt2(t1['recall@10'])} | {t1['recall@10']['mean']:.2f} | {fmt2(t1['mrr'])}"


def test_language_cell(project):
    root, manifest = project
    en = metrics(root, "30_test_aaaaaa")["by_query_lang"]["en"]["recall@10"]
    assert body(fill(root, manifest, "{{v T1 recall@10 en}}")) == fmt2(en)
    assert body(fill(root, manifest, "{{v T1 recall@10 EN}}")) == fmt2(en)


def test_cost_uses_four_digits_and_latency_is_p50(project):
    root, manifest = project
    m = metrics(root, "30_test_aaaaaa")["overall"]
    out = body(fill(root, manifest, "{{cost T1}} | {{lat T1}} | {{v T1 cost_usd}}"))
    c = m["cost_usd"]
    assert out == (f"{c['mean']:.4f} | {m['latency_p50']:.1f} s | "
                   f"{c['mean']:.4f} [{c['lo']:.4f}, {c['hi']:.4f}]")


def test_paired_difference_and_verdict_equal_the_bootstrap(project):
    root, manifest = project
    report = build_report(root, manifest)
    a, b = report.runs["T1"], report.runs["A3"]
    rows_a = {r["id"]: r for r in a.items()}
    rows_b = {r["id"]: r for r in b.items()}
    ids = sorted(i for i in rows_a if rows_a[i].get("recall@1") is not None)
    d = paired_bootstrap([rows_a[i]["recall@1"] for i in ids],
                         [rows_b[i]["recall@1"] for i in ids],
                         [rows_a[i]["group"] for i in ids])
    expected = f"{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"
    word = "claimable" if d["lo"] > 0 or d["hi"] < 0 else "not claimable"
    out = body(fill(root, manifest, "{{d T1 A3 recall@1}} | {{claim T1 A3 recall@1}}"))
    assert out == f"{expected} | {word}"


def test_ratios_counts_and_facts(project):
    root, manifest = project
    t2, cp = (metrics(root, i)["overall"] for i in ("30_test_t2tttt", "30_test_cpcccc"))
    cheaper = f"{(1 - cp['cost_usd']['mean'] / t2['cost_usd']['mean']) * 100:.0f}%"
    faster = f"{t2['latency_p50'] / cp['latency_p50']:.1f} times"
    p2 = metrics(root, "40_test_p2pppp")
    out = body(fill(root, manifest,
                    "{{pct_cheaper Cp T2}} | {{x_faster Cp T2}} | {{traps P2}} | {{n P2}} | "
                    "{{f corpus_acts}}"))
    assert out == (f"{cheaper} | {faster} | {p2['traps']['correct']} of {p2['traps']['n']} | "
                   f"{p2['n_answerable']} | 8")


def test_placeholders_inside_code_are_left_alone(project):
    root, manifest = project
    out = fill(root, manifest, "Use `{{v NOPE x}}` here.\n\n```\n{{also nothing}}\n```")
    assert "`{{v NOPE x}}`" in out and "{{also nothing}}" in out


def test_unknown_kind_run_metric_language_fact_all_fail_with_line(project):
    root, manifest = project
    template = "\n".join([
        "ok {{v T1 recall@10}}",              # line 3 of the file
        "{{wat T1}}",                        # 4
        "{{v ZZ recall@10}}",                # 5
        "{{v T1 nosuch}}",                   # 6
        "{{v T1 recall@10 xx}}",             # 7
        "{{f missing_key}}",                 # 8
        "{{v T1}}",                          # 9
        "{{}}",                              # 10
        "{{v T1 recall@10",                  # 11
    ])
    errors = expect_errors(root, manifest, template)
    joined = "\n".join(errors)
    for line, needle in ((4, "unknown placeholder kind 'wat'"), (5, "unknown run 'ZZ'"),
                         (6, "no metric 'nosuch'"), (7, "unknown language 'xx'"),
                         (8, "no fact 'missing_key'"), (9, "'v' takes 2 or 3"),
                         (10, "empty placeholder"), (11, "malformed")):
        assert any(e.startswith(f"docs/writeup/post.template.md:{line}:") and needle in e
                   for e in errors), (line, needle, joined)
    assert not any(":3:" in e for e in errors)  # the good line is not an error
    assert any("{{v T1 nosuch}}" in e for e in errors)  # the placeholder is named


def test_pending_run_fails_the_build(tmp_path):
    root, manifest = make_root(tmp_path)  # P1 to P3 have no run id here
    errors = expect_errors(root, manifest, "{{v P2 recall@10}}")
    assert any("run 'P2' is pending" in e for e in errors)


def test_pending_run_makes_pairs_and_tables_fail(tmp_path):
    root, manifest = make_root(tmp_path)
    errors = expect_errors(root, manifest, "{{d P2 T1 mrr}}\n\n{{table phase4}}")
    assert any("pending" in e and ":3:" in e for e in errors)
    assert any("pending" in e and ":5:" in e for e in errors)


def test_judge_mismatch_refuses_the_pair(project):
    root, manifest = project
    errors = expect_errors(root, manifest, "{{d B2 A2 mrr}}")  # B2 judged, A2 not
    assert any("cannot pair" in e and "judge_model" in e for e in errors)


def test_traps_and_cost_ratios_fail_instead_of_printing_nonsense(project):
    root, manifest = project
    errors = expect_errors(root, manifest,
                           "{{traps T1}}\n\n{{pct_cheaper T2 Cp}}\n\n{{x_faster T2 Cp}}")
    joined = " ".join(errors)
    assert "has no trap items" in joined
    assert "is not cheaper" in joined and "is not faster" in joined


def test_block_placeholders_must_stand_alone_and_exist(project):
    root, manifest = project
    errors = expect_errors(root, manifest,
                           "text {{figure fig_recall}}\n\n{{figure nope}}\n\n{{table nope}}")
    joined = " ".join(errors)
    assert "must stand alone" in joined
    assert "unknown figure 'nope'" in joined and "unknown table 'nope'" in joined


def test_missing_facts_file_fails(project):
    root, manifest = project
    (root / "docs/writeup/facts.json").unlink()
    errors = expect_errors(root, manifest, "{{f corpus_acts}}")
    assert any("facts file not found" in e for e in errors)


def test_a_title_is_required(project):
    root, manifest = project
    report = build_report(root, manifest)
    with pytest.raises(BuildError):
        build_post(report, "No title here.\n", root)
