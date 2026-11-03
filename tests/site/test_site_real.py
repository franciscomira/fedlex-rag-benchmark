"""The real page: reproduce commands exist, and every table cell equals the committed metrics."""

from __future__ import annotations

import importlib
import importlib.util
import json
import shlex
from pathlib import Path

import pytest
from site_fixtures import parse_cells

from fedlex_bench.site.build import PROJECT_ROOT, build_report, expand_commands, expand_text
from fedlex_bench.site.render import render_page

MANIFEST = PROJECT_ROOT / "site" / "results_manifest.json"

pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="site manifest not present")


@pytest.fixture(scope="module")
def report():
    return build_report(PROJECT_ROOT, MANIFEST)


def reproduce_commands(report) -> list[str]:
    out = []
    for row in report.manifest["reproduce"]:
        for template in row["commands"]:
            out.extend(expand_commands(report, expand_text(report, template)))
    return out


def check_python_module(tokens: list[str], capsys) -> None:
    module = tokens[0]
    target = module + ".__main__" if importlib.util.find_spec(module).submodule_search_locations \
        else module
    spec = importlib.util.find_spec(target)
    assert spec is not None and spec.origin, f"module {module} not found"
    source = Path(spec.origin).read_text(encoding="utf-8")
    assert '__name__ == "__main__"' in source, f"{module} cannot be run with -m"
    args = tokens[1:]
    if "def main(argv" not in source:
        return  # a module with a plain main(): existence is all we can check without running it
    mod = importlib.import_module(target)
    sub = [a for a in args[:1] if not a.startswith("-")]
    with pytest.raises(SystemExit) as exc:
        mod.main([*sub, "--help"])
    assert exc.value.code == 0
    helptext = capsys.readouterr().out
    for flag in (a for a in args if a.startswith("--")):
        assert flag in helptext, f"{module} {' '.join(sub)} has no option {flag}"


def test_every_reproduce_command_references_an_existing_command(report, capsys):
    commands = reproduce_commands(report)
    assert len(commands) >= 10
    runs_dir = PROJECT_ROOT / "eval" / "results"
    for command in commands:
        tokens = shlex.split(command)
        assert tokens[:2] == ["uv", "run"], command
        rest = tokens[2:]
        if rest[0] == "pytest":
            for path in (a for a in rest[1:] if not a.startswith("-")):
                assert (PROJECT_ROOT / path).exists(), path
        elif rest[:2] == ["python", "-m"]:
            check_python_module(rest[2:], capsys)
            if rest[2] == "fedlex_bench.eval" and rest[3] in ("report", "compare"):
                for run_id in rest[4:]:
                    assert (runs_dir / run_id / "metrics.json").exists(), run_id
        elif rest[0] == "python":
            assert (PROJECT_ROOT / rest[1]).is_file(), rest[1]
        else:
            pytest.fail(f"unrecognised command: {command}")


def test_documented_files_exist(report):
    for row in report.manifest["reproduce"]:
        for token in row["needs"].split():
            token = token.strip(".,;()")
            if token.startswith(("notebooks/", "docs/")) and "." in token:
                assert (PROJECT_ROOT / token).exists(), token


def test_real_cells_equal_metrics_json(report):
    cells = parse_cells(render_page(report)).cells
    n = 0
    for (key, metric, lang, _), text in cells.items():
        run = report.runs.get(key)
        if run is None or not run.ok or metric in ("traps", "latency"):
            continue
        block = run.metrics["overall"] if lang is None else run.metrics["by_query_lang"][lang]
        s = block[metric]
        digits = 4 if metric == "cost_usd" else 2
        if not s.get("n"):
            assert text == "n/a"
        else:
            assert text == f"{s['mean']:.{digits}f} [{s['lo']:.{digits}f}, {s['hi']:.{digits}f}]"
        n += 1
    assert n > 60


def test_real_paired_cells_equal_eval_compare(report, capsys):
    """Each paired cell equals the line `eval compare` prints for the same two runs."""
    import fedlex_bench.eval.__main__ as ev

    cells = parse_cells(render_page(report)).cells
    checked = 0
    for section in ("phase3", "phase2"):
        for a, b in report.section_pairs(section):
            comp = report.comparison(a, b)
            if not comp.ready:
                continue
            ev.main(["compare", comp.a.run_id, comp.b.run_id, "--metrics", "recall@10", "mrr"])
            out = capsys.readouterr().out
            for metric in ("recall@10", "mrr"):
                line = next(x for x in out.splitlines() if x.startswith(metric))
                printed = line.split("diff ")[1].split("]")[0] + "]"
                assert cells[(f"{a}-{b}", metric, None, None)].startswith(printed)
                checked += 1
    assert checked >= 10


def test_real_page_shows_no_statute_or_answer_text(report):
    page = render_page(report)
    run = report.runs["B2"]
    answers = [r["answer"] for r in run.items() if r.get("answer")]
    assert answers
    for a in answers[:25]:
        assert a[:60] not in page


def test_real_results_json_roundtrip(report):
    from fedlex_bench.site.build import dump_data

    data = json.loads(dump_data(report))
    assert data["runs"]["T1"]["overall"]["recall@10"]["mean"] == pytest.approx(
        report.runs["T1"].stat("recall@10").mean
    )
