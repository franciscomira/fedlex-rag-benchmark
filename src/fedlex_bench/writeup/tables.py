"""Generated tables for ``{{table NAME}}``; cells use the formatting of the results page."""

from __future__ import annotations

from dataclasses import dataclass

from fedlex_bench.site.build import clopper_pearson
from fedlex_bench.site.data import Run, fmt_seconds, fmt_stat
from fedlex_bench.site.render import COST_COL, RETRIEVAL_COLS

DASH = "—"  # a metric that cannot exist for this kind of run (no answer step)
LANGS = ("en", "de", "fr", "it")
ANSWER_ONLY = {"fact_recall", "faithfulness", "cit_precision", "over_refusal"}


class TableError(Exception):
    """A cell could not be filled; the message names the run and metric."""


@dataclass(frozen=True)
class Row:
    label: str
    name: str
    cells: tuple[str, ...]


@dataclass(frozen=True)
class Table:
    name: str
    headers: tuple[str, ...]  # without the first (system) column
    rows: tuple[Row, ...]
    first_header: str = "System"


# Column kinds: ("stat", header, metric, lang, digits), ("traps", header), ("p50", header).
Col = tuple


def _stat_cell(run: Run, metric: str, lang: str | None, digits: int) -> str:
    stat = run.stat(metric, lang)
    if stat is None:
        if metric in ANSWER_ONLY and not run.has_answers:
            return DASH
        where = f" for language {lang}" if lang else ""
        raise TableError(f"run {run.key} has no metric '{metric}'{where}")
    return fmt_stat(stat, digits)


def _traps_cell(run: Run) -> str:
    t = run.traps
    if t is None:
        if not run.has_answers:
            return DASH
        raise TableError(f"run {run.key} has no trap items")
    k, n = t
    lo, hi = clopper_pearson(k, n)
    return f"{k} of {n} [{lo:.2f}, {hi:.2f}]"


def _p50_cell(run: Run) -> str:
    p50 = run.scalar("latency_p50")
    if p50 is None:
        raise TableError(f"run {run.key} has no latency_p50")
    return f"{fmt_seconds(p50)} s"


def build_table(name: str, keys: list[str], cols: list[Col], runs: dict[str, Run],
                first_header: str = "System") -> Table:
    rows = []
    for key in keys:
        run = runs[key]
        cells = []
        for col in cols:
            if col[0] == "stat":
                cells.append(_stat_cell(run, col[2], col[3], col[4]))
            elif col[0] == "traps":
                cells.append(_traps_cell(run))
            else:
                cells.append(_p50_cell(run))
        rows.append(Row(run.label, run.name, tuple(cells)))
    return Table(name, tuple(c[1] for c in cols), tuple(rows), first_header)


def _stat_cols(cols) -> list[Col]:
    return [("stat", header, metric, lang, digits) for header, metric, lang, digits in cols]


# name -> (run keys, columns, header of the first column)
SPECS: dict[str, tuple[list[str], list[Col], str]] = {
    "phase3": (
        ["A3", "Bp", "T1", "T2", "Cp"],
        [*_stat_cols(RETRIEVAL_COLS), *_stat_cols([COST_COL]), ("p50", "Latency p50")],
        "System",
    ),
    "phase4": (
        ["P1", "P2", "P3"],
        [
            ("stat", "Recall@10", "recall@10", None, 2),
            ("stat", "Fact recall", "fact_recall", None, 2),
            ("stat", "Faithfulness", "faithfulness", None, 2),
            ("traps", "Traps handled (exact 95% interval)"),
            *_stat_cols([COST_COL]),
            ("p50", "Latency p50"),
        ],
        "System",
    ),
    "by_language": (
        ["A3", "T1", "T2", "Cp"],
        [("stat", f"{lang.upper()} recall@10", "recall@10", lang, 2) for lang in LANGS],
        "System",
    ),
}


TITLES = {
    "phase3": "Retrieval results of the Phase 3 systems",
    "phase4": "Results of the shipped configuration",
    "by_language": "Recall@10 by language of the question",
}


def table_names() -> list[str]:
    return sorted(SPECS)


def make_table(name: str, runs: dict[str, Run]) -> Table:
    if name not in SPECS:
        raise TableError(f"unknown table '{name}' (known: {', '.join(table_names())})")
    keys, cols, first = SPECS[name]
    for key in keys:
        run = runs.get(key)
        if run is None:
            raise TableError(f"unknown run '{key}' needed by table '{name}'")
        if not run.ok:
            raise TableError(f"run '{key}' needed by table '{name}' is pending ({run.reason})")
    return build_table(name, keys, cols, runs, first)


def table_markdown(table: Table) -> str:
    def esc(s: str) -> str:
        return s.replace("|", "\\|")

    head = "| " + " | ".join(esc(h) for h in (table.first_header, *table.headers)) + " |"
    sep = "|---|" + "---:|" * len(table.headers)
    body = [
        "| " + " | ".join(esc(c) for c in (f"**{r.label}** {r.name}", *r.cells)) + " |"
        for r in table.rows
    ]
    return "\n".join([head, sep, *body])
