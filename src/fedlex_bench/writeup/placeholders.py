"""Placeholders of the post template: ``{{kind args}}`` filled from harness data.

Every value comes from the ``Report`` of the results page (same runs, same paired bootstrap) or
from docs/writeup/facts.json. An unknown placeholder, an unknown or pending run or a missing
metric is an error that names the placeholder and its line; nothing is ever printed as blank or
"n/a". Placeholders are not expanded inside fenced code blocks or inline code.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from fedlex_bench.site.build import Report
from fedlex_bench.site.data import Comparison, Diff, Run, compare, fmt_diff
from fedlex_bench.site.render import Fmt
from fedlex_bench.writeup.figures import FIGURE_NAMES, Figure, FigureError, make_figures
from fedlex_bench.writeup.tables import LANGS, Table, TableError, make_table

OPEN, CLOSE = "", ""  # delimit a token that stands for one placeholder
TOKEN = re.compile(f"{OPEN}(\\d+){CLOSE}")
PLACEHOLDER = re.compile(r"\{\{(.*?)\}\}")
FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
CODE_SPAN = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)")

# kind -> (min, max) number of arguments
ARITY = {
    "v": (2, 3), "m": (2, 3), "d": (3, 3), "claim": (3, 3), "cost": (1, 1), "lat": (1, 1),
    "pct_cheaper": (2, 2), "x_faster": (2, 2), "traps": (1, 1), "n": (1, 1), "f": (1, 1),
    "table": (1, 1), "figure": (1, 1),
}
BLOCK_KINDS = {"table", "figure"}


class PlaceholderError(Exception):
    """One placeholder could not be filled."""


class BuildError(Exception):
    """The build failed; ``errors`` holds one line per problem."""

    def __init__(self, errors: list[str]):
        super().__init__("\n".join(errors))
        self.errors = errors


@dataclass
class Item:
    """One placeholder found in the template."""

    kind: str
    args: tuple[str, ...]
    line: int
    raw: str
    value: str = ""  # inline kinds
    table: Table | None = None
    figure: Figure | None = None


@dataclass
class Tokenized:
    """The template with each placeholder replaced by a token, plus the filled items."""

    lines: list[str]
    items: list[Item] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def digits_for(metric: str) -> int:
    return 4 if metric == "cost_usd" else 2


class Resolver:
    def __init__(self, report: Report, facts_path: Path | None = None):
        self.report = report
        self.fmt = Fmt(report)
        self.facts_path = facts_path
        self._facts: dict | None = None
        self._cmp: dict[tuple[str, str, str], Comparison] = {}
        self._figures: dict[str, Figure] | None = None
        self._figures_error: str | None = None

    # ---- lookups

    def run(self, key: str) -> Run:
        run = self.report.runs.get(key)
        if run is None:
            raise PlaceholderError(
                f"unknown run '{key}' (manifest runs: {', '.join(sorted(self.report.runs))})")
        if not run.ok:
            raise PlaceholderError(f"run '{key}' is pending ({run.reason or 'no run yet'})")
        return run

    def _lang(self, lang: str | None) -> str | None:
        if lang is None:
            return None
        if lang.lower() not in LANGS:
            raise PlaceholderError(f"unknown language '{lang}' (use one of {', '.join(LANGS)})")
        return lang.lower()

    def _stat(self, key: str, metric: str, lang: str | None):
        run = self.run(key)
        stat = run.stat(metric, self._lang(lang))
        if stat is None:
            where = f" for language {lang}" if lang else ""
            raise PlaceholderError(f"run '{key}' has no metric '{metric}'{where}")
        return stat

    def _diff(self, a: str, b: str, metric: str) -> Diff:
        self.run(a), self.run(b)
        if (a, b, metric) not in self._cmp:
            self._cmp[(a, b, metric)] = compare(self.report.runs[a], self.report.runs[b],
                                                (metric,))
        comp = self._cmp[(a, b, metric)]
        if not comp.ready:
            raise PlaceholderError(f"cannot pair {a} with {b}: {comp.error}")
        diff = comp.diffs.get(metric)
        if diff is None:
            raise PlaceholderError(f"no paired values of '{metric}' for {a} and {b}")
        return diff

    def facts(self) -> dict:
        if self._facts is None:
            if self.facts_path is None or not self.facts_path.exists():
                raise PlaceholderError(f"facts file not found ({self.facts_path})")
            self._facts = json.loads(self.facts_path.read_text(encoding="utf-8")).get("facts", {})
        return self._facts

    def figures(self) -> dict[str, Figure]:
        if self._figures is None and self._figures_error is None:
            try:
                self._figures = make_figures(self.report.runs)
            except FigureError as exc:
                self._figures_error = str(exc)
        if self._figures is None:
            raise PlaceholderError(f"figures cannot be built: {self._figures_error}")
        return self._figures

    # ---- placeholders

    def resolve(self, kind: str, args: tuple[str, ...]) -> str:
        return getattr(self, f"_k_{kind}")(*args)

    def _k_v(self, key, metric, lang=None) -> str:
        self._stat(key, metric, lang)
        return self.fmt.v(key, metric, self._lang(lang), digits_for(metric))

    def _k_m(self, key, metric, lang=None) -> str:
        stat = self._stat(key, metric, lang)
        return f"{stat.mean:.{digits_for(metric)}f}"

    def _k_d(self, a, b, metric) -> str:
        return fmt_diff(self._diff(a, b, metric))

    def _k_claim(self, a, b, metric) -> str:
        return "claimable" if self._diff(a, b, metric).claimable else "not claimable"

    def _k_cost(self, key) -> str:
        return self._k_m(key, "cost_usd")

    def _k_lat(self, key) -> str:
        if self.run(key).scalar("latency_p50") is None:
            raise PlaceholderError(f"run '{key}' has no latency_p50")
        return self.fmt.lat50(key)

    def _k_pct_cheaper(self, a, b) -> str:
        self.run(a), self.run(b)
        ratio = self.fmt.cost_ratio(a, b)
        if ratio is None:
            raise PlaceholderError(f"cost of {a} or {b} is missing or zero")
        if ratio >= 1:
            raise PlaceholderError(f"{a} is not cheaper than {b} (cost ratio {ratio:.2f})")
        return f"{(1 - ratio) * 100:.0f}%"

    def _k_x_faster(self, a, b) -> str:
        self.run(a), self.run(b)
        ratio = self.fmt.ratio(b, a, "latency_p50")
        if ratio is None:
            raise PlaceholderError(f"median latency of {a} or {b} is missing or zero")
        if ratio < 1:
            raise PlaceholderError(f"{a} is not faster than {b} (latency ratio {ratio:.2f})")
        return f"{ratio:.1f} times"

    def _k_traps(self, key) -> str:
        traps = self.run(key).traps
        if traps is None:
            raise PlaceholderError(f"run '{key}' has no trap items")
        return f"{traps[0]} of {traps[1]}"

    def _k_n(self, key) -> str:
        run = self.run(key)
        n = run.metrics.get("n_answerable")
        if n is None:
            stat = run.stat("recall@10")
            n = stat.n if stat else None
        if n is None:
            raise PlaceholderError(f"run '{key}' has no number of answerable items")
        return str(n)

    def _k_f(self, key) -> str:
        entry = self.facts().get(key)
        if not isinstance(entry, dict) or "value" not in entry:
            raise PlaceholderError(f"no fact '{key}' in {self.facts_path}")
        return str(entry["value"])

    def _k_table(self, name) -> str:
        return ""

    def _k_figure(self, name) -> str:
        return ""

    def block(self, item: Item) -> None:
        if item.kind == "table":
            try:
                item.table = make_table(item.args[0], self.report.runs)
            except TableError as exc:
                raise PlaceholderError(str(exc)) from exc
        else:
            if item.args[0] not in FIGURE_NAMES:
                raise PlaceholderError(
                    f"unknown figure '{item.args[0]}' (known: {', '.join(FIGURE_NAMES)})")
            item.figure = self.figures()[item.args[0]]


def tokenize(template: str, resolver: Resolver, source: str) -> Tokenized:
    """Replace every placeholder outside code by a token and fill it; raise with all problems."""
    errors: list[str] = []
    out = Tokenized(lines=[])
    in_fence: str | None = None

    def fail(line: int, raw: str, message: str) -> None:
        errors.append(f"{source}:{line}: {raw}: {message}")

    def replace(segment: str, line: int, whole_line: bool) -> str:
        def sub(match: re.Match) -> str:
            raw = match.group(0)
            words = match.group(1).split()
            if not words:
                fail(line, raw, "empty placeholder")
                return raw
            kind, args = words[0], tuple(words[1:])
            if kind not in ARITY:
                fail(line, raw, f"unknown placeholder kind '{kind}' (known: "
                     f"{', '.join(sorted(ARITY))})")
                return raw
            low, high = ARITY[kind]
            if not low <= len(args) <= high:
                fail(line, raw, f"'{kind}' takes {low if low == high else f'{low} or {high}'} "
                     f"argument(s), got {len(args)}")
                return raw
            item = Item(kind, args, line, raw)
            try:
                if kind in BLOCK_KINDS:
                    if not (whole_line and segment.strip() == raw):
                        raise PlaceholderError(f"'{kind}' must stand alone on its line")
                    resolver.block(item)
                else:
                    item.value = resolver.resolve(kind, args)
            except PlaceholderError as exc:
                fail(line, raw, str(exc))
                return raw
            out.items.append(item)
            return f"{OPEN}{len(out.items) - 1}{CLOSE}"

        return PLACEHOLDER.sub(sub, segment)

    for number, line in enumerate(template.splitlines(), 1):
        fence = FENCE.match(line)
        if in_fence is not None:
            if fence and fence.group(1)[0] == in_fence[0] and len(fence.group(1)) >= len(in_fence):
                in_fence = None
            out.lines.append(line)
            continue
        if fence:
            in_fence = fence.group(1)
            out.lines.append(line)
            continue
        has_code = "`" in line
        if not has_code:
            new = replace(line, number, True)
        else:
            pieces, pos = [], 0
            for span in CODE_SPAN.finditer(line):
                pieces.append(replace(line[pos:span.start()], number, False))
                pieces.append(span.group(0))
                pos = span.end()
            pieces.append(replace(line[pos:], number, False))
            new = "".join(pieces)
        outside = PLACEHOLDER.sub("", CODE_SPAN.sub("", new))
        if "{{" in outside or "}}" in outside:
            fail(number, line.strip(), "unterminated or malformed placeholder")
        out.lines.append(new)
    if in_fence is not None:
        errors.append(f"{source}: code fence opened and never closed")
    if errors:
        raise BuildError(errors)
    return out
