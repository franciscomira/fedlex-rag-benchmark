"""Assemble everything the page shows from the manifest and the committed harness outputs."""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, field
from math import comb
from pathlib import Path

from fedlex_bench.site.data import (
    Comparison,
    Run,
    compare,
    group_count,
    judge_agreement,
    load_run,
    price_date,
    read_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = Path("site") / "results_manifest.json"
EVAL_PREFIX = "uv run python -m fedlex_bench.eval"


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial interval for k successes in n trials (used for trap counts)."""

    def cdf(x: int, p: float) -> float:  # P(X <= x)
        return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(x + 1))

    lower = 0.0 if k == 0 else _bisect_root(lambda p: 1 - cdf(k - 1, p), alpha / 2)
    upper = 1.0 if k == n else _bisect_root(lambda p: cdf(k, p), alpha / 2, decreasing=True)
    return lower, upper


def _bisect_root(f, target: float, decreasing: bool = False) -> float:
    """Root of f(p) = target on [0, 1]; f is increasing unless ``decreasing``."""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if (f(mid) > target) == decreasing:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@dataclass
class Report:
    manifest: dict
    root: Path
    runs: dict[str, Run]
    facts: dict
    warnings: list[str] = field(default_factory=list)
    _cmp: dict[tuple[str, str], Comparison] = field(default_factory=dict)

    def run(self, key: str) -> Run:
        return self.runs[key]

    def section_runs(self, section: str) -> list[Run]:
        return [self.runs[k] for k in self.manifest["sections"][section].get("runs", [])]

    def section_pairs(self, section: str) -> list[tuple[str, str]]:
        return [tuple(p) for p in self.manifest["sections"][section].get("comparisons", [])]

    def comparison(self, a: str, b: str) -> Comparison:
        if (a, b) not in self._cmp:
            self._cmp[(a, b)] = compare(self.runs[a], self.runs[b])
        return self._cmp[(a, b)]

    def section_comparisons(self, section: str) -> list[Comparison]:
        return [self.comparison(a, b) for a, b in self.section_pairs(section)]


def load_manifest(path: Path) -> dict:
    manifest = read_json(path)
    for key in ("links", "runs", "sections", "reproduce"):
        if key not in manifest:
            raise SystemExit(f"{path}: missing '{key}'")
    for name, section in manifest["sections"].items():
        pairs = [k for pair in section.get("comparisons", []) for k in pair]
        for key in section.get("runs", []) + pairs:
            if key not in manifest["runs"]:
                raise SystemExit(f"{path}: section '{name}' names unknown run '{key}'")
    return manifest


def build_report(root: Path = PROJECT_ROOT, manifest_path: Path | None = None) -> Report:
    path = manifest_path or root / DEFAULT_MANIFEST
    manifest = load_manifest(path)
    warnings: list[str] = []
    runs = {k: load_run(root, k, spec, warnings) for k, spec in sorted(manifest["runs"].items())}
    config_path = root / "config.toml"
    budget = tomllib.loads(config_path.read_text("utf-8")).get("budget", {}) if (
        config_path.exists()
    ) else {}
    facts: dict = {
        "cap_usd": _usd(budget.get("monthly_usd")),
        "confirm_usd": _usd(budget.get("confirm_usd")),
        "price_date": price_date(root),
    }
    ref = next((r for k in ("T1", "A3") if (r := runs.get(k)) and r.ok), None)
    if ref is not None:
        stat = ref.stat("recall@10")
        facts["n_answerable"] = stat.n if stat else None
        facts["n_groups"] = group_count(ref)
        langs = [ref.stat("recall@10", lang) for lang in ("en", "de", "fr", "it")]
        sizes = [s.n for s in langs if s]
        facts["lang_n"] = (min(sizes), max(sizes)) if sizes else None
        facts["n_items"] = ref.metrics.get("n_items")
    judged = next((r for r in runs.values() if r.ok and r.has_answers), None)
    if judged is not None:
        facts["judge"] = judged.judge_model
        facts["judge_agreement"] = judge_agreement(root, judged.judge_model)
        facts["n_traps"] = (judged.traps or (0, 0))[1]
    return Report(manifest, root, runs, facts, warnings)


def _usd(value) -> str:
    if value is None:
        return "unknown"
    return f"{value:g}"


# ---------------------------------------------------------------- reproduce table

_RUNS = re.compile(r"\{runs:([a-z0-9_]+)\}")
_COMPARES = re.compile(r"^\{compares:([a-z0-9_]+)\}$")


def expand_commands(report: Report, template: str) -> list[str]:
    """Expand one command template into zero or more concrete commands.

    ``{runs:section}`` becomes the ids of the section's runs that exist; ``{compares:section}`` as
    the whole template becomes one ``eval compare`` per computable pair. Templates whose runs do
    not exist yet expand to nothing.
    """
    m = _COMPARES.match(template)
    if m:
        out = []
        for a, b in report.section_pairs(m.group(1)):
            ra, rb = report.runs[a], report.runs[b]
            if report.comparison(a, b).ready:
                out.append(f"{EVAL_PREFIX} compare {ra.run_id} {rb.run_id}")
        return out

    sections = _RUNS.findall(template)
    if not sections:
        return [template]
    ids = {
        sec: [r.run_id for r in report.section_runs(sec) if r.ok and r.run_id] for sec in sections
    }
    if not all(ids.values()):
        return []
    return [_RUNS.sub(lambda m: " ".join(ids[m.group(1)]), template)]


def expand_text(report: Report, text: str) -> str:
    for key in ("cap_usd", "confirm_usd"):
        text = text.replace("{" + key + "}", str(report.facts.get(key, "unknown")))
    return text


def dump_data(report: Report) -> str:
    """Machine-readable copy of every number the page shows (site/results_data.json)."""
    out: dict = {"runs": {}, "comparisons": {}}
    for key, run in sorted(report.runs.items()):
        entry: dict = {"run_id": run.run_id, "state": run.state, "label": run.label}
        if run.ok:
            entry["config_hash"] = run.config_hash
            entry["frozen_name"] = run.frozen_name
            entry["overall"] = run.metrics["overall"]
            entry["by_query_lang"] = run.metrics["by_query_lang"]
            entry["traps"] = run.metrics["traps"]
        else:
            entry["reason"] = run.reason
        out["runs"][key] = entry
    seen: set[tuple[str, str]] = set()
    for name in sorted(report.manifest["sections"]):
        for a, b in report.section_pairs(name):
            if (a, b) in seen:
                continue
            seen.add((a, b))
            comp = report.comparison(a, b)
            out["comparisons"][f"{a} minus {b}"] = (
                {m: (None if d is None else {"mean": d.mean, "lo": d.lo, "hi": d.hi, "n": d.n})
                 for m, d in sorted(comp.diffs.items())}
                if comp.ready
                else {"error": comp.error}
            )
    return json.dumps(out, indent=1, sort_keys=True) + "\n"
