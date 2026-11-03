"""Read the committed harness outputs for the results page. Nothing here is typed by hand.

Sources, all under ``<root>/eval``: ``results/<run>/metrics.json``, ``config.json`` and
``per_item.jsonl``, ``test_runs.log``, ``frozen_configs.json``, ``prices.json`` and
``calibration/agreement_first_pass.json``. Paired differences use ``paired_bootstrap`` from
``fedlex_bench.eval.stats`` with the same item selection as ``eval compare``
(``fedlex_bench.eval.__main__.cmd_compare``), so a number on the page is the number that command
prints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fedlex_bench.eval.stats import paired_bootstrap

# `eval compare` defaults; the page asks for the same metrics unless a table needs fewer.
COMPARE_METRICS = ("recall@10", "mrr", "fact_recall", "faithfulness", "cost_usd")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


@dataclass(frozen=True)
class Stat:
    mean: float
    lo: float
    hi: float
    n: int


def _stat(raw: dict | None) -> Stat | None:
    if not raw or not raw.get("n"):
        return None
    return Stat(float(raw["mean"]), float(raw["lo"]), float(raw["hi"]), int(raw["n"]))


@dataclass
class Run:
    """One manifest run: either loaded from a run folder or pending."""

    key: str
    label: str
    name: str
    run_id: str | None
    note: str = ""
    state: str = "pending"  # "ok" or "pending"
    reason: str = ""  # why pending
    metrics: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    folder: Path | None = None
    frozen_name: str | None = None
    _items: list[dict] | None = None

    @property
    def ok(self) -> bool:
        return self.state == "ok"

    @property
    def config_hash(self) -> str:
        return str(self.config.get("config_hash", ""))

    @property
    def embedder(self) -> str:
        return str(self.config.get("system", {}).get("embedder", ""))

    def stat(self, metric: str, lang: str | None = None) -> Stat | None:
        if not self.ok:
            return None
        block = self.metrics["overall"] if lang is None else self.metrics["by_query_lang"].get(lang)
        return _stat((block or {}).get(metric))

    def original_stat(self, metric: str) -> Stat | None:
        return _stat(self.metrics.get("originals_only", {}).get(metric)) if self.ok else None

    def scalar(self, metric: str) -> float | None:
        if not self.ok:
            return None
        value = self.metrics["overall"].get(metric)
        return None if value is None else float(value)

    @property
    def traps(self) -> tuple[int, int] | None:
        """(handled correctly, traps) or None when the run has no trap items."""
        if not self.ok:
            return None
        t = self.metrics.get("traps") or {}
        return (int(t["correct"]), int(t["n"])) if t.get("n") else None

    @property
    def judge_model(self) -> str:
        return str(self.config.get("judge_model", ""))

    @property
    def has_answers(self) -> bool:
        return self.ok and self.config.get("system", {}).get("answerer") is not None

    def items(self) -> list[dict]:
        if self._items is None:
            assert self.folder is not None
            self._items = read_jsonl(self.folder / "per_item.jsonl")
        return self._items


def _completed_ids(log: Path) -> set[str]:
    if not log.exists():
        return set()
    return {
        row["run_id"] for row in read_jsonl(log) if row.get("status") == "completed"
    }


def load_run(root: Path, key: str, spec: dict, warnings: list[str] | None = None) -> Run:
    """Load a manifest run. A missing id or folder yields a pending run, never an error."""
    run = Run(
        key=key,
        label=spec.get("label", key),
        name=spec.get("name", ""),
        run_id=spec.get("id") or None,
        note=spec.get("note", ""),
    )
    if not run.run_id:
        run.reason = "no run yet"
        return run
    folder = root / "eval" / "results" / run.run_id
    if not (folder / "metrics.json").exists() or not (folder / "config.json").exists():
        run.reason = "run folder not found"
        if warnings is not None:
            warnings.append(f"{key}: run {run.run_id} not under eval/results, shown as pending")
        return run
    config = read_json(folder / "config.json")
    if config.get("split") == "test" and run.run_id not in _completed_ids(
        root / "eval" / "test_runs.log"
    ):
        run.reason = "not completed in eval/test_runs.log"
        if warnings is not None:
            warnings.append(f"{key}: run {run.run_id} has no completed line in test_runs.log")
        return run
    run.folder, run.config = folder, config
    run.metrics = read_json(folder / "metrics.json")
    frozen_path = root / "eval" / "frozen_configs.json"
    if frozen_path.exists():
        for name, entry in sorted(read_json(frozen_path).items()):
            if entry.get("hash") == run.config_hash:
                run.frozen_name = name
                break
    run.state = "ok"
    return run


@dataclass(frozen=True)
class Diff:
    """Paired difference of metric (first minus second) with its 95% interval."""

    mean: float
    lo: float
    hi: float
    n: int

    @property
    def claimable(self) -> bool:
        return self.lo > 0 or self.hi < 0


@dataclass
class Comparison:
    a: Run
    b: Run
    diffs: dict[str, Diff | None] = field(default_factory=dict)
    error: str = ""  # why it could not be computed, empty when computed

    @property
    def ready(self) -> bool:
        return not self.error


def compare(a: Run, b: Run, metrics: tuple[str, ...] = COMPARE_METRICS) -> Comparison:
    """Same selection and statistics as ``eval compare`` (A minus B)."""
    comp = Comparison(a, b)
    for run in (a, b):
        if not run.ok:
            comp.error = f"pending: {run.label} ({run.reason})"
            return comp
    for key in ("split", "judge_model"):
        if a.config.get(key) != b.config.get(key):
            comp.error = (
                f"not comparable: runs differ in {key} "
                f"({a.config.get(key)!r} vs {b.config.get(key)!r})"
            )
            return comp
    rows_a = {r["id"]: r for r in a.items()}
    rows_b = {r["id"]: r for r in b.items()}
    for metric in metrics:
        ids = [
            i
            for i in sorted(rows_a)
            if i in rows_b
            and rows_a[i].get(metric) is not None
            and rows_b[i].get(metric) is not None
        ]
        if not ids:
            comp.diffs[metric] = None
            continue
        d = paired_bootstrap(
            [float(rows_a[i][metric]) for i in ids],
            [float(rows_b[i][metric]) for i in ids],
            [rows_a[i]["group"] for i in ids],
        )
        comp.diffs[metric] = Diff(d["mean"], d["lo"], d["hi"], len(ids))
    return comp


# ---------------------------------------------------------------- formatting


def fmt_stat(stat: Stat | None, digits: int = 2) -> str:
    """Same layout as ``eval.report._cell``: mean [lo, hi]."""
    if stat is None:
        return "n/a"
    return f"{stat.mean:.{digits}f} [{stat.lo:.{digits}f}, {stat.hi:.{digits}f}]"


def fmt_diff(diff: Diff | Stat | None, digits: int = 3) -> str:
    if diff is None:
        return "n/a"
    return f"{diff.mean:+.{digits}f} [{diff.lo:+.{digits}f}, {diff.hi:+.{digits}f}]"


def fmt_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}" if value >= 1 else f"{value:.2f}"


def fmt_latency(run: Run) -> str:
    p50, p95 = run.scalar("latency_p50"), run.scalar("latency_p95")
    if p50 is None or p95 is None:
        return "n/a"
    return f"{fmt_seconds(p50)} / {fmt_seconds(p95)} s"


# ---------------------------------------------------------------- shared facts


def judge_agreement(root: Path, judge: str) -> float | None:
    """First-pass agreement of the judge with the human labels on facts (not the revised file)."""
    path = root / "eval" / "calibration" / "agreement_first_pass.json"
    if not path.exists():
        return None
    entry = read_json(path).get(judge)
    return float(entry["facts"]["raw"]) if entry else None


def price_date(root: Path) -> str:
    path = root / "eval" / "prices.json"
    return str(read_json(path).get("retrieved", "unknown")) if path.exists() else "unknown"


def group_count(run: Run) -> int:
    """Independent question groups among the answerable items."""
    return len({r["group"] for r in run.items() if not r.get("is_trap")})
