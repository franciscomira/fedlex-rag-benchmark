"""Markdown results table from eval/results/<run_id>/ (one row per run, 95% CIs)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

COLUMNS = (
    ("recall@1", "R@1"),
    ("recall@5", "R@5"),
    ("recall@10", "R@10"),
    ("all_gold@10", "All gold@10"),
    ("mrr", "MRR"),
    ("fact_recall", "Fact recall"),
    ("faithfulness", "Faithful"),
    ("cit_precision", "Cit P"),
    ("cit_recall", "Cit R"),
)


def _cell(stat: dict[str, Any] | None, digits: int = 2) -> str:
    if not stat or not stat.get("n"):
        return "n/a"
    return f"{stat['mean']:.{digits}f} [{stat['lo']:.{digits}f}, {stat['hi']:.{digits}f}]"


def _label(config: dict[str, Any]) -> str:
    s = config.get("system", {})
    parts = [s.get("retriever") or s.get("kind", "?")]
    parts += [x for x in (s.get("reranker"), s.get("answerer")) if x]
    return " + ".join(parts)


def render_table(run_dirs: list[Path]) -> str:
    header = ["System", "n", *[c[1] for c in COLUMNS], "Over-refusal", "Parse fail", "Judge fail", "Cost/q USD", "p50 s", "p95 s", "Traps"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for d in run_dirs:
        metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        config = json.loads((d / "config.json").read_text(encoding="utf-8"))
        o = metrics["overall"]
        traps = metrics["traps"]
        row = [
            _label(config),
            str(metrics.get("n_answerable", metrics["n_items"])),
            *[_cell(o.get(key)) for key, _ in COLUMNS],
            _cell(o.get("over_refusal")),
            _cell(o.get("parse_failure_rate")),
            str(sum(metrics["judge_failures"].values()) if isinstance(metrics.get("judge_failures"), dict) else metrics.get("judge_failures", 0)),
            _cell(o.get("cost_usd"), 4),
            f"{o.get('latency_p50', float('nan')):.2f}",
            f"{o.get('latency_p95', float('nan')):.2f}",
            f"{traps['correct']}/{traps['n']}" if traps["n"] else "n/a",
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
