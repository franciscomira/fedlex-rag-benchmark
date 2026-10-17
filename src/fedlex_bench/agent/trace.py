"""Trace recording for one agent run: a list of step dicts plus an optional JSONL sink.

Step schema (all keys always present, None when not applicable):
    step, round, t_s (seconds since run start), agent, kind, tool, args, result_ids,
    usage (dict of Usage fields), latency_s (wall clock of this step), error, plus
    kind-specific extras (e.g. refs_seen, new_ids, status, citations).

kind is one of: llm, tool, answer, check, round_end, limit, final.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fedlex_bench.types import Usage

ZERO_USAGE = asdict(Usage())


def usage_dict(usage: Usage | None) -> dict[str, Any]:
    return asdict(usage) if usage is not None else dict(ZERO_USAGE)


class TraceRecorder:
    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []
        self.round = 0
        self._t0 = time.perf_counter()

    def add(
        self,
        agent: str,
        kind: str,
        *,
        tool: str | None = None,
        args: dict[str, Any] | None = None,
        result_ids: list[str] | None = None,
        usage: Usage | None = None,
        latency_s: float | None = None,
        error: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        step = {
            "step": len(self.steps),
            "round": self.round,
            "t_s": round(time.perf_counter() - self._t0, 4),
            "agent": agent,
            "kind": kind,
            "tool": tool,
            "args": args or {},
            "result_ids": list(result_ids) if result_ids is not None else None,
            "usage": usage_dict(usage),
            "latency_s": round(latency_s, 4) if latency_s is not None else None,
            "error": error,
        }
        step.update(extra)
        self.steps.append(step)
        return step


class TraceWriter:
    """Appends one JSON line per question to `<dir>/traces.jsonl` (thread safe)."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")
