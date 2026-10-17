"""Test-split guards (docs/eval_spec.md sections 0, 8, 11) and the corpus text lookup."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import pandas as pd

MAX_CHARS = 6000


class TestRunRefused(RuntimeError):
    __test__ = False  # not a pytest class


def article_text_lookup(parquet: Path) -> Callable[[Iterable[str]], Mapping[str, str]]:
    """id -> labelled text of all language versions (for the faithfulness judge)."""
    df = pd.read_parquet(parquet, columns=["article_id", "lang", "text"])
    by_id: dict[str, list[tuple[str, str]]] = {}
    for article_id, lang, text in df.sort_values(["article_id", "lang"]).itertuples(index=False):
        by_id.setdefault(article_id, []).append((lang, text))

    def lookup(ids: Iterable[str]) -> dict[str, str]:
        return {
            i: "\n\n".join(f"({lang}) {text[:MAX_CHARS]}" for lang, text in by_id.get(i, []))
            for i in ids
        }

    return lookup


def require_clean_tree(repo: Path) -> str:
    """Return HEAD sha; refuse if the working tree has uncommitted changes."""
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--", ".", ":!eval/results", ":!eval/test_runs.log"],
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise TestRunRefused("test runs need a clean git tree; commit first")
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return head.stdout.strip()


def _entries(log: Path) -> list[dict]:
    return [json.loads(x) for x in log.read_text("utf-8").splitlines()] if log.exists() else []


def _append(log: Path, entry: dict) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")


def log_test_start(log: Path, *, run_id: str, config_hash: str, date: str) -> None:
    """Record that a test run started; refuse if this configuration already completed one."""
    if any(e["config_hash"] == config_hash and e["status"] == "completed" for e in _entries(log)):
        raise TestRunRefused(f"configuration {config_hash} already ran on test")
    _append(log, {"date": date, "run_id": run_id, "config_hash": config_hash, "status": "started"})


def log_test_done(log: Path, *, run_id: str, config_hash: str, date: str) -> None:
    _append(log, {"date": date, "run_id": run_id, "config_hash": config_hash, "status": "completed"})


def require_frozen(frozen_file: Path, config_hash: str) -> str:
    """The name of the frozen configuration with this hash; refuse any other test run."""
    frozen = json.loads(frozen_file.read_text("utf-8")) if frozen_file.exists() else {}
    for name, entry in sorted(frozen.items()):
        if entry["hash"] == config_hash:
            return name
    raise TestRunRefused(f"configuration {config_hash} is not in {frozen_file.name}; test runs are frozen")
