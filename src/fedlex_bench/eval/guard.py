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
        ["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True
    )
    if status.stdout.strip():
        raise TestRunRefused("test runs need a clean git tree; commit first")
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return head.stdout.strip()


def log_test_run(log: Path, *, run_id: str, config_hash: str, date: str) -> None:
    """Append a test run; refuse a second run of the same frozen configuration."""
    log.parent.mkdir(parents=True, exist_ok=True)
    existing = [json.loads(x) for x in log.read_text("utf-8").splitlines()] if log.exists() else []
    if any(e["config_hash"] == config_hash for e in existing):
        raise TestRunRefused(f"configuration {config_hash} already ran on test")
    entry = {"date": date, "run_id": run_id, "config_hash": config_hash}
    with log.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
