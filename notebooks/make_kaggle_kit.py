"""Pack what the Kaggle notebook needs into one zip (about 4 MB): code, training data, corpus.

    uv run python notebooks/make_kaggle_kit.py

Writes ``notebooks/kit/fedlex_finetune_kit.zip`` (git-ignored: it holds the corpus parquet) with the
same folder layout as the repository, so ``fedlex_bench`` finds its data without changes, plus
``KIT_INFO.json`` (git sha, SHA-256 of every file). Upload the zip as a **private** Kaggle dataset.
Deterministic: sorted entries, fixed timestamps.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "kit" / "fedlex_finetune_kit.zip"
FILES = [
    "data/finetune/pairs.jsonl",
    "data/finetune/hardneg.jsonl",
    "data/finetune/train_pool_ids.txt",
    "data/processed/articles.parquet",
    "benchmark/v1/dev_gold_articles.txt",
    "benchmark/v1/test_gold_articles.txt",
    "tests/test_finetune_leakage.py",
    "tests/test_finetune_train.py",
]


def _info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, (2026, 10, 28, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def entries() -> list[str]:
    code = [p.relative_to(ROOT).as_posix() for p in (ROOT / "src").rglob("*.py")]
    code += [p.relative_to(ROOT).as_posix() for p in (ROOT / "src").rglob("*.toml")]
    return sorted({*code, *FILES})


def main() -> None:
    names = entries()
    missing = [n for n in names if not (ROOT / n).exists()]
    if missing:
        raise SystemExit(f"missing: {missing}")
    sha = {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in names}
    try:
        git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git = "unknown"
    info = json.dumps({"git_sha": git, "sha256": sha}, indent=1, sort_keys=True) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.writestr(_info(n), (ROOT / n).read_bytes())
        z.writestr(_info("KIT_INFO.json"), info)
    size = OUT.stat().st_size / 1e6
    print(f"{OUT.relative_to(ROOT)}: {len(names)} files, {size:.1f} MB, git {git[:8]}")


if __name__ == "__main__":
    main()
