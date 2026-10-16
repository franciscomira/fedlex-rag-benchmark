"""Constants and helpers shared by the benchmark commands (docs/benchmark_spec.md).

Everything here is deterministic: no clock, no unseeded randomness, sorted inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from fractions import Fraction
from pathlib import Path

import pandas as pd

from fedlex_bench.chunk import ARTICLES_PARQUET, natural_key
from fedlex_bench.ingest import PROJECT_ROOT

SEED = 20261015
BUCKET_PREFIX = "fedlex-bench-v1:"
DEV_BUCKETS = 27  # bucket < 27 -> dev, else test

QUERY_LANGS: tuple[str, ...] = ("en", "de", "fr", "it")
CORPUS_LANGS: tuple[str, ...] = ("de", "fr", "it")
LANG_NAMES = {"en": "English", "de": "German", "fr": "French", "it": "Italian"}
SPLITS: tuple[str, ...] = ("dev", "test")

# Section 1: final answerable quotas per act (150 in total), in natural act order.
ACT_QUOTA: dict[str, int] = {
    "210": 14,
    "220": 90,
    "221.214.1": 5,
    "221.229.1": 8,
    "235.1": 8,
    "241": 5,
    "272": 10,
    "281.1": 10,
}
ACT_NAMES = {
    "210": "Civil Code",
    "220": "Code of Obligations",
    "221.214.1": "Consumer Credit Act",
    "221.229.1": "Insurance Contract Act",
    "235.1": "Data Protection Act",
    "241": "Unfair Competition Act",
    "272": "Civil Procedure Code",
    "281.1": "Debt Enforcement and Bankruptcy Act",
}
DEV_ANSWERABLE = 40
TEST_ANSWERABLE = 110
CROSS_REF_FINAL = 22
TRAP_FINAL = {"dev": 5, "test": 15}
TRAP_DRAFT_TOTAL = 30
# Section 2: final language balance of the answerable items.
LANG_FINAL = {"en": 38, "de": 37, "fr": 38, "it": 37}
TRAP_LANG_FINAL = {"en": 5, "de": 5, "fr": 5, "it": 5}
TRAP_TYPES: tuple[str, ...] = ("out_of_area", "repealed", "unspecified_detail", "non_statute")
QTYPES: tuple[str, ...] = ("single", "cross_ref", "trap")
DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")

BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
PLAN_DIR = BENCHMARK_DIR / "plan"
SLOTS_PATH = PLAN_DIR / "slots.jsonl"
TWINS_PATH = PLAN_DIR / "twins.jsonl"
INDEX_PATH = PLAN_DIR / "index.json"
INDEX_PILOT_PATH = PLAN_DIR / "index-pilot.json"
BRIEFS_DIR = PLAN_DIR / "briefs"
DRAFTS_DIR = BENCHMARK_DIR / "drafts"
CRITIC_DIR = BENCHMARK_DIR / "critic"
ASSEMBLED_DIR = BENCHMARK_DIR / "assembled"
CANDIDATES_PATH = ASSEMBLED_DIR / "candidates.jsonl"
SELECTED_DIR = BENCHMARK_DIR / "selected"
RESERVE_DIR = BENCHMARK_DIR / "reserve"
V1_DIR = BENCHMARK_DIR / "v1"


# ------------------------------------------------------------------ split rule (section 4)


def bucket(article_id: str) -> int:
    """``int(sha256("fedlex-bench-v1:" + article_id).hexdigest(), 16) % 100``."""
    digest = hashlib.sha256((BUCKET_PREFIX + article_id).encode("utf-8")).hexdigest()
    return int(digest, 16) % 100


def split_of(article_id: str) -> str:
    """``"dev"`` if ``bucket(article_id) < 27`` else ``"test"``."""
    return "dev" if bucket(article_id) < DEV_BUCKETS else "test"


# ------------------------------------------------------------------ ordering and arithmetic


def id_key(article_id: str) -> tuple:
    """Natural order of ``"<SR>:<number>"`` ids (act first, then article number)."""
    sr, _, number = article_id.partition(":")
    return (natural_key(sr), natural_key(number), article_id)


def act_key(act: str | None) -> tuple:
    """Natural order of SR numbers; ``None`` (traps) sorts after every act."""
    return (1, ()) if act is None else (0, natural_key(act))


def draft_count(final: int, factor: Fraction = Fraction(3, 2)) -> int:
    """Draft size for a final quota: ``ceil(1.5 * final)``, in exact arithmetic."""
    return math.ceil(final * factor)


def largest_remainder(total: int, weights: Mapping, order: Iterable | None = None) -> dict:
    """Split the integer ``total`` over ``weights`` proportionally (Hare quota, largest
    remainder). Ties go to the key that comes first in ``order`` (default: insertion order of
    ``weights``). Exact (``Fraction``) arithmetic, so the result never depends on float rounding."""
    keys = list(order) if order is not None else list(weights)
    weight_sum = sum(weights[k] for k in keys)
    if weight_sum <= 0:
        raise ValueError("weights must have a positive sum")
    shares = {k: Fraction(total * weights[k], weight_sum) for k in keys}
    out = {k: int(shares[k]) for k in keys}  # floor, shares are non-negative
    leftover = total - sum(out.values())
    ranked = sorted(keys, key=lambda k: (-(shares[k] - out[k]), keys.index(k)))
    for k in ranked[:leftover]:
        out[k] += 1
    return out


# ------------------------------------------------------------------ files


def dumps(obj: object) -> str:
    """One JSON line: sorted keys, UTF-8 (no ASCII escaping), compact separators not used so the
    files stay readable."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def write_jsonl(path: Path, records: Iterable[Mapping]) -> str:
    """Write records as JSONL (sorted keys, UTF-8, ``\\n`` endings); returns the file's sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(dumps(r) + "\n" for r in records).encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def numbered_lines(text: str) -> list[tuple[int, str]]:
    """``(line number, line)`` of the non-blank lines. Lines end at a line feed only: U+2028,
    U+0085 and form feed are legal inside a JSON string and are not line breaks here."""
    return [(n, line) for n, line in enumerate(text.split("\n"), 1) if line.strip()]


def split_lines(text: str) -> list[str]:
    return [line for _, line in numbered_lines(text)]


def read_jsonl_text(text: str, label: str) -> list[dict]:
    """Parse JSONL text. Raises ``ValueError`` naming ``label:line`` on bad JSON."""
    out: list[dict] = []
    for n, line in numbered_lines(text):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label}:{n}: invalid JSON ({exc.msg})") from exc
    return out


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file (split at line feeds only). Raises ``ValueError`` naming the line."""
    return read_jsonl_text(path.read_bytes().decode("utf-8"), str(path))


def canonical_sha256(obj: object) -> str:
    """sha256 of the canonical JSON (sorted keys, UTF-8) of ``obj``."""
    return hashlib.sha256(dumps(obj).encode("utf-8")).hexdigest()


def slot_hash(slot: dict) -> str:
    """Hash of a plan slot, gold ids included. The ``pilot`` mark is not part of what a writer
    drafts from, so it does not count."""
    return canonical_sha256({k: v for k, v in slot.items() if k != "pilot"})


def slots_sha256(slots: list[dict]) -> str:
    """sha256 the slots file has when written by :func:`write_jsonl`."""
    data = "".join(dumps(s) + "\n" for s in slots).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_corpus(parquet: Path = ARTICLES_PARQUET) -> pd.DataFrame:
    if not parquet.exists():
        raise FileNotFoundError(
            f"{parquet} is missing: run `uv run python -m fedlex_bench.chunk` first"
        )
    return pd.read_parquet(parquet)
