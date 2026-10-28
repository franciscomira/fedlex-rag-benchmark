"""Training pairs: split the synthetic questions by article and fix a passage language per pair.

- **Split by article** (never by question): about ``VAL_FRACTION`` of the training articles, taken
  per act so that the Code of Obligations does not fill the validation set, go to ``val`` with all
  their questions. Articles are ordered inside each act by a hash of ``SEED`` and the id, so the
  split is deterministic and does not depend on file order.
- **One passage language per pair** (DE, FR or IT), by hash of the pair id. The same article in
  another language is the same positive, never a negative, and a query is never expanded into
  three rows (the other two rows would be false negatives of each other inside a batch).

Output: ``data/finetune/pairs.jsonl``, one row per question, sorted by article and slot. Run:
``uv run python -m fedlex_bench.finetune.dataset``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from fedlex_bench.benchmark.common import id_key
from fedlex_bench.finetune.exclusions import FINETUNE_DIR
from fedlex_bench.finetune.synth import LANGS

SEED = 20261028
VAL_FRACTION = 0.10
QUERIES_FILE = FINETUNE_DIR / "queries.jsonl"
PAIRS_FILE = FINETUNE_DIR / "pairs.jsonl"


def _h(salt: str, key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{SEED}:{salt}:{key}".encode()).digest()[:8], "big")


def split_articles(article_ids: list[str], fraction: float = VAL_FRACTION) -> set[str]:
    """Validation article ids: ``ceil(fraction * n)`` per act, chosen by hash order."""
    by_act: dict[str, list[str]] = defaultdict(list)
    for a in article_ids:
        by_act[a.partition(":")[0]].append(a)
    val: set[str] = set()
    for members in by_act.values():
        ordered = sorted(members, key=lambda a: _h("split", a))
        val |= set(ordered[: math.ceil(fraction * len(members))])
    return val


def pair_id(article_id: str, slot: int) -> str:
    return f"{article_id}#{slot}"


def passage_lang(pid: str) -> str:
    return LANGS[_h("passage", pid) % len(LANGS)]


def build_pairs(queries: list[dict]) -> list[dict]:
    val = split_articles(sorted({q["article_id"] for q in queries}))
    rows = []
    for q in sorted(queries, key=lambda q: (id_key(q["article_id"]), q["slot"])):
        pid = pair_id(q["article_id"], q["slot"])
        rows.append(
            {
                "pair_id": pid,
                "article_id": q["article_id"],
                "split": "val" if q["article_id"] in val else "train",
                "query": q["text"],
                "query_lang": q["query_lang"],
                "style": q["style"],
                "passage_lang": passage_lang(pid),
            }
        )
    return rows


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    rows = build_pairs(read_jsonl(QUERIES_FILE))
    PAIRS_FILE.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    for split in ("train", "val"):
        part = [r for r in rows if r["split"] == split]
        print(
            f"{split}: {len(part)} pairs, {len({r['article_id'] for r in part})} articles, "
            f"query langs {dict(sorted(_count(r['query_lang'] for r in part).items()))}, "
            f"passage langs {dict(sorted(_count(r['passage_lang'] for r in part).items()))}"
        )


def _count(items) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for x in items:
        out[x] += 1
    return out


if __name__ == "__main__":
    main()
