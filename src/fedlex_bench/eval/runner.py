"""Harness: run a System over benchmark items, score, aggregate, write the run layout."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from fedlex_bench.eval.judge import Judge, prompt_hashes
from fedlex_bench.eval.metrics import (
    citation_precision_recall,
    dedupe_ids,
    recall_at_k,
    reciprocal_rank,
    trap_code_checks,
)
from fedlex_bench.eval.stats import bootstrap_ci
from fedlex_bench.types import System, SystemOutput, Usage

TextLookup = Callable[[Iterable[str]], Mapping[str, str]]
RETRIEVAL_METRICS = ("recall@1", "recall@5", "recall@10", "mrr")
ANSWER_METRICS = ("fact_recall", "cit_precision", "cit_recall", "faithfulness")
COST_METRICS = ("cost_usd", "latency_s")
KEPT_RANKS = 20
_DUMP = {"sort_keys": True, "ensure_ascii": False}


def group_id(item: Mapping) -> str:
    return item.get("translation_of") or item["id"]


def load_items(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return sorted((json.loads(x) for x in lines if x.strip()), key=lambda i: i["id"])


def score_item(item: Mapping, out: SystemOutput, judge: Judge, texts: TextLookup) -> dict[str, Any]:
    gold = list(item["gold_article_ids"])
    ranked = dedupe_ids(out.ranked_article_ids)
    ans = out.answer
    row: dict[str, Any] = {
        "id": item["id"],
        "group": group_id(item),
        "query_lang": item["query_lang"],
        "qtype": item["qtype"],
        "is_trap": item["is_trap"],
        "is_translation": bool(item.get("translation_of")),
        "gold": gold,
        "ranked": ranked[:KEPT_RANKS],
        "cost_usd": out.usage.cost_usd,
        "latency_s": out.usage.latency_s,
        "input_tokens": out.usage.input_tokens,
        "output_tokens": out.usage.output_tokens,
        "calls": out.usage.calls,
        "trace_steps": len(out.trace),
        **{m: None for m in RETRIEVAL_METRICS + ANSWER_METRICS},
        "over_refusal": None,
        "trap_correct": None,
        "judge_failed": False,
    }
    judge_usage = Usage()
    if not item["is_trap"]:
        row["recall@1"] = recall_at_k(ranked, gold, 1)
        row["recall@5"] = recall_at_k(ranked, gold, 5)
        row["recall@10"] = recall_at_k(ranked, gold, 10)
        row["mrr"] = reciprocal_rank(ranked, gold)
    if ans is not None:
        row.update(
            status=ans.status,
            answer=ans.answer,
            citations=ans.citations,
            pointers=ans.pointers,
            invalid_citations=ans.invalid_citations,
            parse_failed=ans.parse_failed,
        )
        if item["is_trap"]:
            ok = trap_code_checks(ans)
            if ok:
                leak = judge.trap_leak(item["question"], ans.answer, item.get("trap_reason") or "")
                judge_usage += leak.usage
                row["judge_failed"] = leak.failed
                ok = leak.leaked is False
            row["trap_correct"] = ok
        else:
            row["over_refusal"] = ans.status == "refused"
            precision, recall = citation_precision_recall(ans.citations, gold)
            row["cit_precision"], row["cit_recall"] = precision, recall
            if ans.status == "refused" or ans.parse_failed:
                row["fact_recall"] = 0.0
            else:
                fr = judge.fact_recall(item["question"], ans.answer, item["gold_facts"])
                judge_usage += fr.usage
                row["fact_recall"] = fr.score
                row["judge_failed"] = fr.failed
                if ans.citations:
                    fa = judge.faithfulness(ans.answer, texts(ans.citations))
                    judge_usage += fa.usage
                    row["faithfulness"] = fa.score
                    row["judge_failed"] = row["judge_failed"] or fa.failed
    row["judge_cost_usd"] = judge_usage.cost_usd
    return row


def _stat(rows: Sequence[Mapping], key: str) -> dict[str, Any]:
    kept = [r for r in rows if r.get(key) is not None]
    if not kept:
        return {"n": 0}
    vals = [float(r[key]) for r in kept]
    return {"n": len(kept), **bootstrap_ci(vals, [r["group"] for r in kept])}


def _block(rows: Sequence[Mapping]) -> dict[str, Any]:
    block = {k: _stat(rows, k) for k in RETRIEVAL_METRICS + ANSWER_METRICS + COST_METRICS}
    block["over_refusal"] = _stat(
        [
            {**r, "over_refusal": None if r["over_refusal"] is None else float(r["over_refusal"])}
            for r in rows
        ],
        "over_refusal",
    )
    lat = sorted(r["latency_s"] for r in rows)
    if lat:
        block["latency_p50"] = lat[len(lat) // 2]
        block["latency_p95"] = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
    return block


def aggregate(rows: Sequence[Mapping]) -> dict[str, Any]:
    answerable = [r for r in rows if not r["is_trap"]]
    traps = [r for r in rows if r["is_trap"] and r["trap_correct"] is not None]
    originals = [r for r in answerable if not r["is_translation"]]
    langs = sorted({r["query_lang"] for r in answerable})
    types = sorted({r["qtype"] for r in answerable})
    return {
        "n_items": len(rows),
        "overall": _block(answerable),
        "originals_only": _block(originals),
        "by_query_lang": {k: _block([r for r in answerable if r["query_lang"] == k]) for k in langs},
        "by_qtype": {k: _block([r for r in answerable if r["qtype"] == k]) for k in types},
        "traps": {"correct": sum(bool(r["trap_correct"]) for r in traps), "n": len(traps)},
        "judge_failures": sum(bool(r["judge_failed"]) for r in rows),
        "judge_cost_usd": sum(r["judge_cost_usd"] for r in rows),
    }


def run_eval(
    system: System,
    items: Sequence[Mapping],
    judge: Judge,
    results_dir: Path,
    *,
    run_id: str,
    split: str,
    texts: TextLookup,
    git_sha: str,
    extra_config: Mapping | None = None,
) -> Path:
    out_dir = Path(results_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in sorted(items, key=lambda i: i["id"]):
        output = system.run(item["question"], item["query_lang"])
        rows.append(score_item(item, output, judge, texts))
    config = {
        "run_id": run_id,
        "split": split,
        "git_sha": git_sha,
        "system": system.config(),
        "judge_model": getattr(judge.client, "model_id", "?"),
        "prompt_hashes": prompt_hashes(),
        "n_items": len(rows),
        **(extra_config or {}),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, **_DUMP) + "\n", "utf-8")
    (out_dir / "per_item.jsonl").write_text(
        "".join(json.dumps(r, **_DUMP) + "\n" for r in rows), "utf-8"
    )
    metrics = aggregate(rows)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, **_DUMP) + "\n", "utf-8")
    return out_dir
