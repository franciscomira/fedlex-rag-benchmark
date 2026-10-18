"""Sensitivity judge: re-score the stored answers of finished runs with another judge model.

    uv run python -m fedlex_bench.eval.rejudge --split test --judge gpt-6-luna RUN_B RUN_C RUN_D

Only fact recall is re-scored (refusals and parse failures score 0, as in the main harness). The
systems are not run again. Writes eval/results/sensitivity_<judge>.json with per-run means and the
paired differences between runs, for the comparisons the report makes.
"""

from __future__ import annotations

import argparse
import itertools
import json

from fedlex_bench.eval.__main__ import BENCH, RESULTS
from fedlex_bench.eval.judge import Judge
from fedlex_bench.eval.runner import load_items
from fedlex_bench.eval.stats import bootstrap_ci, paired_bootstrap


def main(argv: list[str] | None = None) -> None:
    from fedlex_bench.llm import build_client

    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--split", default="test")
    p.add_argument("--judge", required=True)
    args = p.parse_args(argv)
    items = {i["id"]: i for i in load_items(BENCH / f"{args.split}.jsonl")}
    judge = Judge(build_client(args.judge))
    scores: dict[str, dict[str, float]] = {}
    groups: dict[str, str] = {}
    failures: dict[str, int] = {}
    for run in args.runs:
        lines = (RESULTS / run / "per_item.jsonl").read_text("utf-8").splitlines()
        scores[run], failures[run] = {}, 0
        for row in map(json.loads, lines):
            if row["is_trap"]:
                continue
            groups[row["id"]] = row["group"]
            if row.get("status") != "answered" or row.get("parse_failed"):
                scores[run][row["id"]] = 0.0
                continue
            it = items[row["id"]]
            res = judge.fact_recall(it["question"], row["answer"], it["gold_facts"])
            if res.score is None:
                failures[run] += 1
            else:
                scores[run][row["id"]] = res.score
    report: dict = {"judge": args.judge, "runs": {}, "paired": {}}
    for run in args.runs:
        ids = sorted(scores[run])
        report["runs"][run] = {
            "n": len(ids), "judge_failures": failures[run],
            "fact_recall": bootstrap_ci([scores[run][i] for i in ids], [groups[i] for i in ids]),
        }
    for a, b in itertools.combinations(args.runs, 2):
        ids = sorted(set(scores[a]) & set(scores[b]))
        d = paired_bootstrap([scores[a][i] for i in ids], [scores[b][i] for i in ids],
                             [groups[i] for i in ids])
        report["paired"][f"{a} - {b}"] = {"n": len(ids), **d}
    out = RESULTS / f"sensitivity_{args.judge}.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
