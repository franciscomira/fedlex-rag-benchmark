"""Self-preference check (docs/eval_spec.md section 5): score the same answers with two judges.

    uv run python -m fedlex_bench.eval.selfpref dev_ans_claude-haiku-5-5 dev_ans_gpt-6-luna ...

For each run, every answered (not refused) answerable dev answer is judged for fact recall by
both judge models. Reports the mean score of each judge, their paired gap and a 95% interval.
A same-family judge scoring higher than the other-family judge hints at self-preference.
"""

from __future__ import annotations

import argparse
import json

from fedlex_bench.eval.__main__ import BENCH, DEFAULT_JUDGE, GPT_ANSWERER_JUDGE, RESULTS
from fedlex_bench.eval.judge import Judge
from fedlex_bench.eval.runner import load_items
from fedlex_bench.eval.stats import paired_bootstrap



def main(argv: list[str] | None = None) -> None:
    from fedlex_bench.llm import build_client

    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--judges", nargs=2, default=[DEFAULT_JUDGE, GPT_ANSWERER_JUDGE])
    p.add_argument("--out", default="selfpref.json")
    args = p.parse_args(argv)
    JUDGES = tuple(args.judges)
    items = {i["id"]: i for i in load_items(BENCH / "dev.jsonl")}
    judges = {m: Judge(build_client(m)) for m in JUDGES}
    report = {}
    for run in args.runs:
        rows = [json.loads(x) for x in (RESULTS / run / "per_item.jsonl").read_text("utf-8").splitlines()]
        rows = [r for r in rows if not r["is_trap"] and r.get("status") == "answered"
                and not r.get("parse_failed")]
        scores = {m: [] for m in JUDGES}
        groups = []
        for r in rows:
            it = items[r["id"]]
            verdicts = [judges[m].fact_recall(it["question"], r["answer"], it["gold_facts"])
                        for m in JUDGES]
            if any(v.score is None for v in verdicts):
                continue
            for m, v in zip(JUDGES, verdicts, strict=True):
                scores[m].append(v.score)
            groups.append(r["group"])
        a, b = JUDGES
        gap = paired_bootstrap(scores[a], scores[b], groups)
        report[run] = {"n": len(groups), f"mean_{a}": sum(scores[a]) / len(groups),
                       f"mean_{b}": sum(scores[b]) / len(groups), "gap_a_minus_b": gap}
        print(run, json.dumps(report[run]))
    (RESULTS / args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
