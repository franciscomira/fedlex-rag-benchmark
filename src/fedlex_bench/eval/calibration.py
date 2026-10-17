"""Judge calibration against hand labels (docs/eval_spec.md section 5).

    uv run python -m fedlex_bench.eval.calibration sample dev_ans_a dev_ans_b ...
    # the user fills eval/calibration/labels.json
    uv run python -m fedlex_bench.eval.calibration agree

`sample` draws 30 dev answers (blind to the model that wrote them), writes a reading sheet with
a machine English gloss of each answer, and a labels template. `agree` re-runs the judge on the
same answers (served from the cache) and compares with the labels.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SEED = 20261017
N_TRAPS = 5
N_EDGE = 5  # answerable but refused or without citations
N_TOTAL = 30
CALIB_DIR = Path(__file__).resolve().parents[3] / "eval" / "calibration"


def _is_edge(r: Mapping) -> bool:
    return (not r["is_trap"]) and (r["status"] == "refused" or not r["citations"])


def draw_sample(pool: Sequence[Mapping], seed: int = SEED) -> list[dict]:
    """One answer per item: 5 traps, 5 refused/uncited answerable, the rest ordinary answers."""
    rng = random.Random(seed)
    ordered = sorted(pool, key=lambda r: (r["id"], r["answerer"]))
    by_item: dict[str, list[Mapping]] = {}
    for r in ordered:
        by_item.setdefault(r["id"], []).append(r)

    def pick(kind: str, n: int, taken: set[str]) -> list[dict]:
        ids = sorted(
            i for i, rs in by_item.items()
            if i not in taken and any(_kind(r) == kind for r in rs)
        )
        out = []
        for i in rng.sample(ids, min(n, len(ids))):
            out.append(dict(rng.choice([r for r in by_item[i] if _kind(r) == kind])))
            taken.add(i)
        return out

    taken: set[str] = set()
    chosen = pick("trap", N_TRAPS, taken) + pick("edge", N_EDGE, taken)
    chosen += pick("normal", N_TOTAL - len(chosen), taken)
    return sorted(chosen, key=lambda r: r["id"])


def _kind(r: Mapping) -> str:
    return "trap" if r["is_trap"] else "edge" if _is_edge(r) else "normal"


def _kappa(a: Sequence[bool], b: Sequence[bool]) -> float | None:
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def agreement(judge: Mapping[str, Any], human: Mapping[str, Any]) -> dict[str, Any]:
    """Per-fact verdict lists (answerable ids) or one bool (trap ids) for judge and human."""
    jf, hf, jt, ht = [], [], [], []
    for i in sorted(human):
        if isinstance(human[i], list):
            jf += list(judge[i])
            hf += list(human[i])
        else:
            jt.append(judge[i])
            ht.append(human[i])
    fa = sum(x == y for x, y in zip(jf, hf, strict=True))
    ta = sum(x == y for x, y in zip(jt, ht, strict=True))
    total = len(jf) + len(jt)
    return {
        "facts": {"n": len(jf), "agree": fa, "raw": fa / len(jf) if jf else None,
                  "kappa": _kappa(jf, hf)},
        "traps": {"n": len(jt), "agree": ta, "raw": ta / len(jt) if jt else None},
        "overall_raw": (fa + ta) / total if total else None,
    }


def _load_pool(run_ids: Sequence[str], results: Path) -> list[dict]:
    pool = []
    for run in run_ids:
        for line in (results / run / "per_item.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("status") is not None and not r.get("parse_failed"):
                # traps the code already fails never reach the judge: nothing to calibrate
                if r["is_trap"] and r["status"] != "refused":
                    continue
                if r["is_trap"] and r["citations"]:
                    continue
                pool.append({**r, "answerer": run, "run": run})
    return pool


GLOSS_SYSTEM = (
    "Translate the text into English. Keep legal terms and article numbers. Output only the "
    "translation, no comments."
)


def cmd_sample(args: argparse.Namespace) -> None:
    from fedlex_bench.eval.__main__ import BENCH, RESULTS
    from fedlex_bench.eval.runner import load_items
    from fedlex_bench.llm import build_client
    from fedlex_bench.types import LLMRequest

    items = {i["id"]: i for s in ("dev",) for i in load_items(BENCH / f"{s}.jsonl")}
    sample = draw_sample(_load_pool(args.runs, RESULTS), seed=SEED)
    client = build_client(args.gloss_model)
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    sheet = ["# Calibration sheet (answerer hidden)", "",
             "For each item, mark every gold fact as supported (true) or not (false) by the "
             "answer, in `labels.json`. Traps: set `asserts_answer` true if the reply states the "
             "specific information asked for as if it were established. The English text is a "
             "machine gloss, labelled as such.", ""]
    template: dict[str, Any] = {}
    meta = []
    for r in sample:
        it = items[r["id"]]
        gloss = r["answer"]
        if it["query_lang"] != "en" and r["answer"].strip():
            gloss = client.complete(LLMRequest(GLOSS_SYSTEM, r["answer"], 1500)).text
        sheet += [f"## {r['id']} ({it['query_lang']}, {it['qtype']})", f"**Question:** {it['question']}", "",
                  f"**Reply (original):** {r['answer']}", "", f"**Reply (machine English gloss):** {gloss}", ""]
        if it["is_trap"]:
            sheet += [f"Why unanswerable: {it['trap_reason']}", ""]
            template[r["id"]] = {"asserts_answer": None}
        else:
            sheet += ["Gold facts:"] + [f"{k}. {f['text']}" for k, f in enumerate(it["gold_facts"])] + [""]
            template[r["id"]] = {"facts": [None] * len(it["gold_facts"])}
        meta.append({"id": r["id"], "run": r["run"], "kind": _kind(r)})
    (CALIB_DIR / "sheet.md").write_text("\n".join(sheet) + "\n", encoding="utf-8")
    (CALIB_DIR / "labels.json").write_text(json.dumps(template, indent=1), encoding="utf-8")
    (CALIB_DIR / "sample.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"wrote {CALIB_DIR / 'sheet.md'} ({len(sample)} items)")


def cmd_agree(args: argparse.Namespace) -> None:
    """Agreement of each candidate judge model with the hand labels (answers come from the cache)."""
    from fedlex_bench.eval.__main__ import BENCH, DEFAULT_JUDGE, GPT_ANSWERER_JUDGE, RESULTS
    from fedlex_bench.eval.judge import Judge
    from fedlex_bench.eval.runner import load_items
    from fedlex_bench.llm import build_client

    items = {i["id"]: i for i in load_items(BENCH / "dev.jsonl")}
    labels = json.loads((CALIB_DIR / "labels.json").read_text(encoding="utf-8"))
    meta = json.loads((CALIB_DIR / "sample.json").read_text(encoding="utf-8"))
    reports: dict[str, Any] = {}
    for model in (DEFAULT_JUDGE, GPT_ANSWERER_JUDGE):
        judge = Judge(build_client(model))
        human: dict[str, Any] = {}
        judge_out: dict[str, Any] = {}
        for m in meta:
            lab = labels[m["id"]]
            lines = (RESULTS / m["run"] / "per_item.jsonl").read_text(encoding="utf-8").splitlines()
            row = next(r for r in map(json.loads, lines) if r["id"] == m["id"])
            it = items[m["id"]]
            if it["is_trap"]:
                if lab["asserts_answer"] is None:
                    continue
                human[m["id"]] = lab["asserts_answer"]
                leak = judge.trap_leak(it["question"], row["answer"], it["trap_reason"])
                judge_out[m["id"]] = leak.leaked
            else:
                if any(v is None for v in lab["facts"]):
                    continue
                human[m["id"]] = lab["facts"]
                fr = judge.fact_recall(it["question"], row["answer"], it["gold_facts"])
                judge_out[m["id"]] = fr.verdicts
        usable = [i for i in human if judge_out[i] is not None and judge_out[i] != []]
        report = agreement({i: judge_out[i] for i in usable}, {i: human[i] for i in usable})
        report["n_items_labelled"] = len(human)
        report["n_items_scored"] = len(usable)
        reports[model] = report
    (CALIB_DIR / "agreement.json").write_text(json.dumps(reports, indent=1), encoding="utf-8")
    print(json.dumps(reports, indent=1))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("runs", nargs="+")
    s.add_argument("--gloss-model", default="gemini-3.5-flash-lite")
    s.set_defaults(fn=cmd_sample)
    a = sub.add_parser("agree")
    a.set_defaults(fn=cmd_agree)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
