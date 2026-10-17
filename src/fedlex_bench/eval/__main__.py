"""CLI: uv run python -m fedlex_bench.eval run --split dev --method hybrid --config all_lang ..."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

from fedlex_bench.eval.guard import article_text_lookup, log_test_run, require_clean_tree
from fedlex_bench.eval.judge import Judge
from fedlex_bench.eval.runner import load_items, run_eval
from fedlex_bench.eval.systems import PipelineSystem

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "benchmark" / "v1"
RESULTS = ROOT / "eval" / "results"
PARQUET = ROOT / "data" / "processed" / "articles.parquet"
DEFAULT_JUDGE = "gpt-6-luna"  # different family from Claude and Gemini answerers
GPT_ANSWERER_JUDGE = "gemini-3.5-flash-lite"  # spec section 5: GPT answers get a Gemini judge


def _git_sha() -> str:
    out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip()


def _judge_ref(answer_ref: str | None, explicit: str | None) -> str:
    if explicit:
        return explicit
    if answer_ref and (answer_ref.startswith("openai/") or answer_ref.startswith("gpt-")):
        return GPT_ANSWERER_JUDGE
    return DEFAULT_JUDGE


def _build_agent(args, env, answerer):
    from fedlex_bench.agent.store import ParquetArticleStore
    from fedlex_bench.agent.system import FedlexAgentSystem
    from fedlex_bench.llm import build_client

    retrievers = {
        lang: env.retriever(args.method, f"single_lang_{lang}", args.chunking)
        for lang in ("de", "fr", "it")
    }
    return FedlexAgentSystem(
        retrievers,
        answerer,
        build_client(args.agent),
        max_rounds=args.max_rounds,
        trace_dir=RESULTS / (args.run_id or "agent_traces") / "traces" if args.run_id else None,
        store=ParquetArticleStore(PARQUET),
    )


def _build(args: argparse.Namespace):
    from fedlex_bench.llm import LLMReranker, build_client
    from fedlex_bench.retrieval import RetrievalEnv

    env = RetrievalEnv()
    retriever = env.retriever(args.method, args.config, args.chunking)
    reranker = (
        LLMReranker(build_client(args.rerank), max_candidates=args.rerank_k) if args.rerank else None
    )
    answerer = None
    if args.answer:
        from fedlex_bench.answer import LLMAnswerer

        answerer = LLMAnswerer(build_client(args.answer))
    if args.agent:
        return _build_agent(args, env, answerer)
    if args.translate:
        from fedlex_bench.eval.systems import TranslateRetrieveSystem
        from fedlex_bench.llm import build_client as _bc

        per_lang = {
            lang: env.retriever(args.method, f"single_lang_{lang}", args.chunking)
            for lang in ("de", "fr", "it")
        }
        return TranslateRetrieveSystem(
            per_lang,
            _bc(args.translate),
            reranker=reranker,
            answerer=answerer,
            retrieve_k=args.retrieve_k,
            rerank_k=args.rerank_k,
            answer_articles=args.answer_articles,
        )
    return PipelineSystem(
        retriever,
        reranker=reranker,
        answerer=answerer,
        retrieve_k=args.retrieve_k,
        rerank_k=args.rerank_k,
        answer_articles=args.answer_articles,
    )


def cmd_run(args: argparse.Namespace) -> None:
    items = load_items(BENCH / f"{args.split}.jsonl")
    if args.config == "same_lang":
        items = [i for i in items if i["query_lang"] != "en"]  # undefined for English
    if args.limit:
        items = items[: args.limit]
    system = _build(args)
    judge_client = None
    if args.answer:
        from fedlex_bench.llm import build_client

        judge_client = build_client(_judge_ref(args.answer, args.judge))
    settings = {**system.config(), "judge": judge_client.model_id if judge_client else None,
                "split": args.split, "n_items": len(items)}
    chash = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:10]
    run_id = args.run_id or f"{dt.date.today().isoformat()}_{args.split}_{chash}"
    git_sha = _git_sha()
    if args.split == "test":
        git_sha = require_clean_tree(ROOT)
        log_test_run(ROOT / "eval" / "test_runs.log", run_id=run_id, config_hash=chash,
                     date=dt.date.today().isoformat())
    judge = Judge(judge_client) if judge_client else Judge(_NoJudge())
    out = run_eval(system, items, judge, RESULTS, run_id=run_id, split=args.split,
                   texts=article_text_lookup(PARQUET), git_sha=git_sha,
                   extra_config={"config_hash": chash})
    print(out)


class _NoJudge:
    """Retrieval-only runs never call the judge; fail loudly if one is needed."""

    model_id = "none"

    def complete(self, request):
        raise RuntimeError("judge called on a run without an answerer")


def cmd_report(args: argparse.Namespace) -> None:
    from fedlex_bench.eval.report import render_table

    print(render_table([RESULTS / r for r in args.runs]))


def cmd_compare(args: argparse.Namespace) -> None:
    from fedlex_bench.eval.stats import paired_bootstrap

    def rows(run: str) -> dict[str, dict]:
        path = RESULTS / run / "per_item.jsonl"
        return {r["id"]: r for r in map(json.loads, path.read_text("utf-8").splitlines())}

    a, b = rows(args.run_a), rows(args.run_b)
    print(f"A = {args.run_a}")
    print(f"B = {args.run_b}   (paired bootstrap of A - B, 95% CI)")
    for metric in args.metrics:
        ids = [i for i in sorted(a) if i in b and a[i].get(metric) is not None
               and b[i].get(metric) is not None]
        d = paired_bootstrap([float(a[i][metric]) for i in ids], [float(b[i][metric]) for i in ids],
                             [a[i]["group"] for i in ids])
        verdict = "differs" if d["lo"] > 0 or d["hi"] < 0 else "no claim"
        print(f"{metric:14s} n={len(ids):3d} diff {d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]  {verdict}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="fedlex_bench.eval")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--split", choices=["dev", "test"], default="dev")
    r.add_argument("--method", choices=["bm25", "embedding", "hybrid"], default="hybrid")
    r.add_argument("--config", default="all_lang")
    r.add_argument("--chunking", default="article")
    r.add_argument("--rerank", help="model ref, e.g. anthropic/small")
    r.add_argument("--answer", help="model ref, e.g. anthropic/small")
    r.add_argument("--agent", help="model ref for the ADK planner/checker; runs the agent system")
    r.add_argument("--translate", help="model ref: LLM query translation + per-language retrieval")
    r.add_argument("--max-rounds", type=int, default=3)
    r.add_argument("--judge", help="model ref; default by answerer family")
    r.add_argument("--retrieve-k", type=int, default=60)
    r.add_argument("--rerank-k", type=int, default=30)
    r.add_argument("--answer-articles", type=int, default=5)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--run-id")
    r.set_defaults(fn=cmd_run)
    c = sub.add_parser("compare")
    c.add_argument("run_a")
    c.add_argument("run_b")
    c.add_argument("--metrics", nargs="+",
                   default=["recall@10", "mrr", "fact_recall", "faithfulness", "cost_usd"])
    c.set_defaults(fn=cmd_compare)
    t = sub.add_parser("report")
    t.add_argument("runs", nargs="+", help="run ids under eval/results/")
    t.set_defaults(fn=cmd_report)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
