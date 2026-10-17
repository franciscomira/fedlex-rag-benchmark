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


def _build(args: argparse.Namespace):
    from fedlex_bench.llm import LLMReranker, build_client
    from fedlex_bench.retrieval import RetrievalEnv

    retriever = RetrievalEnv().retriever(args.method, args.config, args.chunking)
    reranker = LLMReranker(build_client(args.rerank)) if args.rerank else None
    answerer = None
    if args.answer:
        from fedlex_bench.answer import LLMAnswerer

        answerer = LLMAnswerer(build_client(args.answer))
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
    r.add_argument("--judge", help="model ref; default by answerer family")
    r.add_argument("--retrieve-k", type=int, default=60)
    r.add_argument("--rerank-k", type=int, default=30)
    r.add_argument("--answer-articles", type=int, default=5)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--run-id")
    r.set_defaults(fn=cmd_run)
    t = sub.add_parser("report")
    t.add_argument("runs", nargs="+", help="run ids under eval/results/")
    t.set_defaults(fn=cmd_report)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
