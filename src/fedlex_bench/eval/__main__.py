"""CLI: uv run python -m fedlex_bench.eval run --split dev --method hybrid --config all_lang ..."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

from fedlex_bench.eval.guard import (
    TestRunRefused,
    article_text_lookup,
    log_test_done,
    log_test_start,
    require_clean_tree,
    require_frozen,
)
from fedlex_bench.eval.judge import Judge
from fedlex_bench.eval.runner import load_items, run_eval
from fedlex_bench.eval.systems import PipelineSystem

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "benchmark" / "v1"
RESULTS = ROOT / "eval" / "results"
PARQUET = ROOT / "data" / "processed" / "articles.parquet"
DEFAULT_JUDGE = "gpt-6.1-sol"  # primary judge for every answerer (calibration, 2026-10-17)
GPT_ANSWERER_JUDGE = "gpt-6-luna"  # second judge for sensitivity runs, not a per-family switch


def _git_sha() -> str:
    out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip()


def _judge_ref(answer_ref: str | None, explicit: str | None) -> str:
    """One primary judge for all answerers; pass --judge for a sensitivity run."""
    return explicit or DEFAULT_JUDGE


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
        store=ParquetArticleStore(PARQUET),
    )


def _retrieval_env(args: argparse.Namespace):
    from fedlex_bench.retrieval import RetrievalEnv

    if getattr(args, "embedder", None):  # a fine-tuned checkpoint instead of the base model
        from fedlex_bench.finetune.embedder import TorchE5Embedder

        return RetrievalEnv(embedder=TorchE5Embedder(args.embedder))
    return RetrievalEnv()


def _build(args: argparse.Namespace):
    from fedlex_bench.llm import LLMReranker, build_client
    env = _retrieval_env(args)
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


FROZEN = ROOT / "eval" / "frozen_configs.json"
TEST_LOG = ROOT / "eval" / "test_runs.log"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else "missing"


def _provenance() -> dict:
    """Everything outside the code that a reported number depends on (spec section 8)."""
    from fedlex_bench.agent import prompts as agent_prompts
    from fedlex_bench.answer import prompt_hash as answer_hash
    from fedlex_bench.llm import rerank as rerank_mod

    return {
        "answer_prompt": answer_hash(),
        "rerank_prompt": rerank_mod.prompt_hash(),
        "agent_prompts": agent_prompts.prompt_hash(),
        "benchmark_sha256sums": _sha256_file(BENCH / "SHA256SUMS"),
        "corpus_parquet": _sha256_file(PARQUET),
        "price_table": _sha256_file(ROOT / "eval" / "prices.json"),
        "models_toml": _sha256_file(ROOT / "src" / "fedlex_bench" / "llm" / "models.toml"),
        "bootstrap_seed": 20261017,
    }


def _prepare(args: argparse.Namespace):
    """System, judge client and the configuration hash (independent of split and item count)."""
    from fedlex_bench.eval.judge import prompt_hashes

    system = _build(args)
    judge_client = None
    if args.answer:
        from fedlex_bench.llm import build_client

        judge_client = build_client(_judge_ref(args.answer, args.judge))
    ident = {
        "system": system.config(),
        "judge": judge_client.model_id if judge_client else None,
        "judge_prompts": prompt_hashes(),
        "provenance": _provenance(),
    }
    chash = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:10]
    return system, judge_client, chash, ident


def cmd_freeze(args: argparse.Namespace) -> None:
    """Record a configuration (by hash) as frozen for the test run. No model calls."""
    _, _, chash, ident = _prepare(args)
    frozen = json.loads(FROZEN.read_text("utf-8")) if FROZEN.exists() else {}
    frozen[args.name] = {"hash": chash, "system": ident["system"], "judge": ident["judge"]}
    FROZEN.write_text(json.dumps(frozen, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"frozen {args.name}: {chash}")


def cmd_run(args: argparse.Namespace) -> None:
    items = load_items(BENCH / f"{args.split}.jsonl")
    if args.config == "same_lang":
        items = [i for i in items if i["query_lang"] != "en"]  # undefined for English
    if args.split == "test" and (args.limit or args.run_id):
        raise TestRunRefused("test runs use the whole split and an automatic run id")
    if args.limit:
        items = items[: args.limit]
    system, judge_client, chash, ident = _prepare(args)
    today = dt.date.today().isoformat()
    run_id = args.run_id or f"{today}_{args.split}_{chash}"
    git_sha = _git_sha()
    if args.split == "test":
        name = require_frozen(FROZEN, chash)
        git_sha = require_clean_tree(ROOT)
        if (RESULTS / run_id).exists():
            raise TestRunRefused(f"{run_id} already exists; test results are never overwritten")
        log_test_start(TEST_LOG, run_id=run_id, config_hash=chash, date=today)
        print(f"test run of frozen configuration {name} ({chash})")
    if args.agent:  # traces go next to the run's results, for dev and test alike
        from fedlex_bench.agent.trace import TraceWriter

        system.writer = TraceWriter(RESULTS / run_id / "traces" / "traces.jsonl")
    judge = Judge(judge_client) if judge_client else Judge(_NoJudge())
    out = run_eval(system, items, judge, RESULTS, run_id=run_id, split=args.split,
                   texts=article_text_lookup(PARQUET), git_sha=git_sha,
                   extra_config={"config_hash": chash, "provenance": ident["provenance"],
                                 "judge_prompts": ident["judge_prompts"]})
    if args.split == "test":
        log_test_done(TEST_LOG, run_id=run_id, config_hash=chash, date=today)
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

    def cfg(run: str) -> dict:
        return json.loads((RESULTS / run / "config.json").read_text("utf-8"))

    ca, cb = cfg(args.run_a), cfg(args.run_b)
    for key in ("split", "judge_model"):
        if ca.get(key) != cb.get(key):
            raise SystemExit(f"runs differ in {key}: {ca.get(key)!r} vs {cb.get(key)!r}")
    a, b = rows(args.run_a), rows(args.run_b)
    print(f"A = {args.run_a}")
    print(f"B = {args.run_b}   (paired bootstrap of A - B, 95% CI)")
    for metric in args.metrics:
        ids = [i for i in sorted(a) if i in b and a[i].get(metric) is not None
               and b[i].get(metric) is not None]
        if not ids:
            print(f"{metric:14s} no shared items with a value")
            continue
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
    r.add_argument("--embedder", help="folder of a fine-tuned checkpoint (default: base e5-small)")
    r.add_argument("--rerank", help="model ref, e.g. anthropic/small")
    r.add_argument("--answer", help="model ref, e.g. anthropic/small")
    r.add_argument("--agent", help="model ref for the ADK planner/checker; runs the agent system")
    r.add_argument("--translate", help="model ref: LLM query translation + per-language retrieval")
    r.add_argument("--max-rounds", type=int, default=3)
    r.add_argument("--judge", help="model ref; default gpt-6.1-sol")
    r.add_argument("--retrieve-k", type=int, default=60)
    r.add_argument("--rerank-k", type=int, default=30)
    r.add_argument("--answer-articles", type=int, default=5)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--run-id")
    r.set_defaults(fn=cmd_run)
    fz = sub.add_parser("freeze", parents=[], conflict_handler="resolve")
    for action in r._actions:
        if action.dest not in ("help", "run_id", "limit", "split"):
            fz._add_action(action)
    fz.add_argument("--name", required=True)
    fz.set_defaults(fn=cmd_freeze, split="dev", run_id=None, limit=0)
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
