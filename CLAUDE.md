# Fedlex Cross-Lingual RAG Benchmark

Public portfolio project. Plan: docs/PROJECT_PLAN.md (live version linked inside it).

Status: Phase 3 done (fine-tuned embedding model; docs/phase3_report.md; Phase 2 has an errata section for an embedder bug found in Phase 3). Next: Phase 4, ship (demo, MCP server, results page). Benchmark v1 frozen, tag benchmark-v1.

## Core question
Can a small fine-tuned multilingual reranker match a frontier LLM reranker on cross-lingual retrieval over Swiss federal law (DE/FR/IT), at a fraction of the cost? Answers must cite article-level sources.

## Rules
- Measure first. Every pipeline change is judged by the eval harness, never by eyeballing.
- Reproducibility: fixed seeds, deterministic file ordering (sort everything), pinned dependencies. One command reproduces any reported number.
- Report numbers honestly, including unflattering ones. Never tune on the test split.
- Keep the model away from what must be exact: article ids, SR numbers and citations are handled by code, not generated.
- Raw data never goes into git (data/raw is ignored). Secrets live in .env only.
- Small, reviewable commits with clear messages. Ask before pushing or creating a remote.

## Stack
- Python 3.11+, managed with uv. Code in src/fedlex_bench, tests in tests/ (pytest).
- Data: Parquet/JSONL in data/processed. Benchmark files in benchmark/.
- Platform: Windows 11 (PowerShell). Use pathlib, never hard-coded separators.

## Identifiers
- Act id = SR number (e.g. "220" for the Code of Obligations).
- Article id = "<SR>:<article number>" (e.g. "220:257d"). Same id across languages; that is the alignment key.

## Evaluation rules (Phase 2 onward)
- Specs: docs/eval_spec.md (metrics, judge, statistics, run layout) and eval/README.md (commands, test-split rules). Results: docs/phase2_report.md.
- benchmark/v1 is read-only. Dev is for choices; test.jsonl runs once per frozen configuration (`eval freeze`, then `eval run --split test`); never tune on it.
- One judge (gpt-6.1-sol) scores all answerers; cross-model comparisons never mix judges. Paid runs respect config.toml (25 USD per month, confirm above 5 USD); keys live in .env only.
- Report intervals (95% group bootstrap) with every number and say which differences are not claimable.

## Multi-agent conventions
- Subagent roles live in .claude/agents/. Use them for independent work; one owner per file.
- Reviewer agents never edit code; they report findings.
