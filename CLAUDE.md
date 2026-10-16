# Fedlex Cross-Lingual RAG Benchmark

Public portfolio project. Plan: docs/PROJECT_PLAN.md (live version linked inside it).

Status: Phase 1 done (benchmark v1 frozen, tag benchmark-v1, see docs/phase1_report.md). Next: Phase 2, baseline harness.

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

## Multi-agent conventions
- Subagent roles live in .claude/agents/. Use them for independent work; one owner per file.
- Reviewer agents never edit code; they report findings.
