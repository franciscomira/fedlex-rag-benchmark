# Phase 2 progress log

Plan: `docs/prompts/phase2_kickoff.md`. Spec: `docs/eval_spec.md` (approved 2026-10-17).

## Status

- Step 0 housekeeping: done (repo already public with tag `benchmark-v1`; budget cap 25 USD in `config.toml`, confirm threshold 5 USD).
- Step 1 eval spec: done, approved with the draft's choices (judge pairing, English gloss for calibration, pointers allowed on refusals).
- Step 2 build: done and committed (retrieval, llm/answer, agent, eval core, CLI `python -m fedlex_bench.eval run|report`). All tests pass offline.
- Old note (superseded): Lead part done (`src/fedlex_bench/eval/`: metrics, bootstrap, judge roles, runner, guards; interfaces in `src/fedlex_bench/types.py`). Three subagents building `retrieval/`, `llm/` + `answer/`, `agent/` against the interfaces; no live API calls until `.env` exists.
- Still to do in the lead part: CLI (`python -m fedlex_bench.eval run|report`), results table, calibration sheet generator.
- Blocked on the user: Anthropic API key (small prepaid), OpenAI key, Gemini key into `.env`.

## Experiment log

### 2026-10-17: retrieval only, dev, article chunks, no reranker (free, no API)

Runs: `eval/results/dev_ret_<method>_<config>_article/`. Command: `uv run python -m fedlex_bench.eval run --split dev --method M --config C`, report with `... report <run ids>`. n=63 dev items incl. 18 translations (46 for same_lang, English excluded); CIs are 95% group bootstrap.

| Config | BM25 R@10 | Embedding R@10 | Hybrid R@10 |
| --- | --- | --- | --- |
| all_lang | 0.36 | 0.40 | 0.46 [0.29, 0.61] |
| same_lang | 0.45 | 0.44 | 0.52 [0.34, 0.69] |
| single_lang_de | 0.07 | 0.30 | 0.20 |
| single_lang_fr | 0.13 | 0.24 | 0.22 |
| single_lang_it | 0.17 | 0.27 | 0.31 |

Read: cross-lingual retrieval is weak for every method (as expected for BM25; multilingual-e5-small is also modest). Hybrid on the all-language index is best of the cross-lingual configs but the CIs overlap widely (±15 points), so hybrid vs embedding-only is not a claim. BM25 alone collapses on single-language indexes (little shared vocabulary). Nothing here looks too good. Next: LLM reranker on top (needs API keys).
