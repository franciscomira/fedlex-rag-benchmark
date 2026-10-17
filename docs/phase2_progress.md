# Phase 2 progress log

Plan: `docs/prompts/phase2_kickoff.md`. Spec: `docs/eval_spec.md` (approved 2026-10-17).

## Status

- Step 0 housekeeping: done (repo already public with tag `benchmark-v1`; budget cap 25 USD in `config.toml`, confirm threshold 5 USD).
- Step 1 eval spec: done, approved with the draft's choices (judge pairing, English gloss for calibration, pointers allowed on refusals).
- Step 2 build: done and committed (retrieval, llm/answer, agent, eval core, CLI `python -m fedlex_bench.eval run|report`). All tests pass offline.
- Still to do (lead): calibration sheet generator, cross-family benchmark check script (Step 3).
- Blocked on the user: Anthropic (small prepaid), OpenAI and Gemini keys in `.env`; every next step (reranker, answering, judge, Step 3) is a paid call.

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

### 2026-10-17: Step 3, cross-family check of benchmark v1 (live, Gemini)

Judge: `gemini-3.6-flash` (Google; Phase 1 judge was Claude Opus), same rubric as the `question-judge` agent, run by `python -m fedlex_bench.benchmark.cross_family`. The Claude judge could search the corpus; here code supplies the gold text in DE/FR/IT plus the 6 best retrieved competing articles (hybrid, all-language) per item. Output: `benchmark/machine_review/gemini-judge-1.jsonl` (same sidecar format, bound to item content hashes). Cost about 1 USD of the 25 USD cap.

Result: 242 of 242 items approved, 0 disagreements with the Claude judge (0%, under the 10% stop threshold). **This is weak evidence.** Every verdict has confidence "high", and a negative control shows the judge is lenient: on 20 dev items whose gold facts were swapped for another item's facts it rejected 14 and still approved 5 (74% detection of blatant errors). So the check rules out gross family-specific blind spots but does not show the items are right. The benchmark v1 is unchanged; no v2 is needed on this evidence. A second family (GPT) or a per-fact verification prompt would be stronger; not run (cost about 3 to 4 USD).
