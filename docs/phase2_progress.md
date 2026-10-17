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

### 2026-10-17: LLM reranker on hybrid/all_lang/article, dev (live)

All three providers answered a smoke call. Reranker = listwise, one call per query, re-orders the first N retrieved chunks (the rest keep retriever order). n=63 dev items; R@k on unique article ids.

| System | R@1 | R@10 | MRR | Cost/query USD | p50 / p95 s |
| --- | --- | --- | --- | --- | --- |
| no reranker | 0.10 | 0.46 [0.29, 0.61] | 0.24 | 0 | 0.10 / 0.19 |
| claude-haiku-5-5, N=30 | 0.60 | 0.60 [0.44, 0.75] | 0.64 | 0.0008 | 1.06 / 4.14 |
| gemini-3.5-flash-lite, N=30 | 0.60 | 0.60 [0.44, 0.75] | 0.64 | 0.0016 | 0.77 / 2.18 |
| claude-sonnet-5-5, N=30 | 0.60 | 0.60 [0.44, 0.75] | 0.64 | 0.0165 | 1.34 / 4.37 |
| claude-haiku-5-5, N=60 (retrieve 150) | 0.62 | 0.66 [0.50, 0.81] | 0.67 | 0.0015 | 1.21 / 4.59 |
| claude-haiku-5-5, N=100 (retrieve 150) | 0.62 | 0.69 [0.52, 0.84] | 0.68 | 0.0025 | 1.76 / 4.86 |

Checked because three models gave the same R@1/R@10: not a bug. In 37 of 58 answerable items the gold article is inside the first 30 chunks, and all three models put it first every time; the top-5 lists differ between models. So recall is capped by first-stage recall, not by the reranker, and the models cannot be told apart here. Two honest readings: (1) the first stage is the bottleneck, so widening N helps (0.60 to 0.69, but the intervals overlap, so that is not yet a claim); (2) the questions were drafted from the gold article text, so an LLM finds the matching article trivially once it is a candidate. That makes this benchmark good at separating retrievers and weak at separating strong rerankers; the Phase 3 question (can a small fine-tuned reranker match a frontier LLM) will likely show a ceiling effect. Haiku at 0.0008 USD/query is as good as Sonnet at 0.0165 here (20x cheaper).

Open: `gpt-6-luna` reranker not run, the OpenAI key returned 401 after .env was edited. Spend so far about 2 USD of 25.

### 2026-10-17: LLM reranker across methods and index configs; single-shot answering (dev, live)

**Retrieval + Haiku reranker (retrieve 150, rerank first 60 chunks), recall@10, before -> after rerank.** n=63 (46 for same_lang). Runs `dev_rr_<method>_<config>_haiku_rk60`.

| Method | same_lang | single de | single fr | single it | all_lang |
| --- | --- | --- | --- | --- | --- |
| bm25 | 0.45 -> 0.60 | 0.07 -> 0.19 | 0.13 -> 0.20 | 0.17 -> 0.25 | 0.36 -> 0.45 |
| embedding | 0.44 -> 0.64 | 0.30 -> 0.48 | 0.24 -> 0.41 | 0.27 -> 0.56 | 0.40 -> 0.59 |
| hybrid | 0.52 -> 0.71 | 0.20 -> 0.46 | 0.22 -> 0.44 | 0.31 -> 0.58 | 0.46 -> 0.66 |

The reranker lifts every cell by 8 to 26 points. The reranker's gain is largest where first-stage recall is weakest; hybrid is the best first stage everywhere. Individual cells have wide CIs (about +-15 points), so claim the pattern (rerank > no rerank everywhere; hybrid >= embedding >= BM25 on cross-lingual indexes), not single-cell differences.

**Single-shot answering, same retrieved context for all three (hybrid/all_lang, Haiku rerank N=60, top 5 articles).** n=58 answerable dev items + 5 traps. Judge for Claude and Gemini answers: gpt-6-luna; for GPT answers: gemini-3.5-flash-lite (so the GPT row is judged by a different judge than the other two: not directly comparable until the self-preference check is done).

| Answerer | Fact recall | Faithfulness | Cit precision | Cit recall | Cost/q USD | p50 s | Traps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| claude-haiku-5-5 | 0.55 [0.43, 0.69] | 0.77 [0.71, 0.83] | 0.40 | 0.66 | 0.0021 | 5.9 | 5/5 |
| gpt-6-luna | 0.64 [0.51, 0.77] | 0.89 [0.83, 0.95] | 0.55 | 0.66 | 0.0018 | 4.6 | 5/5 |
| gemini-3.5-flash-lite | 0.50 [0.39, 0.63] | 0.79 [0.70, 0.88] | 0.65 | 0.66 | 0.0025 | 2.5 | 5/5 |

Conditional on the gold article being in the 5 articles shown (40 of 58 items): fact recall 0.78 (Claude), 0.90 (GPT), 0.71 (Gemini), and zero refusals. When it is not shown (18 items): fact recall about 0.05 and refusals 44% / 61% / 89%. So most of the fact-recall gap to 1.0 is retrieval, and refusing when the answer is missing is the right behaviour; the "over-refusal rate" must always be read together with context recall. Differences between answerers are inside the CIs (and confounded by the judge change for GPT); no ranking claimed. Citation precision is low because the models cite every article they used, often 2 or 3 for a one-article gold. Trap accuracy 5/5 everywhere says nothing (n=5). Cost of every row is under 0.003 USD per query.

Calibration sheet for the user drawn from these three runs: `eval/calibration/sheet.md` (30 answers, blind), labels go in `eval/calibration/labels.json`; then `python -m fedlex_bench.eval.calibration agree`.
