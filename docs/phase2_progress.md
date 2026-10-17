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

### 2026-10-17: ADK agent, translate-then-retrieve control, chunking ablation (dev, live, all Haiku 5.5 unless noted)

**Bug found and fixed first.** The first agent run (`dev_agent_haiku`, kept only as a record) refused 33 of 58 answerable questions: the citation checker's JSON was cut off at its 600-token cap in 88 of 130 calls, "unparsable" output fails closed (all citations rejected), and the retry sent an identical request, which the cache answered with the same truncated text. Fix: checker cap 2000 tokens, planner 1500, and the retry request now differs. That is a harness bug fixed on dev, not tuning on test. Its metrics (fact recall 0.30) are not a result.

**Agent vs controls**, n=58 answerable dev items (+5 traps), same judge for all rows (gpt-6-luna), 95% group-bootstrap CIs. Runs: `dev_ans_claude-haiku-5-5` (single-shot), `dev_translate_haiku`, `dev_agent_haiku_v2`.

| System | R@10 | MRR | Fact recall | Faithful | Cost/q USD | p50 / p95 s | Traps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Single-shot (hybrid all_lang + rerank N=60 + answer) | 0.66 [0.50, 0.81] | 0.67 | 0.55 [0.43, 0.69] | 0.77 | 0.0021 | 5.9 / 11.5 | 5/5 |
| Translate-then-retrieve (1 LLM call: query into DE/FR/IT; per-language hybrid, RRF, rerank, answer) | 0.79 [0.63, 0.92] | 0.80 | 0.71 [0.57, 0.83] | 0.82 | 0.0029 | 18.3 / 24.3 | 4/5 |
| ADK agent (planner, 3 language retrievers, citation checker, up to 3 rounds) | 0.93 [0.86, 0.98] | 0.87 | 0.68 [0.57, 0.81] | 0.79 | 0.0025 | 16.2 / 38.1 | 4/5 |

Paired differences (A - B, `fedlex_bench.eval compare`):
- translate vs single-shot: recall@10 +0.13 [+0.03, +0.24], fact recall +0.15 [+0.05, +0.25]: **claimable**. Translating the query into the three languages is the single biggest gain found so far, at +0.001 USD per query.
- agent vs single-shot: recall@10 +0.27 [+0.12, +0.44] claimable; fact recall +0.13 [-0.03, +0.31] not claimable.
- agent vs translate: recall@10 +0.14 [-0.01, +0.32], fact recall -0.02 [-0.21, +0.19]: **no claim**. On this benchmark the agent loop adds retrieval recall but not measurably better answers than the one-call translation control, at similar cost and 3x the latency of single-shot (p95 38 s).

Failure classes of the agent (rule-based, from traces, n=63): ok 53, malformed output 4 (answerer JSON failed twice), refused answerable 4, answered trap 1, wrong-language retriever 1. Step limit hit in 14 runs, checker rejected something in 7.

Sanity on "too good to be true": agent recall@10 is 0.93 overall, 1.00 for DE and FR questions (n=12, 15), 0.87 for EN and IT. It is below the 95% stop line overall, and not a leak: the agent sees only the question and the corpus. It matches Phase 1, where a blind solver agent found the gold article for 149 of 150 questions. Reading: questions drafted from an article's text are easy for an LLM searcher once it writes queries in the article's language, so recall near the ceiling is expected and this benchmark separates retrievers better than strong systems.

**Chunking ablation** (hybrid, all_lang index, no answer; dev recall@10, then with Haiku rerank N=60 of 150):

| Chunking | R@10 no rerank | R@10 rerank | Cost/q rerank USD |
| --- | --- | --- | --- |
| article | 0.46 [0.29, 0.61] | 0.66 [0.50, 0.81] | 0.0015 |
| paragraph | 0.34 [0.20, 0.49] | 0.57 [0.41, 0.71] | 0.0008 |
| fixed window (200 words, 50 overlap) | 0.45 [0.29, 0.60] | 0.66 [0.49, 0.80] | 0.0015 |

Paragraph chunks are worse by about 9 to 12 points (CIs overlap, so not claimed) and cheaper to rerank; window equals article. Answer-level numbers for window vs article follow when the run finishes.

### 2026-10-17: judge self-preference check and a correction to the single-shot table

`python -m fedlex_bench.eval.selfpref` scored the same answered dev answers with both candidate judges (`eval/results/selfpref.json`). **gemini-3.5-flash-lite scores 0.13 to 0.17 higher than gpt-6-luna on every answerer's answers** (Claude -0.13 [-0.23, -0.03], GPT -0.17 [-0.24, -0.09], Gemini -0.15 [-0.22, -0.07], gap = luna minus Gemini judge). The gap is the same size for Gemini's own answers as for the others, so this is a leniency difference between the judges, not a visible self-preference.

**Consequence:** the earlier single-shot table judged the GPT answers with the Gemini judge (the spec's family rule) and the others with luna, so its "GPT best" (0.64 vs 0.55 and 0.50) was a judge artifact. With one judge (gpt-6-luna) for all three (`dev_ans_gpt-6-luna_judgeluna`): fact recall Claude 0.55 [0.43, 0.69], GPT 0.51 [0.38, 0.64], Gemini 0.50 [0.39, 0.63], faithfulness 0.77 / 0.83 / 0.79. The three answerers are indistinguishable on dev. **Rule from now on: cross-model comparisons use one judge for all rows (gpt-6-luna) and the other judge is a sensitivity check;** the spec's per-family judge rule is amended below. Which judge is closer to a human is for the calibration to decide.

Window vs article chunks, with answers (Haiku, same judge): fact recall 0.51 [0.39, 0.63] vs 0.55 [0.43, 0.69], faithfulness 0.75 vs 0.77, cost equal. No difference; article chunks stay the default (simpler, same quality).

### 2026-10-17: judge calibration against 30 hand-labelled dev answers (63 fact labels + 5 trap labels)

Sheet: blind, answers from the three single-shot dev runs, with machine English glosses (the reviewer reads English only). Labels via the local page (`python -m fedlex_bench.eval.calibration label`). Files: `eval/calibration/` (`labels_first_pass.json`, `labels.json` revised, `agreement_first_pass.json`, `agreement_revised.json`).

**First pass (independent of any judge)**, raw fact agreement / kappa, traps 5/5 for every judge:

| Judge | Fact agreement | Kappa |
| --- | --- | --- |
| gpt-6.1-sol | 51/63 = 0.81 | 0.61 |
| claude-sonnet-5-5 | 44/56 = 0.79 (7 facts unparsable) | 0.57 |
| gpt-6-luna | 48/63 = 0.76 | 0.52 |
| claude-haiku-5-5 | 46/63 = 0.73 | 0.48 |
| gemini-3.6-flash | 45/63 = 0.71 | 0.46 |
| gemini-3.5-flash-lite | 44/63 = 0.70 | 0.43 |

The small judges were lenient: luna called 11 facts stated that the reviewer marked not stated, and the reverse only 4 times. This hit the stop rule (below 80%), so I asked; the user had the 15 luna-disagreement facts shown with the judge's verdict and rechecked them.

**Caveat on the revised labels.** After the recheck, all 15 flagged labels were changed to the judge's verdict, so luna agrees 63/63 with the revised labels. Showing the judge's answer next to the label invites anchoring, and flipping all 15 means the revised agreement for luna (and, since the judges mostly agree, for the others: sol 0.95, sonnet 0.96, haiku 0.90, Gemini 3.6 0.89, Flash-Lite 0.87) is not independent evidence. **The first-pass table is the number to report.** The revised labels are kept only to show how stable the ranking is: sol and sonnet stay on top.

**Decision: gpt-6.1-sol is the primary judge for all answerers** (best first-pass agreement, 81%, just over the 80% bar; the margin over luna is 3 facts, so not a significant difference). Cost: about 0.003 to 0.005 USD per judged answer.

Self-preference check for sol vs claude-sonnet-5-5 on dev answers (`selfpref_sol_sonnet.json`, gap = sol minus sonnet): Claude answers -0.10 [-0.18, -0.02], GPT answers -0.04 [-0.10, +0.02], Gemini answers -0.03 [-0.07, 0.00]. Sonnet scores Claude's answers about 0.06 to 0.07 higher relative to sol than it scores the others (a hint of same-family favouring by the Claude judge); sol shows no sign of favouring GPT answers (its GPT gap equals its Gemini gap). So the primary judge's family does not visibly inflate the GPT answerer.

Limitations to report: 30 answers, one non-lawyer English-only labeller working from machine glosses, one labelling pass independent of any judge; judge agreement of about 80% means fact-recall differences below a few points are not resolvable by the judge either.

### 2026-10-17: dev table with the primary judge (gpt-6.1-sol), the three answering systems

Runs `dev_sol_B_singleshot`, `dev_sol_C_translate`, `dev_sol_D_agent` (all Claude Haiku 5.5 for rerank/translate/plan/answer). n=58 answerable + 5 traps; judge cost 1.11 USD for the three.

| System | R@10 | MRR | Fact recall | Faithful | Cit P | Cit R | Cost/q USD | p50 / p95 s | Traps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B single-shot | 0.66 [0.50, 0.81] | 0.67 | 0.54 [0.42, 0.67] | 0.82 [0.77, 0.87] | 0.40 | 0.66 | 0.0021 | 5.9 / 11.4 | 5/5 |
| C translate-then-retrieve | 0.79 [0.63, 0.92] | 0.80 | 0.66 [0.52, 0.79] | 0.86 [0.81, 0.91] | 0.46 | 0.79 | 0.0029 | 11.4 / 18.2 | 4/5 |
| D ADK agent | 0.97 [0.91, 1.00] | 0.93 | 0.72 [0.61, 0.84] | 0.85 [0.81, 0.89] | 0.53 | 0.93 | 0.0027 | 16.5 / 38.0 | 4/5 |

Paired (A - B): C vs B recall@10 +0.13 [+0.03, +0.24], fact recall +0.12 [+0.03, +0.21], faithfulness +0.05 [+0.01, +0.09]: claimable. D vs B recall@10 +0.30 [+0.15, +0.48], fact recall +0.18 [+0.02, +0.35]: claimable; faithfulness no claim. D vs C recall@10 +0.17 [+0.03, +0.35] claimable; fact recall +0.06 [-0.11, +0.24] and faithfulness -0.01: no claim. With the stronger judge, the agent's answer quality edge over single-shot becomes claimable, and its edge over the translation control stays unproven; retrieval recall is where the agent clearly wins.

**Run-to-run noise (measured):** the agent's dev recall@10 went from 0.93 (`dev_agent_haiku_v2`) to 0.97 (`dev_sol_D_agent`) with an identical configuration. 5 of 63 items changed; the 4 answerer parse failures of the first run did not recur (failed calls are not cached, so they are retried on a rerun; 208 of 222 model calls were cache hits). So one agent run carries a noise of a few points from transient provider failures, and this is the reason the report states `malformed_output` counts and does not read single items. Recall@10 0.97 is above the "too good" line of 95%, so I checked again: same explanation as before (the agent searches in the article's language; Phase 1's blind solver found 149 of 150 gold articles), and the CI lower bound is 0.91. Reported as it is, with that caveat, and the test split will show whether it holds.

## Freeze for the test run (written 2026-10-17, before any test number exists)

Four configurations, each run once on `benchmark/v1/test.jsonl` (179 items: 125 originals + 54 translations), primary judge `gpt-6.1-sol`, clean git tree, logged in `eval/test_runs.log`. Models are the dated ids pinned in `src/fedlex_bench/llm/models.toml` and prices in `eval/prices.json` (2026-10-17).

| ID | System | Command (run on test with `--split test`) |
| --- | --- | --- |
| A | Retrieval baseline, no model: hybrid, all_lang, article chunks | `run --method hybrid --config all_lang` |
| B | Single-shot: A + Haiku rerank (150 retrieved, first 60 reranked) + Haiku cited answer, top 5 articles | `run --method hybrid --config all_lang --rerank claude-haiku-5-5 --retrieve-k 150 --rerank-k 60 --answer claude-haiku-5-5` |
| C | Translate-then-retrieve: Haiku translates the query into DE/FR/IT, per-language hybrid (60 each), RRF, Haiku rerank 60, Haiku answer | `run --method hybrid --translate claude-haiku-5-5 --rerank claude-haiku-5-5 --retrieve-k 60 --rerank-k 60 --answer claude-haiku-5-5` |
| D | ADK agent: Haiku planner and checker, per-language hybrid retrievers, up to 3 rounds, Haiku answerer | `run --method hybrid --agent claude-haiku-5-5 --answer claude-haiku-5-5` |

Why these four: A is the no-model floor; B is the standard reranker-plus-answerer baseline; C is the control that isolates query translation; D is the agent. Not frozen, with reasons: GPT and Gemini answerers (no difference on dev once judged by one judge), Sonnet reranker (identical recall to Haiku at 20x the cost), paragraph and window chunks (no gain), other index configurations (cross-lingual all_lang is the headline setting; the matrix stays a dev result).
Hypotheses stated now: D > C > B on recall@10 and fact recall; the C vs B gap is claimable on test; D vs C answer quality is unlikely to be resolvable with 110 answerable test items.
