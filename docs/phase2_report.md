# Phase 2 report: baseline harness, provider comparison, agent

Date: 2026-10-18. Benchmark: `benchmark/v1/` (tag `benchmark-v1`, unchanged). Spec: [eval_spec.md](eval_spec.md) with its dated amendments. Working log: [phase2_progress.md](phase2_progress.md). Harness review: [phase2_harness_review.md](phase2_harness_review.md).

> **Errata, 2026-10-28.** A bug in the ONNX embedder (padding tokens were averaged into the vector of every batched text, so the passage vectors of the index were degraded) affected every retrieval number in this report that uses embeddings: A, B, C and D. The statements below were measured with the buggy embedder and are kept as published. See [Errata](#errata-2026-10-28-embedder-padding-bug) at the end for the corrected system A (test recall@10 0.55 against 0.49, English 0.24 against 0.14) and for what is and is not known about B, C and D.

## Summary

- Four frozen configurations were each run once on the test split (179 items: 110 original answerable questions, 54 translations of 18 of them, 15 traps; 164 answerable items in the tables). Configurations, hashes and the hypotheses were written down before any test number existed.
- **Retrieval is the bottleneck, and the gain from better queries is concentrated in English questions.** A plain hybrid retriever finds the gold article in the top 10 for 49% of test questions, but for only 14% of English questions (German 58%, French 65%, Italian 60%). Asking a small LLM to translate the question into German, French and Italian and searching each language index (C) lifts the overall figure to 81% (English 71%); an ADK agent that plans its own searches (D) reaches 92%. In German, French and Italian taken alone, C over single-shot is +0.04 to +0.09 recall@10 and none of those gaps is claimable; the large claimable gaps are in English (+0.46).
- **Answer quality follows retrieval.** Fact recall (judge gpt-6.1-sol): single-shot 0.61, translate-then-retrieve 0.76, agent 0.81. C over single-shot (+0.16) and the agent over single-shot (+0.20) are claimable; **agent over C (+0.04, CI -0.05 to +0.14) is not**. The agent retrieves more than C (+0.11 recall@10, lower bound +0.01): a borderline, unplanned-for-correction result, to be treated as exploratory. The same conclusions hold with the second judge gpt-6-luna (fact recall 0.63 / 0.78 / 0.83; C over B +0.15 [+0.08, +0.23], D over B +0.20 [+0.10, +0.30], D over C +0.05 [-0.05, +0.14]).
- **Cost and latency.** All pipelines cost 0.002 to 0.003 USD per question with Claude Haiku 5.5. The agent has a p95 latency of 45 s against 10 s for single-shot.
- **Traps remain unsolved:** 10, 10 and 11 of 15 trap questions were handled correctly by the three answering systems (exact 95% intervals [0.38, 0.88], [0.38, 0.88], [0.45, 0.92]); the failures are spread over three trap types (see limitations).
- Total API spend for the phase: 10.9 USD of the 25 USD monthly cap.
- This is a baseline for the phase 3 question (can a small fine-tuned reranker match a frontier LLM reranker?). On dev, Claude Haiku, Gemini Flash-Lite, GPT-6 Luna and Sonnet rerankers all gave the same recall@10 once the gold article was among the candidates, so on this benchmark the reranker is not where the models differ; first-stage retrieval is.

## Systems and protocol

All models are dated ids pinned in `src/fedlex_bench/llm/models.toml`, prices in `eval/prices.json` (2026-10-17). Everything below with an LLM used Claude Haiku 5.5 ($0.10 / $0.50 per million input / output tokens), judged by gpt-6.1-sol.

| ID | System |
| --- | --- |
| A | Hybrid retrieval (BM25 + multilingual-e5-small embeddings, RRF) over the index of all three languages, article chunks. No model. |
| B | A, then a listwise LLM reranker over the first 60 of 150 retrieved chunks, then a cited answer from the top 5 articles. |
| C | A Haiku call translates the question into DE, FR and IT; each language index is searched; results are fused with RRF; then B's reranker and answerer. |
| D | Google ADK agent: planner, three per-language retriever agents, citation checker that can trigger another search (at most 3 rounds), cited answer. |

Answers are structured (`answered`/`refused`, citations, pointers); code validates every cited id against the retrieved set. Statistics: 95% percentile bootstrap, 10,000 resamples, resampling question groups (a question and its translations stay together), seed 20261017. "Claimable" below means the paired-difference interval excludes zero; no multiple-comparison correction was applied.

## Test results (n = 164 answerable items)

| | R@1 | R@10 | All gold@10 | MRR | Fact recall | Faithfulness | Cit. precision | Cit. recall | Over-refusal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A retrieval only | 0.22 [0.15, 0.29] | 0.49 [0.40, 0.58] | 0.46 [0.38, 0.55] | 0.32 [0.25, 0.40] | n/a | n/a | n/a | n/a | n/a |
| B single-shot | 0.59 [0.49, 0.68] | 0.65 [0.56, 0.73] | 0.63 [0.54, 0.72] | 0.64 [0.55, 0.73] | 0.61 [0.52, 0.69] | 0.85 [0.82, 0.88] | 0.55 [0.48, 0.64] | 0.79 [0.72, 0.86] | 0.18 [0.12, 0.25] |
| C translate + per-language indexes | 0.74 [0.65, 0.83] | 0.81 [0.73, 0.89] | 0.80 [0.71, 0.88] | 0.80 [0.71, 0.88] | 0.76 [0.68, 0.84] | 0.86 [0.83, 0.89] | 0.57 [0.49, 0.65] | 0.85 [0.78, 0.92] | 0.05 [0.02, 0.10] |
| D agent | 0.78 [0.70, 0.85] | 0.92 [0.87, 0.97] | 0.91 [0.86, 0.96] | 0.87 [0.81, 0.93] | 0.81 [0.75, 0.86] | 0.86 [0.84, 0.89] | 0.61 [0.55, 0.68] | 0.93 [0.87, 0.97] | 0.07 [0.04, 0.11] |

Parse failures: 0 for B, C, D. Judge failures: 0 in all runs. Originals only (n = 110, no translations): recall@10 0.51 / 0.65 / 0.82 / 0.93 and fact recall n/a / 0.60 / 0.75 / 0.80 for A / B / C / D, so translations do not drive the picture.

Faithfulness is about 0.85 for all three answering systems: roughly one claim in seven is not supported by the cited article text, whatever the retrieval. Citation precision of 0.55 to 0.61 is low because the models cite every article they used (two or three) when the gold is one article.

### Paired differences (first system minus second, 95% interval)

| Comparison | Recall@10 | MRR | Fact recall | Faithfulness |
| --- | --- | --- | --- | --- |
| B vs A (reranker + answerer over bare retrieval) | +0.16 [+0.10, +0.22] | +0.32 [+0.25, +0.40] | n/a | n/a |
| C vs B (query translation, per-language indexes, smaller candidate pool) | +0.17 [+0.09, +0.24] | +0.16 [+0.08, +0.23] | +0.16 [+0.09, +0.23] | +0.01 [-0.02, +0.04] |
| D vs B (agent over single-shot) | +0.28 [+0.18, +0.39] | +0.23 [+0.12, +0.34] | +0.20 [+0.10, +0.30] | +0.01 [-0.02, +0.04] |
| D vs C (agent over translation) | +0.11 [+0.01, +0.22] | +0.07 [-0.03, +0.18] | +0.04 [-0.05, +0.14] | 0.00 [-0.02, +0.03] |

The planned hypothesis was D > C > B on retrieval and fact recall, with the C-over-B gap claimable and D-over-C answer quality probably not resolvable. The results match: the agent beats the translation control on recall@10 (barely: lower bound +0.01) but not on fact recall.

### By question language (recall@10 | fact recall)

| Language (n) | A | B | C | D |
| --- | --- | --- | --- | --- |
| English (42) | 0.14 | 0.25 \| 0.26 | 0.71 \| 0.68 | 0.92 \| 0.84 |
| German (43) | 0.58 | 0.76 \| 0.74 | 0.85 \| 0.84 | 0.92 \| 0.80 |
| French (40) | 0.65 | 0.79 \| 0.67 | 0.84 \| 0.75 | 0.95 \| 0.78 |
| Italian (39) | 0.60 | 0.81 \| 0.77 | 0.85 \| 0.79 | 0.91 \| 0.80 |

English questions are the hardest cross-lingual case, since no article is in English: bare hybrid retrieval gets 14%, the reranker lifts it only to 25% because the right article is rarely among the candidates, and translating the query is what fixes it (71%). Paired differences by language (first minus second, 95% interval):

| Language (n) | C vs B recall@10 | D vs B recall@10 | D vs C recall@10 | C vs B fact | D vs B fact | D vs C fact |
| --- | --- | --- | --- | --- | --- | --- |
| English (42) | +0.46 [+0.32, +0.62] | +0.67 [+0.50, +0.82] | +0.20 [+0.02, +0.37] | +0.42 [+0.27, +0.57] | +0.58 [+0.40, +0.75] | +0.16 [-0.00, +0.33] |
| German (43) | +0.09 [-0.02, +0.23] | +0.16 [+0.02, +0.30] | +0.07 [-0.06, +0.20] | +0.09 [-0.02, +0.23] | +0.06 [-0.09, +0.21] | -0.03 [-0.16, +0.09] |
| French (40) | +0.05 [-0.07, +0.17] | +0.16 [+0.01, +0.31] | +0.11 [-0.03, +0.25] | +0.09 [-0.03, +0.21] | +0.11 [-0.02, +0.25] | +0.03 [-0.12, +0.17] |
| Italian (39) | +0.04 [-0.05, +0.13] | +0.10 [-0.05, +0.26] | +0.06 [-0.09, +0.22] | +0.01 [-0.08, +0.11] | +0.03 [-0.13, +0.20] | +0.01 [-0.15, +0.17] |

So the headline gains of C and D are mostly an English-question effect: in German, French and Italian the agent's recall advantage over single-shot is claimable only for German and French (+0.16), and nothing else in those three languages is. Per-language cells hold 39 to 43 items, which resolves only large gaps.

By question type: single-article questions (n = 147) recall@10 0.48 / 0.65 / 0.82 / 0.93; two-article questions (n = 17) 0.56 / 0.65 / 0.71 / 0.91 (too few for conclusions). Questions whose language differs from the language of the article text the drafter saw (n = 127) behave like same-language ones (n = 37) once a query is translated (0.81 vs 0.80 recall@10 for C), but single-shot and bare retrieval do worse on the cross-lingual ones (A 0.47 vs 0.57; B 0.63 vs 0.69), as expected.

## Cost and latency (test, per question, Claude Haiku 5.5, price table of 2026-10-17)

| | Cost per question (USD) | p50 latency | p95 latency |
| --- | --- | --- | --- |
| A | 0 | 0.1 s | 0.1 s |
| B | 0.0021 | 5.8 s | 9.9 s |
| C | 0.0028 | 18.3 s | 23.7 s |
| D | 0.0026 | 14.4 s | 44.6 s |

Judging with gpt-6.1-sol cost 0.83 to 1.00 USD per 179-item run, about twice the cost of the answering system itself (0.38 to 0.50 USD per run), and is not included above. Latency comes from a single Windows machine on one day, serial calls, and is replayed from cache for model calls (local retrieval time is measured live); it describes the relative cost of the pipelines, not provider speed. The agent makes 5.3 model calls per question on average (median 5), against 2 for B, and its tail is long: it hit its step limit on 35 of 179 questions.

## Agent findings and failure taxonomy

Failure categories (fixed in the spec before test, assigned by rules from the traces, test run D, 179 items): ok 153; refused an answerable question 12; wrong-language retriever (the planner searched the wrong language index and never recovered) 5; gold retrieved but not cited 5; answered a trap 4; flags (not exclusive): citation checker rejected something 13, step limit hit 35, partial gold (only one of two articles) 2.

Findings:
- The planner's main contribution is writing the query in the language of the article. That is also what the translation control C does with one call and no loop, which is why C closes most of the gap on answer quality.
- Over-refusal is largely a retrieval symptom. On dev, when the gold article was among the five articles shown, none of the three small answerers refused; they refused (44% to 89%) only when it was missing. B refuses 18% on test and the gold article is missing from its five shown articles for 55 of 164 questions (34%). D's 7% over-refusals (12 of 164) are a different thing: for 10 of the 12 the gold article was in the top 10, and for 11 of the 12 the citation checker (a Haiku call) rejected the citations, so the checker is the main source of the agent's over-refusals.
- Tool-selection accuracy was not measured separately; the traces record every tool call (search by language, `get_article`, `follow_reference`), but I did not hand-label which call a perfect agent would have made. The only automatic signal is the wrong-language-retriever category (5 of 179).
- A bug found on dev shows how fragile such a loop is: with a 600-token cap on the checker's JSON, 88 of 130 checker calls were truncated, fail-closed logic rejected every citation, and the agent refused 33 of 58 answerable questions. It looked like a modelling weakness and was a harness bug.

## Judge calibration

The judge (gpt-6.1-sol for fact recall, faithfulness and the trap leak check) was calibrated on 30 dev answers (63 fact labels and 5 trap labels), labelled blind by one reviewer who reads English only, from machine English glosses of the answers.

| Judge | Agreement with the first-pass labels (63 facts) | Kappa |
| --- | --- | --- |
| gpt-6.1-sol (used) | 0.81 | 0.61 |
| claude-sonnet-5-5 | 0.79 (56 scored) | 0.57 |
| gpt-6-luna | 0.76 | 0.52 |
| claude-haiku-5-5 | 0.73 | 0.48 |
| gemini-3.6-flash | 0.71 | 0.46 |
| gemini-3.5-flash-lite | 0.70 | 0.43 |

All judges matched the 5 trap labels. The small judges are lenient (gpt-6-luna marked 11 facts as stated that the reviewer did not, the reverse 4 times). 0.81 clears the pre-set bar of 80% only narrowly, and the margin over the next judges is a few facts, so judge-based differences of a few points should not be read as real. A recheck pass in which the reviewer saw the judge's verdict changed all 15 disputed labels to the judge's side; that agreement (sol 0.95) is contaminated by anchoring and is not used anywhere. Self-preference: gpt-6-luna and Gemini Flash-Lite differed by 0.13 to 0.17 on every answerer's answers (a leniency difference, not family bias); against Sonnet, gpt-6.1-sol scored GPT answers no higher than Gemini's, while Sonnet scored Claude answers about 0.06 to 0.07 higher relative to sol than the others. Using one judge for all answerers is a result of this: the first pass had used a different judge per family and produced an artifact (GPT answers looked best by 0.09 to 0.14; with one judge the three small answerers were indistinguishable on dev).

## Cross-family check of the benchmark

Gemini 3.6 Flash re-judged all 242 v1 items with the Phase 1 rubric, plus code-supplied corpus access: 242 of 242 approved, 0 disagreements with the Claude judge. This is weak evidence: every verdict had high confidence, and a negative control with facts swapped between items was still approved 5 times in 19. Benchmark v1 is unchanged and no v2 was made.

## What was tried and did not work or did not matter

- **Rerankers.** On dev, the Haiku, Gemini Flash-Lite and Sonnet 5.5 rerankers gave identical recall@10 (0.60) over the same 30 candidates, at 0.0008 to 0.0165 USD per query: once the gold is in the window all three put it first, so the reranker cannot be told apart on this benchmark and the benchmark is easy for LLM rerankers by construction (questions were drafted from the article text). Widening the window from 30 to 100 chunks helped (0.60 to 0.69), still inside the noise.
- **Chunking** (dev, hybrid, all-language index): article and fixed 200-word windows tie (recall@10 0.66 after reranking); paragraph chunks are 9 to 12 points worse, with overlapping intervals.
- **Index configurations** (dev): single-language indexes are much worse than the all-language index for bare retrieval; the reranker lifts every cell by 7 to 29 points; hybrid is at least as good as BM25 or embeddings alone on the all-language and same-language indexes, but on the single German and French indexes embeddings alone were ahead before reranking; all between-method differences are inside the intervals.
- **Answerer provider.** Claude Haiku, GPT-6 Luna and Gemini Flash-Lite answered the same retrieved context and could not be separated on dev (fact recall 0.55, 0.51, 0.50 with one judge). They were therefore not carried to test.
- **Agent loop vs translation.** The extra planning and checking steps did not buy measurably better answers than one translation call.

## Limitations

1. **Benchmark ceiling and drafting bias.** All 150 answerable questions were written by an LLM from the text of one article; LLM searchers find them easily once the query is in the article's language (agent recall 92 to 97%, close to the Phase 1 blind solver's 149 of 150). The benchmark separates retrievers well and strong rerankers poorly. A fine-tuned reranker in phase 3 will likely be compared on a ceiling.
2. **Review coverage.** Of 242 v1 items only 34 had a human check, by a reviewer with no legal training who reads English only; no Swiss-law expert reviewed anything. Gold facts are model-written.
3. **Judge reliability.** 81% agreement with one non-expert on 63 facts from 30 answers, using machine glosses. Fact recall and faithfulness carry judge noise of several points that the bootstrap intervals do not include. A second judge (gpt-6-luna, 76% agreement) re-scored the stored test answers with the same conclusions (see summary); it was not run for faithfulness.
4. **Small cells.** 110 originals; 164 answerable items with translations that are not independent (handled by group resampling, but they add little information). Per-language cells hold about 40 items, same-language controls 37, cross-reference questions 17.
5. **Traps (n = 15).** Counts only, no ranking. B and C each failed 5 traps (2 repealed-law, 2 unspecified-detail, 1 out-of-area) and D failed 4 (1 repealed-law, 2 unspecified-detail, 1 out-of-area). Several failures are partial answers to questions the sources only partly cover (a dog-bite question that the Code of Obligations answers in part, the registration duty of a sole proprietorship whose documents are set elsewhere), which a strict trap rule penalises, but the repealed-law failures are real mistakes (a repealed provision presented as current), so the failures mix helpfulness with errors. Dev had only 5 traps.
6. **One model family for the pipeline.** The tested systems use Claude Haiku 5.5 throughout; the provider comparison exists only on dev and only for the answerer.
7. **Reproducibility.** Provider models are not bit-deterministic and no temperature is sent (Claude 5.x and GPT-6 do not accept it), so a number is reproduced from the disk cache (`eval/cache`, git-ignored), not from scratch. The agent changed on a rerun with identical configuration: dev recall@10 0.93 to 0.97, 5 of 63 items, because failed calls are not cached and were retried. Test run D was stopped deliberately, not by a crash, after I noticed its traces were not being written to disk; no D test number existed at that point (its results are written only at the end). I fixed the trace writing (commit ea7d0eb, after the freeze commit 80d8409) and restarted it; the configuration hash is unchanged but it covers configuration, prompts and data files, not code, so the code that produced the test numbers differs from the code at freeze by that trace-writing change only. The log has two `started` lines and one `completed` for D. The rule the log enforces allows a rerun of a configuration that has not completed.
8. **Latency** measured on one machine, serially, on one day.
9. **Test split used once per configuration**, four configurations in total; dev was used to choose them, so test numbers are not selected on test. The choice of translate, reranker and agent designs was made on dev with 40 originals; some of the gaps between B, C, D may also reflect dev-driven design (for example the agent's checker settings).

## Reproduction

```
uv run pytest
uv run python -m fedlex_bench.eval report 2026-10-18_test_78df4b6f59 2026-10-18_test_be2a67022b 2026-10-18_test_89c5213cec 2026-10-18_test_89a1a1a0ad
uv run python -m fedlex_bench.eval compare 2026-10-18_test_89a1a1a0ad 2026-10-18_test_89c5213cec
```

Run folders (`config.json`, `per_item.jsonl`, `metrics.json`; the agent run also has `traces/traces.jsonl`) are under `eval/results/`. A rerun from scratch needs the three API keys and the cache; the frozen configuration hashes are in `eval/frozen_configs.json` and every test run is in `eval/test_runs.log`.

## Errata (2026-10-28): embedder padding bug

**What was wrong.** `OnnxE5Embedder` embedded texts in batches, and the tokenizer padded every text of a batch to the longest one. The embedder took each text's length from the padded ids, so its attention mask was 1 over the padding and the pad tokens were averaged into the mean-pooled vector. A text embedded alone (every query at search time) was correct; a text embedded in a batch was not (cosine to its correct vector 0.93 to 1.00 on a sample, lowest for short texts that shared a batch with long ones). All passage vectors of the index came from batches (chunks of 1,024 sorted by length, then batches of 32). It was found in Phase 3 when a torch validation of the same untrained model gave recall@10 0.78 where the ONNX pipeline gave 0.44 on the same questions. Fix: commit 7c5bdbe (the real length is the mask sum; the corrected vectors have their own cache id; `OnnxE5Embedder(mask_padding=False)` reproduces the old vectors; a regression test compares a short text in a batch with the same text alone). Details and the diagnosis: `docs/phase3_progress.md`.

**What was re-measured.**

| | Phase 2 (buggy embedder) | Corrected embedder |
| --- | --- | --- |
| Test, system A (hybrid, all-language, no model), recall@10, n = 164 | 0.49 [0.40, 0.58] | **0.55 [0.46, 0.63]** |
| recall@1 / MRR | 0.22 / 0.32 | 0.25 / 0.36 |
| English / German / French / Italian recall@10 | 0.14 / 0.58 / 0.65 / 0.60 | 0.24 [0.12, 0.37] / 0.60 / 0.72 / 0.63 |
| Originals only (n = 110) recall@10 | 0.51 | 0.58 |
| Paired difference corrected minus old, recall@10 | | +0.055 [+0.009, +0.104] (MRR +0.037 [+0.010, +0.064]) |
| Dev, hybrid all-language, recall@10 (n = 58 answerable) | 0.46 | 0.53 [0.36, 0.67] |
| Dev, embedding only, all-language | 0.40 | 0.53 |

The corrected test run is run `2026-10-28_test_75cfac7c94` (frozen as `A_padfix`, hash 75cfac7c94, which differs from A's hash because the configuration now records the embedder id). It is retrieval only: no model call, no tuning, and no cost. It is the **second use of the test split for system A**; the hypothesis was written before the run (`docs/phase3_progress.md`): overall 0.50 to 0.60 (observed 0.55), English 0.25 to 0.40 (observed 0.24, just below the range). The overall gain is claimable but small (the interval's lower bound is +0.009); per language, the intervals overlap.

**What is not known.** B, C and D used the same retrieval stage with the buggy embedder (B and D over the all-language hybrid index, C over the per-language indexes), so their candidate lists were weaker than the model allows. Their reranker, answerer and judge numbers are what they are, but their recall, fact recall and the gaps between systems (C over B, D over B, D over C) were measured on degraded candidates, and might change in either direction (a better first stage helps B most, which narrows the gap that C's query translation closes; the English gap of C over A will shrink from 0.57 at least in part). These claims in the report should be read as conditional on the buggy first stage until the frozen B, C and D are rerun with the corrected embedder. Decision on that rerun: open (see `docs/phase3_progress.md`).

**Process.** The dev retrieval runs with the corrected embedder are `dev_ret_*_padfix`. The test log has one extra `started` and `completed` pair for the new frozen configuration; the Phase 2 runs are untouched.
