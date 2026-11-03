# Phase 4 progress log

Working log for the ship phase (serving pipeline, demo, MCP server, results page). Numbers here are copied from `eval/results/<run>/metrics.json`; every test run is in `eval/test_runs.log`.

## 2026-11-03, plan and scope

Approved by Francisco. Serving configuration: tuned int8 ONNX embedder (embedding only, no BM25), Haiku reranker over the first 60 of 150 candidates, Haiku cited answer from the top 5 articles, ids validated by code. The serving code (`src/fedlex_bench/serve/pipeline.py`) builds the harness' own `PipelineSystem`, so the thing served is the thing measured. The int8 file served is byte-identical to the one evaluated (sha256 prefix 9d9e6cd1fdd2, checked against the published Hugging Face file).

Three new frozen configurations, each to be run once on test:
- **P1**: T1 with the int8 embedder, retrieval only (no model call). `--method embedding --config all_lang --chunking article --embedder models/r1_inbatch/onnx/model_int8.onnx --retrieve-k 60 --rerank-k 30`.
- **P2**: the serving pipeline: P1's embedder, `--rerank claude-haiku-5-5 --retrieve-k 150 --rerank-k 60 --answer claude-haiku-5-5`, judge gpt-6.1-sol.
- **P3**: Phase 2's C with the corrected embedder and the answer step: `--method hybrid --translate claude-haiku-5-5 --rerank claude-haiku-5-5 --answer claude-haiku-5-5 --retrieve-k 60 --rerank-k 60`, same judge. It gives an answer-level comparison that is not confounded by the Phase 2 embedder bug.

System D (the ADK agent) is not rerun; it stays marked as not re-measured.

## 2026-11-03, dev dry runs (before freezing; 58 answerable items, intervals about 25 to 30 points wide)

| Dev | R@1 | R@10 | MRR | Fact recall | Faithfulness | Over-refusal | Traps | Cost / q (USD) | p50 / p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P2 (int8 + Haiku rerank + answer) | 0.83 [0.67, 0.94] | 0.88 [0.73, 0.98] | 0.88 | 0.68 [0.56, 0.80] | 0.80 [0.75, 0.86] | 0.03 | 5/5 | 0.0026 | 5.4 / 11.2 s |
| P3 (translation + corrected hybrid + rerank + answer) | 0.75 [0.58, 0.88] | 0.84 [0.69, 0.96] | 0.82 | 0.69 [0.59, 0.81] | 0.81 [0.74, 0.88] | 0.02 | 4/5 | 0.0032 | 11.9 / 19.2 s |
| B, Phase 2 (buggy embedder) | 0.62 | 0.66 | 0.67 | 0.56 [0.43, 0.69] | 0.83 | 0.14 | 5/5 | 0.0021 | 5.9 / 11.4 s |
| C, Phase 2 (buggy embedder) | 0.75 | 0.79 | 0.80 | 0.66 [0.52, 0.80] | 0.86 | 0.09 | 4/5 | 0.0029 | 11.4 / 18.0 s |

P2 minus P3 on dev (paired): recall@10 +0.034 [-0.041, +0.122], MRR +0.055 [-0.027, +0.148], fact recall -0.011 [-0.125, +0.105], faithfulness -0.013 [-0.059, +0.032]: none claimable; cost lower (claimable). P2 had 3 invalid ids over 58 answers (0.05 per answer), all dropped by code. Judge cost per dev run: 0.43 USD. Month spend after both dev runs: 1.55 USD of 25.

## 2026-11-03, hypotheses for the three test runs (fixed before any of them is run)

Reference values on test: T1 (fp32) recall@10 0.81, English 0.80; T2 recall@10 0.89, recall@1 0.79; Phase 2 fact recall B 0.61, C 0.76 (buggy embedder); Phase 2 faithfulness about 0.85; traps 10, 10, 11 of 15.

1. **P1** (int8, retrieval only): recall@10 in 0.77 to 0.84; the paired difference to T1 (fp32) is within 0.04 and not claimable; English recall@10 at least 0.70. *Stop rule:* recall@10 below 0.75, or a claimable loss against T1, and I ask before going on.
2. **P2** (full serving pipeline): recall@10 in 0.84 to 0.93, recall@1 in 0.70 to 0.85; fact recall in 0.62 to 0.78; faithfulness in 0.78 to 0.88; over-refusal at most 0.08; traps 9 to 13 of 15; invalid ids at most 0.1 per answer; cost per question 0.0020 to 0.0032 USD; p50 latency 3 to 8 s. *Stop rule:* recall@10 below 0.80 and I ask before using the number anywhere.
3. **P3** (translation pipeline with the answer step): recall@10 in 0.82 to 0.91; fact recall in 0.64 to 0.80; faithfulness in 0.78 to 0.88; traps 9 to 13 of 15; cost 0.0026 to 0.0038 USD; p50 8 to 14 s.
4. **P2 against P3**: no claimable difference in recall@10, fact recall or faithfulness (absolute differences below 0.08); P2 cheaper and faster (claimable). The claim to make if it holds: "the tuned embedder pipeline gives the same answer quality as LLM translation at about a half to a third of the latency", with the intervals, not "better".
5. **P2 against Phase 2's B** (paired, same judge, B had the buggy first stage; informative only): fact recall higher, claimable. P2 against Phase 2's C: not claimable.
6. **Traps:** counts only (k of 15), no ranking claim.

Budget estimate for the three test runs: about 1.7 USD each for P2 and P3 (answering 0.5 to 0.6, judge about 1.2 for 179 items, scaled from the dev runs), 0 for P1; month stands at 1.55 USD.

## 2026-11-03, freeze

Frozen in `eval/frozen_configs.json` before any test number: **P1 35611dd1aa**, **P2 d6a30cf69a**, **P3 716222848a**. The P2 and P3 hashes equal the hashes of their dev dry runs (`dev_P2`, `dev_P3`), so the configurations run on test are the ones that were tried on dev. Each is run once on test.

## 2026-11-03, P1 test result (int8 embedder, retrieval only; run `2026-11-03_test_35611dd1aa`, once)

n = 164 answerable items, 95% group bootstrap. Next to T1 (fp32, `2026-10-28_test_aec4726ef0`):

| | Recall@1 | Recall@10 | MRR | English recall@10 | p50 |
| --- | --- | --- | --- | --- | --- |
| P1 int8 | 0.44 [0.34, 0.55] | **0.79 [0.69, 0.88]** | 0.58 [0.48, 0.68] | 0.80 | 0.05 s |
| T1 fp32 | 0.48 [0.38, 0.59] | 0.81 [0.71, 0.90] | 0.63 [0.53, 0.73] | 0.80 | 0.07 s |

Paired, P1 minus T1: recall@1 -0.040 [-0.099, +0.013], recall@10 -0.021 [-0.060, +0.012] (not claimable), **MRR -0.055 [-0.097, -0.018] (claimable loss)**.

Against hypothesis 1: recall@10 in 0.77 to 0.84 held (0.79), the recall@10 difference to T1 within 0.04 and not claimable held, English at least 0.70 held (0.80). What I did not anticipate: the MRR loss is claimable (about 5 points), where dev (58 items) showed -0.007 [-0.085, +0.073], which could not resolve it. P1's absolute numbers are in line with its dev numbers (recall@10 0.78, MRR 0.583 on dev; 0.79 and 0.58 on test); T1's test numbers were higher than its dev numbers (0.79 and 0.589 on dev). **The pre-set stop rule ("a claimable loss against T1") applies, so P2 and P3 are not run until Francisco decides.**

## 2026-11-03, P2 and P3 test results (each run once; 164 answerable items, 15 traps; judge gpt-6.1-sol; 95% group bootstrap)

Runs `2026-11-03_test_d6a30cf69a` (P2) and `2026-11-03_test_716222848a` (P3). Zero parse failures and zero judge failures in both.

| Test | R@1 | R@10 | MRR | Fact recall | Faithfulness | Cit. precision | Cit. recall | Over-refusal | Traps | Cost / q | p50 / p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **P2** serving pipeline (int8 + Haiku rerank + answer) | 0.76 [0.67, 0.85] | 0.86 [0.78, 0.94] | 0.84 [0.75, 0.92] | 0.80 [0.71, 0.89] | 0.84 [0.80, 0.87] | 0.58 [0.50, 0.65] | 0.88 [0.80, 0.95] | 0.02 [0.00, 0.05] | 9/15 | 0.0024 | 5.4 / 10.6 s |
| **P3** translation pipeline + answer | 0.76 [0.67, 0.84] | 0.87 [0.79, 0.94] | 0.84 [0.76, 0.91] | 0.82 [0.75, 0.89] | 0.85 [0.83, 0.88] | 0.57 [0.49, 0.64] | 0.88 [0.80, 0.94] | 0.01 [0.00, 0.03] | 11/15 | 0.0030 | 11.9 / 17.5 s |

English recall@10 0.87 (P2) and 0.86 (P3); English fact recall 0.83 and 0.81. Invalid ids dropped by code: 0.006 per answer (P2), 0 (P3). Judge cost 1.03 and 1.07 USD. Month spend after all Phase 4 test runs: 4.16 USD of 25.

Paired differences (first minus second; recall@10, fact recall, faithfulness; 95% interval):
- **P2 minus P3**: recall@10 -0.009 [-0.084, +0.060], MRR +0.002 [-0.076, +0.075], fact recall -0.021 [-0.099, +0.051], faithfulness -0.018 [-0.047, +0.012]: none claimable. Cost per question lower by 0.0006 USD and latency lower by a paired mean of 6.1 s [5.6, 6.5] (median of the paired differences 6.2 s), both claimable. P2 and P3 ran at the same time (`eval/test_runs.log`), so their live latencies share the machine and the network; model-call latency is real, not replayed, in both.
- **P2 minus Phase 2's B** (buggy first stage, same judge): recall@10 +0.216 [+0.121, +0.314], fact recall **+0.196 [+0.103, +0.293]** (claimable); faithfulness -0.003 [-0.033, +0.027].
- **P2 minus Phase 2's C** (buggy first stage): recall@10 +0.052 [-0.037, +0.140], fact recall +0.040 [-0.051, +0.130] (not claimable).

Against the hypotheses written before the runs:
- **Held:** P2 recall@10 (0.86, expected 0.84 to 0.93), recall@1 (0.76), faithfulness, over-refusal (0.02 against at most 0.08), invalid ids, cost, latency; traps 9 of 15 (expected 9 to 13, at the bottom edge; Phase 2: 10, 10, 11 of 15); P3 recall@10 (0.87), faithfulness, traps (11), cost, latency; hypothesis 4 (P2 and P3 not distinguishable on quality, P2 cheaper and about twice as fast); hypothesis 5 (P2 over B claimable, P2 over C not).
- **Missed, upwards:** fact recall was above the expected range for both: P2 0.80 (expected 0.62 to 0.78), P3 0.82 (expected 0.64 to 0.80). Dev (0.68 and 0.69) had understated it; both dev intervals contained the test values. Nothing failed in an unfavourable direction.
- **Stop rule:** P2 recall@10 is 0.86, above the 0.80 threshold; no stop.

What this means, with its limits: the tuned-embedder pipeline and the LLM-translation pipeline show no resolved difference in answer quality (fact recall -0.021, interval about +-0.075), at 20% lower cost and about half the median latency. Phase 2's fact recall for B was 0.61 on a buggy first stage and is 0.80 for P2; the answerer and judge are the same, but the retrieval stage changed in four ways at once (fine-tuning, the padding fix, no BM25, int8), so this does not say which change helped, and it is not a claim about any single one. Traps: 9 of 15 for P2, 11 for P3 (counts, no ranking). Citation precision stays near 0.58 as in Phase 2: models cite every article they used, and the gold is one article. The benchmark ceiling and drafting-bias limitations from Phase 2 and 3 still apply (questions written by an LLM from the article text; 110 independent question groups).

## 2026-11-03, owner decision on the P1 stop rule, review and fixes

- **Decision on the stop rule.** After P1 (int8, retrieval only) showed a claimable MRR loss against T1 (-0.055 [-0.097, -0.018]), the owner was asked whether to keep int8. Francisco asked for my recommendation (keep int8: a reranker on top covers the ordering loss (shown afterwards by P2's MRR of 0.84 against 0.87 for fp32 T2, overlapping), 118 MB against 470 MB, and P2 measures the deployed pipeline) and then agreed. Only then were P2 and P3 run. The demo's retrieval-only page shows the int8 embedder's own ordering with no reranker, where the MRR loss is not covered by a reranker; the page's "how good is it" text says rankings are imperfect.
- **Published file.** The int8 file in the Hub repository at revision 786ca378 has the same SHA-256 as the local file that was measured (9d9e6cd1fdd27597c368..., compared by the Hub's file metadata; nothing was downloaded). The serving code now refuses to start with a model file whose digest differs.
- **Review of this log** (eval-reviewer, read-only): all table numbers matched the harness output, the hypotheses and the freeze were committed before the P1 test run (hypotheses 16:57, freeze 16:58, first test run 17:03), each configuration ran once on a clean tree, no eval code changed between dev and test. Fixes made after the review: the stop-rule decision recorded here; the latency figure relabelled (paired mean) with the note that P2 and P3 ran concurrently; the "same answers" and "mostly retrieval" wording; the P2 median latency 5.4 s; the model card text about int8 (now says it was run on test and lost 0.055 MRR); the results page now states that the headline retrieval numbers are for the fp32 T1 and that the served int8 file ordered 0.055 MRR worse, and marks the comparisons with Phase 2's B and C as exploratory with the buggy-embedder label. The Hugging Face model card still has the old sentence; re-uploading it needs Francisco's OK.
