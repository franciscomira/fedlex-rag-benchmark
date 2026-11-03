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
