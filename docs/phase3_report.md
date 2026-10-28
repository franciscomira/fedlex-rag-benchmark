# Phase 3 report: fine-tuning a small multilingual embedding model for Swiss-law retrieval

Date: 2026-10-28. Benchmark: `benchmark/v1/` (tag `benchmark-v1`, unchanged). Working log with every number and its context: [phase3_progress.md](phase3_progress.md). Data: [../data/finetune/README.md](../data/finetune/README.md). Phase 2 report with its errata: [phase2_report.md](phase2_report.md). Evaluation rules: [eval_spec.md](eval_spec.md).

## Summary

- **Question.** Can a small fine-tuned multilingual model close the gap that LLM query translation closes in cross-lingual retrieval over Swiss federal law, without an LLM call at query time? Phase 2 had shown that the first stage, not the reranker, was the bottleneck, so the model tuned is the bi-encoder behind system A (`intfloat/multilingual-e5-small`, 118M parameters), not a reranker.
- **Answer on test (164 answerable items, 95% group bootstrap).** The tuned embedder alone (T1, no LLM, 0.1 s, 0 USD) reaches recall@10 **0.81 [0.71, 0.90]** against 0.55 [0.46, 0.63] for the untuned hybrid (embedding plus BM25) with the corrected embedder (+0.26 [+0.17, +0.35]; the comparison changes the weights and drops BM25 at once, but on dev the untuned model scored the same 0.53 as embedding-only and as hybrid, so the gain is the tuning; on the 142 test items whose gold article is not touched by a pool article contained in it, the gain is +0.30 [+0.20, +0.40] in group means); English questions go from 0.24 to **0.80**. The LLM-translation pipeline (C', 8 s, 0.0024 USD per question) reaches 0.87 [0.79, 0.94]; the difference to T1 on recall@10 (-0.06 [-0.15, +0.02]) is **not claimable**. On ordering the gap is clear: MRR 0.63 against 0.84 (-0.21 [-0.31, -0.11]), recall@1 0.48 against 0.76.
- **With a cheap reranker (T2: tuned embedder plus the Haiku reranker)** recall@10 is 0.89 [0.80, 0.97], MRR 0.87, at 1.6 s and 0.0016 USD per question: not distinguishable from C' (+0.015 [-0.05, +0.07] recall@10, +0.03 [-0.04, +0.09] MRR) and 5 times faster, a third cheaper.
- **Retrieval only.** No answer was generated or judged in Phase 3. Nothing here shows that final answers improve.
- **Training is cheap.** 11,504 synthetic questions (0.87 USD to generate), 3 epochs, under 15 minutes on a free Kaggle T4 (0.4 s per step). A quarter of the data gives most of the dev gain; hard negatives did not help.
- **A bug in the Phase 2 embedder** (padding averaged into batched vectors) was found along the way and corrected; Phase 2's retrieval numbers were understated. See Errata in the Phase 2 report and the section below.
- Spend in Phase 3: 2.16 USD (ledger of 2026-10-28; data generation 0.87, audits 0.08, dry and test runs about 1.2); the month stands at 13.08 of the 25 USD cap.

## What was done

| Step | What | Where |
| --- | --- | --- |
| Pool and leakage | 3,663 retrieval-eligible articles minus 172 dev/test gold, 288 neighbours, 3 twins (character 5-gram Jaccard at least 0.5): **3,201** training articles; tests fail on overlap with gold, neighbours and symmetric twins (see limitation 10) | `finetune/exclusions.py`, `tests/test_finetune_leakage.py` |
| Synthetic questions | 4 per article (2 English, 2 in two of DE/FR/IT; the language not asked for is the language shown to the generator), six styles, Claude Haiku 5.5, 12,800 questions for 3,200 articles | `finetune/synth.py` |
| Split | by article: 11,504 train and 1,296 validation questions (324 articles), one passage language per pair | `finetune/dataset.py` |
| Audit | a model from another family (gpt-6-luna) judged 330 pairs | `finetune/audit.py` |
| Training | plain PyTorch, in-batch ranking loss (scale 20), AdamW 2e-5, warmup 10%, batch 64, 3 epochs, fp16, Kaggle T4 | `finetune/train.py`, `notebooks/finetune_kaggle.ipynb` |
| Evaluation | the Phase 2 harness with the tuned model as embedder | `eval --embedder` |

## Results

### Training (validation: 1,296 synthetic questions ranked among all 3,201 articles)

| Epoch | Train loss | Val loss | Recall@10 | English | German |
| --- | --- | --- | --- | --- | --- |
| 0 (untrained) | | 3.12 | 0.781 | 0.670 | 0.911 |
| 1 | 0.96 | 0.300 | 0.925 | 0.920 | 0.915 |
| 2 | 0.28 | 0.257 | 0.934 | 0.927 | 0.938 |
| 3 | 0.22 | 0.246 | 0.9375 | 0.927 | 0.946 |

Validation loss fell at every epoch (3.12, 0.300, 0.257, 0.246) and ended close to the epoch-average training loss (0.22). The two are not strictly comparable (training loss is averaged over the epoch with dropout on, and starts at 0.96), but validation loss never rose, so there is no sign of overfitting. Validation uses questions from the same generator as training and is not reported as a result; it was used for choices between runs.

### Dev (58 answerable items; intervals about 30 points wide; embedding only, all-language index)

| | Recall@10 | Recall@1 | MRR | English recall@10 |
| --- | --- | --- | --- | --- |
| Base model, corrected embedder | 0.53 [0.35, 0.69] | 0.22 | 0.33 | 0.43 |
| Tuned, 25% of the training articles | 0.76 [0.59, 0.89] | 0.38 | 0.53 | 0.80 |
| Tuned, 50% | 0.76 [0.59, 0.89] | 0.39 | 0.54 | 0.80 |
| **Tuned, 100% (run 1, in-batch negatives)** | **0.79 [0.63, 0.92]** | 0.43 | 0.59 | 0.80 |
| Tuned, 100% plus hard negatives (run 2) | 0.81 [0.65, 0.93] | 0.39 | 0.58 | 0.80 |

Paired differences on dev: run 1 over base +0.27 [+0.15, +0.41] recall@10 (claimable); run 2 over run 1 +0.017 [0.000, +0.061] (not claimable, and 1.7 times the training time per step), so run 1 is the main configuration; 25% over base +0.23 [+0.12, +0.36]; 100% over 50% +0.034 [0.000, +0.093] recall@10, +0.053 [+0.011, +0.096] MRR. The smaller runs also take fewer steps (135, 270, 540); one seed per size.

### Test (each frozen configuration run once; 164 answerable items, 110 originals; retrieval metrics only)

| System | Recall@1 | Recall@10 | MRR | English / German / French / Italian recall@10 | Cost / q | Latency p50 / p95 |
| --- | --- | --- | --- | --- | --- | --- |
| A, Phase 2 (buggy embedder) | 0.22 | 0.49 [0.40, 0.58] | 0.32 | 0.14 / 0.58 / 0.65 / 0.60 | 0 | 0.1 / 0.1 s |
| A, corrected embedder | 0.25 [0.18, 0.33] | 0.55 [0.46, 0.63] | 0.36 | 0.24 / 0.60 / 0.72 / 0.63 | 0 | 0.1 / 0.1 s |
| B', corrected (hybrid + Haiku reranker) | 0.65 [0.56, 0.74] | 0.74 [0.65, 0.82] | 0.72 | 0.52 / 0.77 / 0.86 / 0.81 | 0.0016 | 1.4 s p50 |
| **T1** tuned embedder, no LLM | 0.48 [0.38, 0.59] | **0.81 [0.71, 0.90]** | 0.63 [0.53, 0.73] | **0.80** / 0.79 / 0.81 / 0.83 | **0** | **0.1 / 0.1 s** |
| **T2** tuned embedder + Haiku reranker | **0.79 [0.70, 0.88]** | **0.89 [0.80, 0.97]** | **0.87 [0.78, 0.95]** | 0.89 / 0.87 / 0.88 / 0.91 | 0.0016 | 1.6 / 4.5 s |
| C', corrected (translation + hybrid + Haiku reranker) | 0.76 [0.67, 0.84] | 0.87 [0.79, 0.94] | 0.84 [0.76, 0.91] | 0.86 / 0.90 / 0.84 / 0.90 | 0.0024 | 8.0 / 10.5 s |

Per-language cells hold 39 to 43 items, so their intervals are about 25 points wide (T1 English [0.68, 0.90]); read them as directions. Cost and latency come from live calls on one Windows machine on one day; model-call latency of cached calls is replayed.

Paired differences, 95% interval, recall@10 and MRR:

| Comparison | Recall@10 | MRR | Claimable |
| --- | --- | --- | --- |
| T1 minus A corrected | +0.262 [+0.174, +0.351] | +0.271 [+0.182, +0.361] | yes |
| T2 minus T1 | +0.079 [+0.029, +0.139] | +0.238 [+0.158, +0.320] | yes |
| T2 minus B' | +0.149 [+0.077, +0.225] | +0.148 [+0.074, +0.225] | yes |
| T2 minus C' | +0.015 [-0.052, +0.074] | +0.032 [-0.036, +0.091] | no |
| T1 minus C' | -0.064 [-0.149, +0.016] | -0.207 [-0.305, -0.106] | recall@10 no, MRR yes (C' better) |

All five hypotheses written before the test runs held (`phase3_progress.md`); none failed. The test split was used once for each of T1, T2, C' and B', and twice for system A (the corrected A, an errata run). B' was frozen after the T1, T2 and C' results were known (an additional errata run for Phase 2, approved afterwards); its configuration is Phase 2's B minus the answerer with the corrected embedder, so no choice was open, but the T2 minus B' comparison was not planned before the other three results.

### Deployment: CPU latency and the ONNX export

The tuned model (run 1, epoch 3) exported to ONNX with `finetune/export_onnx.py` (opset 17, dynamic batch and sequence length; the same inputs and pooling as the Phase 2 embedder). Latency of embedding one question on this machine (AMD Ryzen 5 5500U, 6 cores / 12 threads, Windows 11, no other load; 200 validation questions after 10 warm-up calls; tokenisation, model and pooling; `eval/results/latency_r1_inbatch.json`):

| Variant | File size | 1 thread p50 / p95 | 12 threads p50 / p95 |
| --- | --- | --- | --- |
| torch fp32 | 449 MB (safetensors) | 44.4 / 57.6 ms | 32.9 / 38.7 ms |
| ONNX fp32 | 470 MB | 22.4 / 31.1 ms | 13.9 / 22.3 ms |
| ONNX int8 (dynamic, per channel) | **118 MB** | **19.9 / 32.9 ms** | **11.6 / 17.4 ms** |

Searching 11,000 stored vectors is a 1.1 ms matrix product, so a query's retrieval cost is the embedding time plus about 1 ms (the harness' end-to-end 0.1 s p50 for T1 includes its own overhead and the Python retrieval code). The int8 model is four times smaller and 10 to 17% faster than ONNX fp32 on this CPU; speed is not the reason to quantise here, size and memory are (a 118 MB model fits a small Cloud Run instance comfortably).

Accuracy of the exports (dev, embedding only, all-language index, 58 items; runs `dev_ret_embedding_all_lang_article_r1_onnx_*`):

| | Recall@10 | Recall@1 | MRR | English / German / French / Italian recall@10 | Cosine to torch |
| --- | --- | --- | --- | --- | --- |
| torch fp32 | 0.79 [0.63, 0.92] | 0.431 | 0.589 | 0.80 / 0.88 / 0.70 / 0.81 | 1 |
| ONNX fp32 | 0.79 [0.63, 0.92] | 0.431 | 0.589 | identical | 1.0000 (min) |
| ONNX int8 | 0.78 [0.61, 0.91] | 0.440 | 0.583 | 0.73 / 0.88 / 0.70 / 0.81 | 0.981 mean on 100 queries and 100 passages (min 0.92) |

The fp32 export reproduces the torch results exactly. The int8 model is not worse in any claimable way (recall@10 -0.017 [-0.077, +0.041], MRR -0.007 [-0.085, +0.073]); English recall@10 fell from 0.80 to 0.73 (one or two question groups of 15), which dev cannot resolve. The default (per-tensor) int8 quantisation was clearly lossier (mean cosine 0.945) and was not used. The int8 model has not been run on the test split; Phase 4 should check it there before serving, as a new frozen configuration.

## The embedder bug and what it changed

`OnnxE5Embedder` took each text's length from a batch that the tokenizer had padded to the longest text, so the attention mask was 1 over the padding and pad tokens were averaged into the vector. A text embedded alone (every query at search time) was right; a text in a batch (every passage of the index, and my mining queries) was not (cosine to the correct vector 0.93 to 1.00). It surfaced because the untrained model scored recall@10 0.78 in a torch validation and 0.44 in the ONNX pipeline on the same questions. Effects:
- Phase 2's retrieval numbers were too low: corrected system A on test recall@10 0.55 against 0.49 (+0.055 [+0.009, +0.104]), English 0.24 against 0.14; B' 0.74 against B 0.65; C' 0.87 against C 0.81 (unpaired). Phase 2's report carries an erratum. Fact recall, faithfulness and trap results were not re-measured; D was not rerun.
- My first hard-negative mining, the "hub articles" finding, the "keyword questions are hardest" finding and the first audit were artifacts; they were rebuilt.
- Fixed in commit 7c5bdbe with a regression test; the old behaviour is available as `OnnxE5Embedder(mask_padding=False)`; eval configurations (also the ADK agent's, after review) now record the embedder id.

## Limitations

1. **The benchmark rewards what the tuned model learns.** Every benchmark question was written by an LLM from the text of one article, and so were the training questions (by the same model family, with six deliberately different styles and a different prompt). The risk is that part of the gain is learning the drafting style, not cross-lingual retrieval. The held-out synthetic validation shows the same shape as dev, which does not rule it out. Real user questions and an expert review of the gold have not been tried.
2. **Dev and test are small.** 58 and 164 answerable items, about 40 and 110 independent question groups. Per-language cells hold 15 (dev) and 39 to 43 (test) items. Only large differences are claimable.
3. **Retrieval only.** No answer, citation, faithfulness or refusal measurement was made in Phase 3. T2's better candidates should help answers, but that was not tested. The Phase 2 answer-level results (fact recall, faithfulness, traps) were obtained with the buggy first stage and not repeated.
4. **No human reviewed the synthetic data.** Nobody on the project reads German, French and Italian well enough; a second-family judge audited 100 sampled questions (80% fully answered by their article, 20% partly, 0 not; of the chosen hard negatives 1% fully and 24% partly answer the question, against 4% and 34% without the filter), with a lenient judge that was not calibrated for this task. No Swiss-law expert reviewed anything.
5. **One seed, one run per configuration.** Training seed variation was not measured; the ablation confounds data size with the number of steps.
6. **Hybrid versus embedding only.** After tuning, embedding alone (dev 0.79) beat the hybrid with BM25 (0.72) on dev: recall@10 +0.069 [-0.013, +0.173] (not claimable), MRR +0.154 [+0.052, +0.259] (claimable); not tested further, and not run on test. T1 and T2 use embedding only.
7. **Latency and cost** come from one machine on one day; cost comes from the price table of 2026-10-17.
8. **Dev reuse.** Dev was used for several decisions (run 1 against base, run 2 against run 1, hybrid against embedding only, which runs to freeze); no choice was made on test.
9. **Provider reproducibility and provenance.** Reranker and translation responses are reproducible from the disk cache, not from scratch. The tuned checkpoint is not in git (about 450 MB); T1's weights digest (4265091842c5) is part of its frozen configuration. Run 1's `config.json` records `git_sha: unknown` (runs 2 and 3 record the kit's sha 45a81b62), so run 1 cannot be tied to a code version beyond its data hashes and settings, and the tokenizer files are not hashed.
10. **Twin exclusion is symmetric.** The pool excluded articles with a character 5-gram Jaccard of at least 0.5 to a gold article. A short pool article contained in a long gold article has a low Jaccard and is not excluded: 6 pool articles are at least 80% contained in a gold article, 37 at least 60% (touching 18 gold articles), 84 at least 50%. This was found in review and not fixed (the pool would change and the model would have to be retrained). Sensitivity: on the 22 test items whose gold article is touched at the 60% level T1's recall@10 is 0.98 (corrected A: 0.68), on the other 142 it is 0.78 (A: 0.52), and the gain of T1 over A on the untouched items is +0.30 [+0.20, +0.40] in group means; so the headline gain does not depend on those items, but part of T1's absolute level on the 22 touched items may.
11. **Synthetic questions: the planned fixed roles were never applied.** The third-party style was meant to name a role per question; the code for that was dead (found in review and removed), so third-party questions got roles invented by the generator (often a client or a business partner).

## Reproduction

```
uv run pytest
uv run python -m fedlex_bench.finetune.exclusions && uv run python -m fedlex_bench.finetune.dataset
# training: notebooks/README.md (Kaggle T4; settings 2e-5, 3 epochs, batch 64, seed 20261028)
uv run python -m fedlex_bench.eval report 2026-10-28_test_aec4726ef0 2026-10-28_test_ae60b9e8c1 2026-10-28_test_033e9b80b2 2026-10-28_test_b7846648c0 2026-10-28_test_75cfac7c94
uv run python -m fedlex_bench.eval compare 2026-10-28_test_aec4726ef0 2026-10-28_test_75cfac7c94
```

Frozen configuration hashes: T1 aec4726ef0, T2 ae60b9e8c1, C' 033e9b80b2, B' b7846648c0, A corrected 75cfac7c94 (`eval/frozen_configs.json`); every test run is in `eval/test_runs.log`. Training logs and configurations (with data hashes): `models/<run>/log.jsonl` and `config.json` (not in git).
