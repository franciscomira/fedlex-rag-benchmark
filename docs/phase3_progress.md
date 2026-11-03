# Phase 3 progress log (fine-tuning, learning mode)

## 2026-10-28, Step 0: housekeeping

- CLAUDE.md status line updated to "Phase 3: fine-tuning (learning mode)".
- Hardware (this machine): AMD Ryzen 5 5500U (6 cores / 12 threads), 15.3 GB RAM, 260 GB free disk, AMD Radeon integrated graphics (512 MB). **No NVIDIA GPU, no CUDA** (`nvidia-smi` not found). Python 3.13.3, uv 0.13.1. torch and sentence-transformers are not installed; the repo runs embeddings through ONNX Runtime only.
- Proposal where to train:
  - Local CPU: data building, hard-negative mining, all evaluation, ONNX export and latency tests (inference of e5-small is fast on this CPU).
  - Training: free **Kaggle** notebook GPU (T4 x2 or P100, about 30 GPU hours per week, no payment details needed) as the main option, **Colab** free T4 as backup, in `notebooks/`. A rough estimate (not measured) puts CPU training of a 118M-parameter model on about 10k+ query/article pairs at several hours, above the 2-hour local limit set in the kickoff, so local CPU training is only for a smoke test of a few steps.
  - No paid compute. A new `finetune` dependency group (torch CPU, sentence-transformers) will be added to `pyproject.toml` when needed, pinned.
- Budget: 10.9 of 25 USD spent. Synthetic-query estimate (to be confirmed before any paid call, Step 2): about 3,000 training articles, one call per article producing about 4 questions, roughly 1,000 input and 400 output tokens per call with a Claude Haiku 5.5 class model ($0.10 / $0.50 per million): about 0.3 + 0.6 = **under 1 USD**, far below the 5 USD confirmation line. Exact figure comes from the real article count after exclusions.

## 2026-10-28, Step 2: training pool and synthetic queries

- Exclusions (`src/fedlex_bench/finetune/exclusions.py`, test `tests/test_finetune_leakage.py`): 3,663 retrieval-eligible article ids; removed 172 dev/test gold, 288 neighbours (previous and next eligible article of the same act), 3 twins of gold at character 5-gram Jaccard >= 0.5 (`220:257c`, `220:808`, `281.1:108`; the benchmark itself used 0.7 and found none). Training pool: **3,201 articles**. The test fails on any overlap; a poisoned pool made the gold and twin checks fail.
- Synthetic queries (`src/fedlex_bench/finetune/synth.py`, model `claude-haiku-5-5`, prompt hash in the run, seed 20261028): 4 questions per article, 2 English plus two of DE/FR/IT; the article is shown in the language left out. Six styles, four distinct per article (a planned fixed role for `third_party` was never applied: see the correction below).
- Pilots: 2 x 20 articles (0.005 USD each). The first showed a repeated "my business partner" persona and invented details; an instruction to use a role named in the slot and a no-new-facts rule were added before the full run (the code that was to name the roles never took effect; found in review, 2026-10-28).
- Full run: 3,201 articles, **12,800 questions** from 3,200 articles (EN 6,400, DE 2,130, FR 2,162, IT 2,108); 1 article failed validation twice (`272:163`, id leak) and is left out; 22 articles needed the retry. Styles 2,082 to 2,163 each. Mean length in words: keywords 7.5, short_question 18, confused 28, third_party 31, lay_situation 32, legal_wording 32. Live cost **0.87 USD** (estimate was 1.45); month spend about 11.8 of 25 USD.
- Known noise (not measured yet): a few questions garble a legal term, and some questions on generic provisions fit a neighbouring article too. Planned handling: the hard-negative false-negative filter, and a manual read of 20 mined examples before training.

## 2026-10-28, Step 2 finished: split, mining, hard-negative filter, README

- Split by article (hash, per act): 11,504 train pairs (2,876 articles), 1,296 validation pairs (324 articles). One passage language per pair by hash (DE/FR/IT about a third each).
- Mining (`mine.py`, 5 min on CPU): base-model rank of the positive among the 3,201 pool articles: top-1 20%, top-10 39%, top-30 50%. By question language, top-10: EN 29%, DE 49%, FR 50%, IT 46%. By style, top-10: legal_wording 76%, short_question 43%, third_party 39%, lay_situation 32%, confused 28%, keywords 13%. Cosine scores are compressed: positive 0.874 on average, best wrong article 0.892, 30th-best 0.873.
- Filter (`hardneg.py`, draft by Claude, thresholds for Francisco's review): the first draft produced hubness (100 short generic articles filled 49% of first-negative slots, median chosen length 140 characters against 377 in the pool). Francisco chose to add a minimum length (100 characters) and a hub cap (25 uses per article): top-100 share 14%, median chosen length 347, but 2,007 of 11,504 training questions (17%) now have no hard negative (keywords 85% with fewer than 3, English 21% empty). Decision: leave it and measure whether hard negatives help at all (run 1 in-batch only, run 2 with hard negatives) before tuning the filter.
- My reading of the 20-example sheet before the cap: no clear false negatives, about 6 of 20 questions with informative negatives. The project owner's own count of false negatives in the sheet is still to be recorded (kickoff asks for it before training).
- Leakage tests extended to `pairs.jsonl` and `hardneg.jsonl` (positives in the pool, no article in both splits, negatives in the pool, not the positive, not gold): 11 tests pass. README written: `data/finetune/README.md`.

## 2026-10-28, change of working mode and audit of the training data

- Francisco reads English only, so reviewing German, French and Italian examples is not possible. Agreed: replace manual review with an LLM audit by a second family plus English glosses, keep the English-only decisions (learning rate and epochs, predictions as numbers, reading curves), and run Steps 3 and 4 more automatically with documented defaults. A visual follow-up project (images or document pages, results judged by eye) is a later idea, not part of this phase.
- Audit (`src/fedlex_bench/finetune/audit.py`, judge gpt-6-luna, 330 calls, 0.04 USD): first hard negative chosen by the filter is a false negative in 3 of 100 (95% CI 0.01 to 0.08); the base model's unfiltered best wrong article in 2 of 100; random control 0 of 30; the labelled positive fully answers the question in 79 of 100 (partly 20, no 1). The filter's benefit is hubness (49% to 14% of first slots in the top 100 articles), not fewer false negatives. All 3 filtered false negatives are repeated rules for another company form in the Code of Obligations. About 1 in 5 questions is only partly answered by its article (`lay_situation` 48% of 23). Details in `data/finetune/README.md`. English glosses of the 20-example sheet: `data/finetune/review_hardneg_en.md`.

## 2026-10-28, bug found in the Phase 2 embedder (affects Phase 2 retrieval numbers)

- **Symptom.** The torch validation of the untrained model (all 3,201 pool articles as candidates) gave recall@10 0.78 (English 0.67), while my mining with the repo's ONNX embedder gave 0.44 (English 0.34) for the same questions.
- **Diagnosis.** Torch embeddings equal ONNX embeddings of single texts to four decimals (cosine 1.0000), and the ONNX graph itself is padding-invariant. But `OnnxE5Embedder._embed` took its lengths from `encode_batch`, and the tokenizer was set to pad every text to the longest of the call, so every text in a batch looked full length and its attention mask was 1 over the pad tokens: the pad tokens were averaged into the vector. A single text (what the retrievers do for a query) is unaffected; batched texts (every passage, and queries embedded in a batch) were not. Cosine between a passage embedded alone and in a batch: 0.93 to 1.00; for the cached corpus matrix a lot of the damage came from 1,024-passage chunks padded to the chunk's longest text. Query vectors embedded in a batch of 81 gave recall@10 0.36 against 0.54 embedded one by one (same cached passages).
- **Fix** (`retrieval/embedding.py`, commit 7c5bdbe): the real length is the mask sum. The corrected embedder has its own model id (`.../onnx-fp32/padmask`) and cache store; `OnnxE5Embedder(mask_padding=False)` reproduces the Phase 2 vectors. Regression test: a short text in a batch with a long one equals the text alone.
- **What is affected.** (1) All Phase 2 retrieval numbers that use embeddings (A, B, C, D, and the dev retrieval table) came from corrupted passage vectors; they understate the base model, probably by a lot for English. The frozen test results in `docs/phase2_report.md` were produced with the buggy embedder. The reranker, answerer and judge numbers are model outputs given those candidates and are not wrong as measured, but the pipelines they describe were weaker than the model allows. (2) Everything I built on the mined candidates in Step 2 used corrupted vectors: the base-model ranks of the positives, the hubness finding (short generic articles with padded vectors were the hubs, very probably an artifact), the hard negatives, the review sheets and the audit of negatives. These are rebuilt with the corrected embedder (mining, hard negatives, dev retrieval baselines `dev_ret_*_padfix`); the audit and the README numbers follow. The torch training and the in-batch run are unaffected (training uses torch).
- **Open decision for the project owner.** How to treat the frozen Phase 2 test results: (a) leave them, add an errata section to the Phase 2 report with the corrected dev numbers and a corrected retrieval-only test number for A, stating that test was reused for the corrected retrieval-only run; (b) rerun the frozen B, C, D configurations with the corrected embedder (paid, about 3 USD for answering, judge extra, and the test split is used a second time). Nothing has been rerun on test.

## 2026-10-28, errata run for Phase 2 (option a): corrected system A on test

Written before any corrected test number exists. Corrected dev retrieval (`dev_ret_*_padfix`, 63 items, intervals about 30 points wide): hybrid on the all-language index recall@10 0.46 -> 0.53 overall (English 0.27 -> 0.40, German 0.54 -> 0.62, French 0.43 -> 0.50, Italian 0.59 -> 0.59); embedding only 0.40 -> 0.53 (English 0.27 -> 0.43). Frozen configuration `A_padfix` (hash 75cfac7c94): the Phase 2 system A (hybrid, all-language, article chunks, retrieve 60, no reranker, no model) with the corrected embedder; the config now records the embedder id, which is why its hash differs from A's `78df4b6f59`.
- Hypothesis: corrected A has a higher test recall@10 than Phase 2's A (0.49 overall, English 0.14). Expected overall 0.50 to 0.60 and English 0.25 to 0.40; German, French and Italian change by less. Only the overall and English differences can plausibly be claimable with 110 groups.
- This is the second use of the test split for system A, a retrieval-only run with no model call and no tuning. It is logged in `eval/test_runs.log` and will be stated in the Phase 2 errata.

## 2026-10-28, run 1 (in-batch negatives only) on a Kaggle T4

Settings chosen by Francisco: learning rate 2e-5, 3 epochs, batch 64, scale 20, warmup 10%, linear decay, fp16 autocast, passages cut at 256 tokens, queries at 128, seed 20261028; 11,504 train pairs, 180 steps per epoch, 0.4 s per step (first epoch), about 4 minutes per epoch with evaluation. Validation: 1,296 synthetic questions on 324 held-out articles, ranked among all 3,201 pool articles (best of the three language versions).

| Epoch | Train loss | Val loss | Val R@1 | Val R@10 | R@10 EN | R@10 DE | R@10 FR | R@10 IT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 (untrained) | n/a | 3.120 | 0.507 | 0.781 | 0.670 | 0.911 | 0.871 | 0.893 |
| 1 | 0.957 | 0.300 | 0.699 | 0.925 | 0.920 | 0.915 | 0.933 | 0.944 |
| 2 | 0.279 | 0.257 | 0.717 | 0.934 | 0.927 | 0.938 | 0.933 | 0.953 |
| 3 | 0.218 | 0.246 | 0.719 | 0.938 | 0.927 | 0.946 | 0.943 | 0.953 |

- Validation loss fell at every epoch and ended 0.03 above the training loss; there is no sign of overfitting yet. Most of the gain came in the first epoch (English R@10 0.67 to 0.92, R@1 0.40 to 0.67); epochs 2 and 3 add 1.3 points of overall R@10 together, about the width of the interval of 1,296 questions (roughly 1.4 points), so they are not distinguishable from noise. No forgetting is visible on the synthetic validation: German, French and Italian improved too.
- **Stop-rule flag.** English recall@10 is above 90% (0.927) with no LLM. This is on the synthetic validation, drawn from the same generator as the training questions, and the base model already scored 0.67 there; it is not a dev or test number, and the dev baseline of the base model is 0.43 (English) with the corrected embedder. Whether the gain carries to the human-style benchmark questions is exactly what the dev evaluation tests. The rule is treated as a reason to check dev next and to not report validation numbers as a result.
- Checkpoint: `models/r1_inbatch/best` (epoch 3 by validation recall@10). Not yet evaluated on dev.

## 2026-10-28, run 1 on dev (first look at the real benchmark)

`models/r1_inbatch/best` (epoch 3) through the Phase 2 harness (`TorchE5Embedder`, passages up to 512 tokens, runs `dev_ret_*_r1`), compared with the corrected base model (`dev_ret_*_padfix`). Dev, 58 answerable items in 40+ question groups, 95% group-bootstrap intervals (about 30 points wide for one system; the paired differences are tighter).

| Dev, all-language index | Base (corrected) | Tuned run 1 | Paired difference |
| --- | --- | --- | --- |
| Embedding only, recall@10 | 0.53 [0.35, 0.69] | **0.79 [0.63, 0.92]** | **+0.267 [+0.148, +0.407]** |
| Embedding only, recall@1 / MRR | 0.22 / 0.33 | 0.43 / 0.59 | MRR +0.257 [+0.158, +0.360] |
| English / German / French / Italian recall@10 | 0.43 / 0.67 / 0.37 / 0.66 | 0.80 / 0.88 / 0.70 / 0.81 | per-language cells hold 15 groups: direction only |
| Hybrid (BM25 + embedding), recall@10 | 0.53 [0.36, 0.67] | 0.72 [0.55, 0.87] | +0.198 [+0.096, +0.308] |

- Gains are in every language and every index configuration (same-language, single-language, all-language); in the same-language controls (German, French, Italian questions against their own language) the tuned model is not worse (0.88 / 0.70 / 0.81 against 0.67 / 0.37 / 0.66): no sign of forgetting on dev.
- After tuning, **embedding alone (0.79) is better than the hybrid (0.72)** on this index: fusing BM25 with a tuned embedding model drags it down, particularly for English questions where BM25 finds little. One dev comparison, not claimable on its own; to be checked with a weighted fusion.
- **Against the LLM query translation (system C, dev run `dev_final_C`, built on the buggy embedder, with an LLM reranker over its candidates):** recall@10 0.79 [0.63, 0.92] for both, English 0.80 for both; latency 0.09 s against 11.4 s (p50), cost 0 against 0.0029 USD per question. But recall@1: tuned embedder 0.43, C 0.75 (the LLM reranker orders the top 10, which the embedder alone does not). So on this dev set the tuned model closes the *candidate-finding* gap that translation closes, not the *ordering* gap that the reranker closes. Dev only; not frozen, not on test, and C's first stage was degraded by the embedder bug, so the real gap to a corrected C is unknown.
- Stop rules: no model worse than base; no overfitting seen in run 1; English dev recall@10 is 0.80, below the 90% flag.
- Not done: hard negatives (run 2 in progress), data-size ablation, forgetting check beyond the same-language controls, larger base model, ONNX export and CPU latency of the tuned model (the 0.09 s above is the harness' retrieval time with torch on CPU for a single query).

## 2026-10-28, run 2 (in-batch plus one hard negative per question) on the Kaggle T4

Same settings as run 1 (2e-5, 3 epochs, batch 64), `hard negatives True`, corrected hard negatives (kit 45a81b62). Epoch 0 identical to run 1 (validation loss 3.1196, recall@10 0.7809), so data and validation are the same.

| Epoch | Train loss (run 2 / run 1) | Val loss (run 2 / run 1) | Val R@1 | Val R@10 | R@10 EN | R@10 DE |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.38 / 0.96 | 0.292 / 0.300 | 0.701 / 0.699 | 0.927 / 0.925 | 0.918 / 0.920 | 0.929 / 0.915 |
| 2 | 0.52 / 0.28 | 0.249 / 0.257 | 0.721 / 0.717 | 0.934 / 0.934 | 0.921 / 0.927 | 0.942 / 0.938 |
| 3 | 0.41 / 0.22 | 0.240 / 0.246 | 0.721 / 0.719 | 0.9375 / 0.9375 | 0.926 / 0.927 | 0.942 / 0.946 |

- On validation, hard negatives made no measurable difference: recall@10 is identical at epoch 3 (0.9375), recall@1 and MRR differ by 0.002 and 0.000, validation loss is 0.006 lower (validation loss is computed without hard negatives in both runs). Training loss is higher because the extra negatives are harder, not because learning is worse; the two training curves are not comparable.
- Cost: 0.7 s/step in epoch 1 against 0.4 s/step (about 1.7 times slower per step).
- Validation sits near a ceiling for in-domain synthetic questions, so it is a weak instrument for this comparison. Dev (the human-style benchmark questions) is the check; the checkpoint has to be downloaded first.

## 2026-10-28, run 2 on dev (hard negatives) and the choice between runs

Dev, `models/r2_hardneg/best` (runs `dev_ret_*_r2`), paired with run 1 (`*_r1`) and the corrected base model (`*_padfix`), 58 answerable items:

| Dev, all-language index | Base | Run 1 (in-batch) | Run 2 (+ hard negatives) | Run 2 minus run 1 |
| --- | --- | --- | --- | --- |
| Embedding only, recall@10 | 0.53 | 0.79 [0.63, 0.92] | 0.81 [0.65, 0.93] | +0.017 [+0.000, +0.061], no claim |
| Embedding only, recall@1 / MRR | 0.22 / 0.33 | 0.43 / 0.59 | 0.39 / 0.58 | MRR -0.014 [-0.058, +0.026], no claim |
| Hybrid, recall@10 | 0.53 | 0.72 | 0.74 | +0.017 [+0.000, +0.061], no claim |
| English / German / French / Italian (embedding only) | 0.43 / 0.67 / 0.37 / 0.66 | 0.80 / 0.88 / 0.70 / 0.81 | 0.80 / 0.88 / 0.77 / 0.81 | one French item |

- Run 2 against base: recall@10 +0.284 [+0.164, +0.427] (embedding only), +0.216 [+0.112, +0.328] (hybrid).
- **Hard negatives do not help measurably** on dev (the difference is +0.017 with an interval touching zero; the only language that moved is French, by one item group) nor on validation, and they cost 1.7 times the training time per step. Planned comparison, decided before the dev run: use the simpler in-batch run unless hard negatives are clearly better. **Run 1 (in-batch only) is the main configuration.** The hard-negative filter, the audit and the corrected mining stay in the repository as measured negative results.
- Why they probably did not matter: the base model was already strong at ranking the positive in the top 30 for 84% of questions, in-batch negatives (63 random articles per question) already provide most of the signal, and the filter keeps only safe negatives, which are by construction not the confusing ones. These are explanations, not tested ones.
- Dev has now been used for three decisions (run 1 versus base, run 2 versus run 1, tuned versus hybrid); no choice was made on test.

## 2026-10-28, data-size ablation (in-batch, 2e-5, 3 epochs; 25% and 50% of the training articles, nested)

Validation (1,296 synthetic questions, 3,201 candidates): recall@10 0.781 (base), 0.918 (25%: 2,876 pairs, 720 articles), 0.925 (50%: 5,752 pairs), 0.9375 (100%: 11,504 pairs); English 0.670, 0.906, 0.912, 0.927; validation loss 3.12, 0.368, 0.308, 0.246.

Dev, embedding only, all-language index (runs `dev_ret_embedding_all_lang_article_r3_frac25|frac50|r1|padfix`):

| Training data | Recall@10 | Recall@1 | MRR | EN / DE / FR / IT recall@10 |
| --- | --- | --- | --- | --- |
| 0% (corrected base) | 0.53 [0.35, 0.69] | 0.22 | 0.33 | 0.43 / 0.67 / 0.37 / 0.66 |
| 25% | 0.76 [0.59, 0.89] | 0.38 | 0.53 | 0.80 / 0.88 / 0.63 / 0.75 |
| 50% | 0.76 [0.59, 0.89] | 0.39 | 0.54 | 0.80 / 0.88 / 0.63 / 0.75 |
| 100% (run 1) | 0.79 [0.63, 0.92] | 0.43 | 0.59 | 0.80 / 0.88 / 0.70 / 0.81 |

Paired differences on dev: 25% over base +0.233 [+0.117, +0.362] recall@10, MRR +0.194 [+0.111, +0.288] (claimable); 50% over 25% 0.000 (no claim); 100% over 50% recall@10 +0.034 [0.000, +0.093] (no claim), MRR +0.053 [+0.011, +0.096] (claimable).
- Most of the gain (recall@10 +0.23 of +0.26) comes from the first 25% of the data (720 articles, 2,876 questions, 135 steps); further data improves the ordering within the top 10 (MRR) and the French and Italian cells, not the English cell (0.80 at every size). Per-language cells are 15 groups, so only the direction counts.
- Confound: the smaller runs also take proportionally fewer steps (135, 270, 540). A fixed-steps ablation would separate data from steps; it was not run. One seed per size; seed-to-seed variation was not measured.

## 2026-10-28, freeze for test: T1, T2 and C' (written before any of these three has a test number)

Dry runs on dev (run once each before freezing, to catch bugs; same arguments as the frozen configurations), 58 answerable items, intervals about 30 points wide:

| Dev | Recall@1 | Recall@10 | MRR | English R@10 | Cost / question | p50 / p95 latency |
| --- | --- | --- | --- | --- | --- | --- |
| T1 tuned embedding, no model | 0.43 | 0.79 [0.63, 0.92] | 0.59 | 0.80 | 0 | 0.1 / 0.3 s |
| T2 tuned embedding + Haiku reranker | 0.83 [0.68, 0.94] | 0.91 [0.75, 1.00] | 0.88 | 0.93 | 0.0018 USD | 1.4 / 4.5 s |
| C' translation + corrected hybrid + Haiku reranker | 0.75 [0.58, 0.88] | 0.84 [0.69, 0.96] | 0.82 | 0.93 | 0.0025 USD | 8.1 / 12.0 s |
| C (Phase 2, buggy embedder) | 0.75 | 0.79 | 0.80 | 0.80 | 0.0029 USD | 11.4 / 18.0 s |

Frozen configurations (each run once on test; retrieval metrics only: no answerer, no judge):
- **T1**: `--method embedding --config all_lang --chunking article --embedder models/r1_inbatch/best --retrieve-k 60 --rerank-k 30` (tuned run 1, epoch 3, weights hash in the config).
- **T2**: T1's embedder with `--rerank claude-haiku-5-5 --retrieve-k 150 --rerank-k 60`.
- **C'**: Phase 2's C without the answerer and with the corrected embedder: `--translate claude-haiku-5-5 --method hybrid --chunking article --rerank claude-haiku-5-5 --retrieve-k 60 --rerank-k 60`.

Hypotheses, fixed now:
1. T1 beats the corrected system A (0.55 on test) on recall@10 and MRR, claimably. Expected T1 recall@10 0.70 to 0.85 overall, English 0.60 to 0.85, recall@1 0.35 to 0.50.
2. T2 beats T1 on recall@1 and MRR, claimably (the reranker orders the top 10); expected T2 recall@10 0.80 to 0.92, recall@1 0.65 to 0.85.
3. T2 and C' are not distinguishable on recall@10 (expected difference below 0.08, not claimable) and T2 has lower latency (about 1 to 2 s against 8 to 10 s p50) and lower cost per question.
4. T1 against C' on recall@10: T1 lower by 0 to 0.15; with about 110 groups the interval may or may not exclude zero. The claim to make: "within X of the LLM translation pipeline at no cost and 0.1 s", with the interval and the recall@1 gap stated, not "matches".
5. English: T1's English recall@10 is at least 0.60 (Phase 2 corrected A: 0.24).
Notes: cost per question for T2 and C' comes from live calls; the latency of cached model calls is replayed. Judge cost is zero (no judge). These three are the up-to-three configurations allowed; no other configuration will be run on test in Phase 3 unless a bug requires it, which would be reported. Total spend estimate for the three test runs: about 0.8 USD.

## 2026-10-28, test results for the three frozen configurations (each run once; retrieval metrics only)

Runs `2026-10-28_test_aec4726ef0` (T1), `..._ae60b9e8c1` (T2), `..._033e9b80b2` (C'); the corrected A is `..._75cfac7c94`. n = 164 answerable items (110 originals plus translations), 95% group-bootstrap intervals.

| System | Recall@1 | Recall@10 | MRR | Cost / q (USD) | Latency p50 / p95 |
| --- | --- | --- | --- | --- | --- |
| A, Phase 2 (buggy embedder) | 0.22 [0.15, 0.29] | 0.49 [0.40, 0.58] | 0.32 [0.25, 0.40] | 0 | 0.1 / 0.1 s |
| A, corrected embedder | 0.25 [0.18, 0.33] | 0.55 [0.46, 0.63] | 0.36 [0.29, 0.44] | 0 | 0.1 / 0.1 s |
| **T1** tuned embedding, no LLM | 0.48 [0.38, 0.59] | **0.81 [0.71, 0.90]** | 0.63 [0.53, 0.73] | 0 | 0.1 / 0.1 s |
| **T2** tuned embedding + Haiku reranker | **0.79 [0.70, 0.88]** | **0.89 [0.80, 0.97]** | 0.87 [0.78, 0.95] | 0.0016 | 1.6 / 4.5 s |
| **C'** translation + corrected hybrid + Haiku reranker | 0.76 [0.67, 0.84] | 0.87 [0.79, 0.94] | 0.84 [0.76, 0.91] | 0.0024 | 8.0 / 10.5 s |
| C, Phase 2 (buggy embedder; answer step included) | 0.74 [0.65, 0.83] | 0.81 [0.73, 0.89] | 0.80 [0.71, 0.88] | 0.0028 | 18.3 / 23.7 s |

Recall@10 by question language (English / German / French / Italian): A corrected 0.24 / 0.60 / 0.72 / 0.63; **T1 0.80 [0.68, 0.90] / 0.79 / 0.81 / 0.83**; T2 0.89 / 0.87 / 0.88 / 0.91; C' 0.86 / 0.90 / 0.84 / 0.90; C (Phase 2) 0.71 / 0.85 / 0.84 / 0.85. Originals only (n = 110) recall@10: A corrected 0.58, T1 0.88, T2 0.93, C' 0.89.

Paired differences (first minus second, recall@10 and MRR, 95% interval):
- T1 minus corrected A: **+0.262 [+0.174, +0.351]**, MRR +0.271 [+0.182, +0.361] (claimable).
- T2 minus T1: recall@10 **+0.079 [+0.029, +0.139]**, MRR **+0.238 [+0.158, +0.320]** (claimable).
- T2 minus C': recall@10 +0.015 [-0.052, +0.074], MRR +0.032 [-0.036, +0.091] (**not claimable**); T2 is 5 times faster (p50 1.6 against 8.0 s) and 33% cheaper (0.0016 against 0.0024 USD per question).
- T1 minus C': recall@10 **-0.064 [-0.149, +0.016]** (not claimable), MRR **-0.207 [-0.305, -0.106]** (C' orders better; claimable).
- C' against Phase 2's C was not compared pairwise (the harness refuses runs with different judges); unpaired recall@10 0.87 [0.79, 0.94] against 0.81 [0.73, 0.89] overlap.

Against the hypotheses fixed before the runs: 1 (T1 beats corrected A, T1 recall@10 in 0.70 to 0.85, English at least 0.60, recall@1 in 0.35 to 0.50): held (0.81, 0.80, 0.48). 2 (T2 beats T1 on recall@1 and MRR; T2 recall@10 0.80 to 0.92, recall@1 0.65 to 0.85): held (0.89, 0.79; MRR claimable). 3 (T2 and C' indistinguishable on recall@10, T2 faster and cheaper): held. 4 (T1 below C' by 0 to 0.15 on recall@10, maybe not claimable): held (-0.064, not claimable). 5: held. No hypothesis failed.

What this means for the core question (stated with its limits):
- Candidate finding with no LLM: a fine-tuned 118M-parameter embedding model reaches recall@10 0.81 on test, within 0.06 of the LLM-translation pipeline's 0.87 (interval -0.15 to +0.02: the difference is not resolved either way), at 0 cost and 0.1 s instead of 0.0024 USD and 8 s. It is clearly worse at putting the right article first (MRR 0.63 against 0.84; recall@1 0.48 against 0.76).
- With the Haiku reranker on top, the tuned model is as good as the LLM-translation pipeline (+0.015 recall@10, +0.032 MRR, both inside the noise) at a fifth of the latency and two thirds of the cost; the English gap is closed (0.89 against 0.86).
- The result is on one benchmark that rewards finding an article an LLM drafted a question from (see the Phase 2 limitations). It does not show anything about answers: no answerer or judge ran.
- Spend: three test runs plus dry runs about 0.6 USD; month spend 12.67 of 25 USD.

## 2026-10-28, B' (Phase 2 errata run, option b in its small form): corrected B, retrieval only

Francisco approved one more frozen retrieval-only configuration to give Phase 2's system B a corrected number; it is an errata run for Phase 2, not one of Phase 3's three configurations (T1, T2, C'). **B'** = Phase 2's B without the answerer, with the corrected embedder: `--method hybrid --config all_lang --chunking article --rerank claude-haiku-5-5 --retrieve-k 150 --rerank-k 60`. Dev dry run (`dev_Bprime`): recall@1 0.75 [0.59, 0.88], recall@10 0.81 [0.66, 0.93], MRR 0.81; Phase 2's B on dev: 0.62 / 0.66 / 0.67 (buggy embedder).
Hypotheses, fixed before the test run: B' recall@10 on test is higher than Phase 2's B (0.65), expected 0.72 to 0.85, English 0.45 to 0.75 (B: 0.25); B' is below T2 on recall@10 (T2 0.89) but the difference may not be claimable; no claim is made about answers.

B' on test (run `2026-10-28_test_b7846648c0`, once): recall@1 0.65 [0.56, 0.74], recall@10 **0.74 [0.65, 0.82]**, MRR 0.72 [0.63, 0.81], English recall@10 0.52 [0.38, 0.67] (B: 0.25), German 0.77, French 0.86, Italian 0.81; cost 0.0016 USD per question, p50 1.4 s. Phase 2's B (buggy embedder, with answer step): recall@1 0.59, recall@10 0.65 [0.56, 0.73], MRR 0.64. The hypotheses held (recall@10 in 0.72 to 0.85, English in 0.45 to 0.75). T2 minus B': recall@10 +0.149 [+0.077, +0.225], MRR +0.148 [+0.074, +0.225] (claimable): the tuned first stage beats the corrected untuned hybrid by 15 points. B' and B were not compared pairwise (different judge setting in the run configuration); the unpaired intervals for recall@10 (0.74 against 0.65) overlap slightly. Phase 2's answer-level numbers (fact recall, faithfulness, traps) were not re-measured.

## 2026-10-28, review by the eval-reviewer agent, and what was done about it

No high-severity finding; every test number in the report matched the evidence. Findings and actions:
1. **Agent config hash ignored the embedder** (a run of the frozen D configuration with another embedder would have passed the guard): fixed, the ADK agent's configuration now records the embedder id. D's frozen hash therefore no longer matches that code; D was not rerun and would need a new freeze.
2. **The T1 and T2 gains are tuned embedding-only against untuned hybrid**: wording fixed in the report and model card; supporting dev evidence (the base model scored 0.53 as embedding-only and as hybrid) stated.
3. **Provenance of run 1** (git sha unknown, tokenizer not hashed): stated in the report's limitations.
4. **B' was frozen after T1, T2 and C' were known**: disclosed in the report.
5. **Report claim "validation loss within 0.03 of training loss"** was false for epoch 1: reworded.
6. **Audit reported selectively** (1% "yes" omitted 24% "partly"): both reported.
7. **Embedding-only against hybrid after tuning on dev**: MRR +0.154 [+0.052, +0.259] is claimable, recall@10 +0.069 [-0.013, +0.173] is not: stated.
8. **Twin exclusion is symmetric** (containment gap): not fixed (would change the pool and require retraining); quantified (6, 37 and 84 pool articles at 80%, 60% and 50% containment, touching 5, 18 and 33 gold articles) and the headline gain re-estimated on test items not touched at 60%: +0.30 [+0.20, +0.40] in group means (n = 142 items, 97 groups); T1's recall@10 is 0.98 on the 22 touched items and 0.78 on the rest (untuned: 0.68 and 0.52). Disclosed as limitation 10.
9. **Leakage tests weaker than claimed**: added an independent twin implementation, a test that every model's recorded data hashes match the current files, and a batch-equals-single test for the torch embedder. The corpus tests still skip when the data is absent.
10. **Persona code never applied**: dead code removed, data README, report and model card corrected (the generated data itself is unchanged).

## 2026-11-03, publication

- GitHub: `franciscomira/fedlex-rag-benchmark` (public) pushed to `main` after a read-only scan (no secrets in the tree or the history, no corpus parquet or `.env` tracked, largest tracked file 3.8 MB); README updated first (Phase 3 table, Phase 2 erratum notice).
- Hugging Face: `franciscomira/fedlex-e5-small` (public), uploaded with `src/fedlex_bench/finetune/release.py` (checkpoint, int8 ONNX, tokenizer, training config and log, model card). The base model's licence (MIT) was checked on its model page. Verified by downloading the published files and comparing embeddings with the local checkpoint: cosine 1.000000 for queries and passages. The synthetic dataset was not published to Hugging Face (it is in the GitHub repository). The int8 file was checked on dev only.
