Phase 2 is done (docs/phase2_report.md). We are starting Phase 3: fine-tuning. Read CLAUDE.md, docs/PROJECT_PLAN.md, docs/eval_spec.md and docs/phase2_report.md first.

This phase is also my first fine-tuning project, so it runs in learning mode. Teaching me matters as much as the result.

## How we work in this phase
- Output style is Learning (I set it with /output-style learning, or "outputStyle": "Learning" in .claude/settings.local.json). Keep the work in this main session: subagents do not use the output style, so use them only for reviews.
- No agent team in this phase.
- Before each step, write a short lesson in docs/learning/NN_topic.md: the concept in plain English, why it matters here, what can go wrong, and 2 or 3 questions for me. Wait for my answers before running anything. Correct my answers instead of giving them to me first.
- Before each experiment, ask me to write a prediction in docs/learning/journal.md (what will happen and why). After the run, compare result and prediction with me and add what we learned.
- Leave the decisions that teach something to me as TODO(human): the training loss setup, the hard-negative filter, the evaluation loop, and the choice of learning rate and epochs. Handle the routine code yourself.
- Keep explanations short and concrete, tied to our own data and numbers. Assume I know Python and RAG well, but not ML training.

## Step 0: housekeeping
1. Update the CLAUDE.md status line to "Phase 3: fine-tuning (learning mode)".
2. Check the hardware: GPU (NVIDIA, CUDA version, VRAM) or CPU only, RAM and disk. Propose where to train: locally if a usable GPU exists, otherwise a notebook for free Colab or Kaggle GPUs under notebooks/. Do not sign me up for paid compute.
3. Budget: 10.9 USD of the 25 USD monthly cap is spent. Phase 3 should need only a few dollars for synthetic query generation. Show me the estimate before any paid call.

## Step 1: lesson and decision: what exactly do we fine-tune?
Phase 2 found that LLM rerankers are indistinguishable on this benchmark and that first-stage cross-lingual retrieval is the bottleneck. Bare hybrid retrieval finds only 14% of English questions; translating the query with an LLM (C) lifts that to 71%, at about 18 s per question.
Teach me the difference between a bi-encoder (embedding model) and a cross-encoder (reranker), then propose:
- Primary: fine-tune the multilingual embedding model (multilingual-e5-small, the one in system A) for cross-lingual legal retrieval. The new question: can a fine-tuned small embedding model close the gap that LLM query translation closes, with no LLM call at query time?
- Secondary, only if time allows: fine-tune a small multilingual cross-encoder reranker on a wider window and compare it with the Haiku reranker on quality, cost and latency (the original Phase 3 question, which Phase 2 suggests will tie).
Write the decision and the updated core question into docs/PROJECT_PLAN.md after I agree.

## Step 2: training data (lessons: contrastive learning, positives and negatives, leakage)
- Source: retrieval-eligible articles only.
- Exclusions, enforced by code and covered by a test that fails on any overlap:
  - dev and test gold articles (benchmark/v1/*_gold_articles.txt);
  - their neighbours (one article number either side);
  - the near-identical twins of any gold article.
- Synthetic queries: a cheap pinned model writes about 4 questions per article, in EN, DE, FR and IT. Mix lay situations and legal wording. Vary the style deliberately, so we do not just copy the benchmark's drafting style. Cache everything.
- Positives: the same article in all three languages counts as a positive. A different language version is never a negative.
- Hard negatives: mine them with BM25 and the base model. Filter likely false negatives (TODO(human)), and make me look at 20 examples before training.
- Split the training articles into train and validation for early stopping. Dev is for model selection only; test is untouched.
- Write data/finetune/README.md: how the data was built, the counts, and the leakage check result.

## Step 3: baseline and first fine-tune (lessons: loss, batch size, learning rate, epochs, overfitting)
- Baseline: base multilingual-e5-small on dev, in the same configurations as Phase 2 system A (all-language and single-language indexes). Report recall@1/10 and MRR per query language.
- Fine-tune with sentence-transformers and MultipleNegativesRankingLoss. Remember the "query: " / "passage: " prefixes e5 expects.
- First run: in-batch negatives only. Second run: add hard negatives.
- Log training and validation loss and validation recall per epoch. Plot learning curves and explain them to me.
- Fix seeds, and save the config, the data hash and the git sha with every checkpoint. Checkpoints go to models/, which is git-ignored.

## Step 4: experiments on dev (lessons: ablations, noise, confidence intervals)
Compare, with the Phase 2 bootstrap CIs and paired differences:
1. Base vs fine-tuned, in-batch negatives only, vs fine-tuned with hard negatives.
2. A data-size ablation (25%, 50%, 100% of training queries), to show the learning curve.
3. Fine-tuned A (hybrid with the tuned embeddings, no LLM) vs Phase 2 C (LLM translation) on recall@10, latency and cost per query. This is the headline comparison.
4. Optional: a larger base model zero-shot (for example multilingual-e5-base), to see whether size alone does the same job.
5. A forgetting check: does the tuned model get worse on same-language German, French and Italian questions where the base model was already good?
Log everything in docs/phase3_progress.md.

## Step 5: freeze, test once, package
- Freeze up to 3 configurations in eval/frozen_configs.json before running test. Run each once on test, logged in eval/test_runs.log. Re-judge answers only if an answering system changes.
- Measure CPU inference latency of the tuned model, and export a quantized ONNX version for Phase 4 serving on Cloud Run. Report the size and speed difference, and check that recall did not drop.
- Prepare a Hugging Face model card (training data, exclusions, results with CIs, limitations, the Fedlex non-official notice). Ask me before pushing the model or the synthetic dataset anywhere.
- Write docs/phase3_report.md. Run the eval-reviewer on the training code, the leakage test and the report.
- Write docs/learning/summary.md: the concepts I learned, my predictions vs the results, and the 5 questions an interviewer is most likely to ask about this fine-tune, with my own answers corrected.

## Stop and ask me if
- the leakage test finds any overlap with dev or test gold, their neighbours or twins;
- the fine-tuned model is worse than base on dev, or validation loss rises while training loss falls (overfitting);
- a result looks too good to be true (for example English recall@10 above 90% with no LLM);
- training would take more than 2 hours locally, or would need paid compute.
