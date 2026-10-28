# Prediction journal (Phase 3)

Rule: write the prediction **before** the run, with a reason. After the run we compare and add what we learned. Numbers are recall@10 on **dev** (63 items: 40 originals, 5 traps, 18 translations; traps excluded from retrieval metrics), system A style (hybrid vs embedding only is stated per entry).

Reference points from Phase 2 (test split, bare hybrid, all-language index): overall 0.49, English 0.14, German 0.58, French 0.65, Italian 0.60. After LLM query translation (C): overall 0.81, English 0.71.

## Entry 1: baseline, base multilingual-e5-small on dev (no training)

Embedding only, all-language index, article chunks. Predict recall@10 for English questions and for German questions, and say why they differ.

- Prediction (Francisco): _write here_
- Result: _after the run_
- What we learned: _after the run_

## Entry 2: first fine-tune, in-batch negatives only (learning rate and epochs chosen by you)

Predict English recall@10 and German recall@10 after training (embedding only), the learning rate and number of epochs you choose, and whether validation loss will rise before the last epoch.

- Prediction (Francisco): _write here_
- Result: _after the run_
- What we learned: _after the run_
