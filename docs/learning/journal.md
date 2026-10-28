# Prediction journal (Phase 3)

Rule: write the prediction **before** the run, with a reason. After the run we compare and add what we learned. Numbers are recall@10 on **dev** (63 items: 40 originals, 5 traps, 18 translations; traps excluded from retrieval metrics), system A style (hybrid vs embedding only is stated per entry).

Reference points from Phase 2 (test split, bare hybrid, all-language index): overall 0.49, English 0.14, German 0.58, French 0.65, Italian 0.60. After LLM query translation (C): overall 0.81, English 0.71.

## Entry 1: baseline, base multilingual-e5-small on dev (no training)

Embedding only, all-language index, article chunks. Predict recall@10 for English questions and for German questions, and say why they differ.

- Prediction (Francisco, 2026-10-28, before looking at dev numbers): English recall@10 25%, German 55%. Reason: English fails because of the cross-lingual gap.
- Result (Phase 2 dev runs, `dev_ret_embedding_all_lang_article`, 63 items, 95% group-bootstrap intervals): English recall@10 **0.27** [0.07, 0.47] (15 question groups), German **0.79** [0.58, 1.00], French 0.30 [0.10, 0.53], Italian 0.31 [0.12, 0.56]; overall 0.40 [0.24, 0.55]. Hybrid on the same index: overall 0.46 [0.29, 0.61], English 0.27, German 0.54, French 0.43, Italian 0.59.
- What we learned: the English guess (25%) was close and for the right reason (all English is cross-lingual). German was 24 points off (guess 55%, result 79%), but so was the pattern: French and Italian came out at 30%, far below German, though Phase 2's test split had them at the level of German (hybrid 0.58 to 0.65). Each language cell holds only 15 groups, so intervals are 40 points wide and the German/French/Italian differences on dev are mostly noise. Practical consequence for Phase 3: dev (63 items) can only confirm large gains; the 1,296-question synthetic validation set is the finer instrument for choices between training runs, and dev stays a once-per-decision check.

## Entry 2: first fine-tune, in-batch negatives only (learning rate and epochs chosen by you)

Predict English recall@10 and German recall@10 after training (embedding only), the learning rate and number of epochs you choose, and whether validation loss will rise before the last epoch.

- Choices (Francisco, 2026-10-28): learning rate 2e-5, 3 epochs (Claude's suggested default, accepted). No prediction given this time.
- Result: _after the run_
- What we learned: _after the run_
