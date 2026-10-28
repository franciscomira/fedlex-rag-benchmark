# Lesson 04: the loss, batch size, learning rate, epochs and overfitting

## What training does, in one paragraph

The model starts as multilingual-e5-small (118M numbers). For a batch of questions it computes vectors, a **loss** says how wrong the ranking is, and **gradient descent** nudges every number a little in the direction that lowers the loss. One nudge is a **step**; one pass over all 11,504 training pairs is an **epoch**. Nothing else happens. Everything below is about how big and how many nudges.

## The loss: MultipleNegativesRankingLoss (MNRL)

For a batch of B pairs (question q_i, positive p_i), compute the B x B table of similarities `sim(q_i, p_j)`. Row i should have its highest value on column i. The loss is cross-entropy over each row: it is small when `sim(q_i, p_i)` is much larger than the other B-1 entries. Two details matter:

- **Scale (temperature).** Cosines are squeezed in a narrow band (our positives averaged 0.874, the best wrong article 0.892). The loss multiplies similarities by a scale (default 20, i.e. temperature 0.05) before the softmax; that is what makes a 0.02 difference matter. A scale that is too low gives weak gradients, too high gives brittle ones.
- **Hard negatives** are extra columns: with a triplet (q, p, n) the row also contains `sim(q_i, n_i)`, and every other row's negative too.

Prefixes: e5 expects `"query: "` before questions and `"passage: "` before articles. If training forgets them, the model is fine-tuned on a different input format from the one used at search time. A silent bug.

## The knobs, with our numbers

| Knob | What it does | Our situation | Failure mode |
| --- | --- | --- | --- |
| **Batch size** | pairs per step; also the number of in-batch negatives (B-1) | 11,504 pairs; B = 64 gives 180 steps per epoch | too small: easy negatives, noisy; too big: needs more memory (a free T4 handles 64 to 128 for e5-small at 256 tokens), and two questions on the same article in one batch become false negatives |
| **Learning rate (LR)** | the size of each nudge | typical for fine-tuning a small encoder: 1e-5 to 5e-5 | too high: the model forgets what it knew (same-language retrieval gets worse), loss jumps; too low: almost nothing changes in 3 epochs |
| **Warmup** | LR rises from 0 over the first ~10% of steps | avoids a violent first few steps | none really; skipping it can hurt at higher LR |
| **Epochs** | passes over the data | with 11.5k pairs, 1 to 5 | too many: **overfitting** |
| **Max sequence length** | tokens kept per article | most pool articles are about 110 tokens; the longest are cut at 512 in the index | shorter in training than at search time hides the tail |

## Overfitting, visible in two curves

Training loss falls as long as you train: the model can always memorise more. **Validation loss** (articles never seen) is the honest curve:

- both falling: learning something general;
- training falls, validation flat or rising: memorising articles, not learning the cross-lingual bridge. Stop, or lower the LR and the epoch count.

Validation **recall@10** per epoch matters more than validation loss for us, because loss measures all ranks and we care about the top 10. We log both, and keep the checkpoint with the best validation recall.

Our validation set is synthetic and from the same generator as training; it will look better than dev. The gap between the two tells us how much of the gain is "learned our synthetic style".

## What can go wrong here specifically

1. **Prefix mismatch** between training and the ONNX search code (check both).
2. **Catastrophic forgetting:** LR too high, so English improves while German gets worse. Step 4 checks same-language questions explicitly.
3. **Duplicate articles in a batch** (several of an article's four questions): false negatives, a small effect at B = 64 (about one collision per batch).
4. **Comparing runs with different seeds** and calling the difference real. With 40 English dev questions one seed can move recall@10 by several points; the Phase 2 bootstrap intervals are the yardstick.
5. **Early stopping on dev:** never. Validation only.

## Your decisions (TODO(human), in the notebook config, next)

You will choose: the **learning rate** and the **number of epochs** for the first run. Write down your guess and why, in `docs/learning/journal.md`, before the run. We will then see if the curves agree.

## Questions for you

1. In the B x B table, what does a "false negative" look like if two questions of the same article sit in one batch, and what does the loss do about it?
2. Training loss 0.4 -> 0.1 over 3 epochs, validation loss 0.5 -> 0.6. What happened, and what would you change first?
3. Why can English recall@10 improve while German recall@10 drops, and which knob do you suspect first?

Answer in chat. I will correct, not answer first.
