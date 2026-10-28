# Lesson 03: hard negatives, false negatives and the split

## Where we are

We have 12,800 (question, article id) pairs for 3,200 articles. Before training we must (a) split by article into train and validation, (b) pick the passage language per pair, and (c) decide whether to add hard negatives. This lesson covers (c), which is where fine-tuning most often goes wrong.

## Why hard negatives help

With in-batch negatives only (batch of 64) the 63 other positives in a batch are random articles. Against "my landlord wants to raise the rent", a random article on company mergers is trivially far away, so the model learns almost nothing from it. After a few hundred steps the loss is small and learning slows.

A **hard negative** is an article the *current* model scores close to the question, but which is wrong. For "can my landlord raise my rent?" that could be the article on rent reductions, or a neighbouring tenancy article. Training against it forces fine distinctions, and at retrieval time those are exactly the mistakes that push the right article out of the top 10.

How we mine them:
1. **BM25** over the same corpus: articles sharing words with the question.
2. **The base embedding model** (e5-small): articles near the question in vector space.
3. Take the top-ranked hits that are not the positive article (any language version of the positive is excluded, since it is the same article).

## The danger: false negatives

Mining finds the articles most similar to the question. The most similar article that is *not the labelled positive* is very often a correct answer too:
- statutes repeat a rule for related cases (lease vs. commercial lease),
- a general rule and its specific exception both answer the question,
- our own pilot showed it: the questions about `210:2`, `210:3` and `210:4` fit each other's articles.

If we train against a false negative, the loss says "push this correct article away". The model learns the wrong thing, and it is silent: loss still goes down. Typical effect: recall@1 gets worse while the training curve looks healthy.

## The trade-off you will decide (TODO(human))

You will write the filter that decides which mined candidates are safe to use. Options, from the data we have:

- **Rank window:** skip the top few hits and take ranks 4 to 30, on the assumption that the very closest ones are the ambiguous ones. Cheap, but arbitrary.
- **Score margin:** use a candidate only if its score is below the positive's score minus a margin. If the model rates the "negative" as high as the positive, it probably is not wrong.
- **Same-act or neighbour rule:** drop candidates from the same act within a few article numbers of the positive.
- **Text overlap rule:** drop candidates that look like twins of the positive (Jaccard, as in the leakage code).
- **LLM check:** ask a model "does this article also answer the question?" for a sample. Costs money; we could use it to *measure* the false-negative rate of a rule, rather than to filter all pairs.

Whatever you pick, there are two failure modes: too loose (false negatives get through), too strict (no hard negatives left, so they teach nothing). You will look at 20 mined examples before training, as agreed.

## The split by article

Remember Lesson 02: validation articles are never seen in training. We will hold out about 10% of articles (320) with all of their questions, split deterministically (hash of the article id, sorted), stratified loosely by act so the Code of Obligations does not dominate validation. Validation recall on those 320 articles is the signal for early stopping. Dev stays untouched for model selection.

## What can go wrong

- **Leakage through the validation set:** if validation articles share near-duplicates with training articles, validation looks better than it is. We use the same twin logic.
- **Using dev as a validation set by accident:** every time you look at dev to decide, you spend a bit of its honesty. Validation is for early stopping only.
- **Passage language collisions:** one query paired with several language versions of the same article in a batch makes false negatives. We pick one language per pair, fixed by hash.

## Questions for you

1. Your mined list for a question about apartment rent increases shows, at rank 1, an article about rent reductions after interest rates fall. At rank 2, an article about notice of termination of a lease. Which looks more like a false negative, and why? What information would you check to decide?
2. If your filter is "take ranks 4 to 30", what is one thing it assumes about the data, and one situation where that assumption fails?
3. Why do we hold out validation by article but keep dev as a separate set? What would be the problem with using only one of them?

Answer in chat. I will correct, not answer first.
