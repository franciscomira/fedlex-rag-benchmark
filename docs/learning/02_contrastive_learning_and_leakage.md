# Lesson 02: contrastive learning, positives and negatives, leakage

## The idea

We do not teach the model "the answer"; we teach it **which vectors should be close**. Training data are triples:

- **query** ("Can my landlord keep the deposit after I move out?", English)
- **positive** (the article that answers it, e.g. `220:257e` in German)
- **negatives** (articles that should be *farther* from the query than the positive)

The loss (we will use MultipleNegativesRankingLoss, next lesson) pushes `sim(query, positive)` up and `sim(query, negative)` down. Weights move a little each step; that is all "fine-tuning" is.

**In-batch negatives for free.** With a batch of 32 queries and their 32 positives, every other query's positive in the batch is a negative for this query. 31 negatives per query at no extra cost. That is why big batches help and why the same article must not appear twice in one batch (see "false negatives").

**Hard negatives.** Random negatives are too easy (a tenancy article vs a question about company law). A *hard* negative is one the current model wrongly ranks high, for example the neighbouring article `220:257d` instead of `220:257e`. These teach the fine distinctions, but they are also the dangerous ones.

## Positives and negatives in our data

- The same article in DE, FR and IT is the **same** positive. Article `220:257e` in French is never a negative for a German question about it. (Id is the alignment key.)
- A different article is a negative only if it truly does not answer the question. Statutes repeat and overlap, so some "negatives" actually answer the question: **false negatives**. Training against them teaches the model to push away correct answers. The TODO(human) filter in Step 2 targets this.

## Leakage: the part that can silently ruin the project

If any dev/test gold article (or a near copy) is in the training pairs, the model learns exactly the items we then grade it on, and the benchmark stops measuring generalisation. Our numbers, on the current corpus:

| Set | Article ids |
| --- | --- |
| Retrieval-eligible articles (each in DE, FR, IT) | 3,663 |
| Dev + test gold articles | 172 |
| Their neighbours (previous and next article in the same act, sorted order) | 288 |
| Twins of gold (character 5-gram Jaccard >= 0.7) | 0 |
| **Left for training** | **3,203** |

Three things to notice:

1. **Neighbours matter** because adjacent articles share vocabulary and often the same topic; a question written about article N is half-answered by N+1.
2. **Twins found: 0.** That is by construction, since Phase 1 removed twinned articles from the gold pool. But 0.7 was a threshold chosen for gold uniqueness; for leakage we may want a looser one (for example 0.5), because a 60%-similar article still teaches the model the gold text. This is one of the decisions we will test with data.
3. **A test, not a promise.** The exclusion will be code plus a pytest that fails on any overlap. If it ever fails, we stop.

Also: the 3,203 articles sit in 8 acts, 1,526 of them in the Code of Obligations (220). The model will learn that act's style most; the per-act breakdown is something to check in the forgetting test.

## What can go wrong

- **Style copying.** Our synthetic questions come from an LLM reading an article, the same way the benchmark questions were drafted. If we use the same prompt, training queries look like test queries and we overstate gains. Hence: vary the style (lay situation, legal wording, short keyword, long scenario), different prompt from Phase 1.
- **Language imbalance.** If most training queries are English and German, the model may get worse on Italian.
- **Overfitting to 3,203 articles.** The model can memorise "this phrase belongs to this article". Validation split by *article* (not by query!) is what catches it. A query-level split would put the same article in train and validation and look good for nothing.

## Questions for you

1. Why must the train/validation split be by article and not by question? What would you see in the validation curve if you split by question?
2. Neighbours of gold articles are excluded from training. Give one reason this protects the dev/test numbers, and one cost of doing it.
3. A hard negative for the question "how long is the notice period for an apartment lease?" is a different article that mentions notice periods for commercial leases. Is it a safe hard negative? How would you decide with the data we have?

Answer in chat. I will correct, not answer first.
