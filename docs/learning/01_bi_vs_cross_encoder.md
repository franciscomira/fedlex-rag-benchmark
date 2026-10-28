# Lesson 01: what exactly do we fine-tune?

## The two kinds of model

**Bi-encoder (embedding model).** Encodes the query and each article *separately* into a vector (384 numbers for multilingual-e5-small). Relevance = dot product of the two vectors. Article vectors are computed once, offline, and stored (our `store.npz`). At query time only the query is encoded (about 0.1 s on your CPU), then a nearest-neighbour search. This is system A's embedding half.

**Cross-encoder (reranker).** Reads the query and one article *together* in a single forward pass, so every query word can attend to every article word, and outputs one relevance score. More accurate, but nothing can be precomputed: scoring 150 candidates means 150 forward passes per query. So it can only rerank a short list; it cannot search the corpus. Your Haiku reranker is this idea, with an LLM instead of a small encoder.

| | Bi-encoder | Cross-encoder |
| --- | --- | --- |
| Input | text alone | query + article together |
| Precompute articles | yes | no |
| Cost per query | 1 encode + search over ~11k chunks | 1 pass per candidate |
| Role | first stage (find candidates) | second stage (reorder them) |
| Can fix a missing candidate | yes | no |

## Why this matters for our numbers

Phase 2 says the reranker is not where systems differ: once the gold article is among the candidates, every LLM reranker puts it first. The loss happens earlier. Bare hybrid retrieval finds the gold article in the top 10 for only **14%** of English questions, while an LLM translation of the question gets **71%**. A reranker cannot recover an article that was never retrieved (cross-encoder row, last line above).

Why English fails: `multilingual-e5-small` was trained to put translations close together, but a short English lay question about "a landlord keeping the deposit" and a long German legal article are far apart in vector space, and BM25 cannot help at all (no shared words). Fine-tuning on (question in language X, article in language Y) pairs teaches the embedding space exactly this bridge, for this legal domain.

So the proposal:

- **Primary:** fine-tune the **bi-encoder** (multilingual-e5-small, 118M parameters) with synthetic EN/DE/FR/IT questions paired to their articles. Question: can a fine-tuned small embedding model close the gap that LLM query translation closes (14% to 71% on English), with **no LLM call at query time** (0 USD, about 0.1 s instead of 18 s)?
- **Secondary, only if time allows:** fine-tune a small multilingual **cross-encoder** on a wider candidate window and compare it with the Haiku reranker on quality, cost and latency. Phase 2 predicts a tie on quality, so the interesting axes are cost and latency.

## What can go wrong (preview)

- **Leakage:** if a dev or test gold article (or its twin) is in the training pairs, the model memorises the answer and the benchmark stops measuring anything. This is the single largest risk and gets a failing test.
- **Style copying:** our synthetic questions come from an LLM reading the article, just like the benchmark questions. If both share a drafting style, the model learns the style, not the legal-language bridge, and dev looks better than reality would.
- **Forgetting:** fine-tuning on EN-to-DE pairs can make same-language German/French/Italian retrieval worse. We check it explicitly.
- **Too good to be true:** English recall@10 above 90% with no LLM is a stop-and-ask signal.

## Questions for you (answer in your own words, short is fine)

1. A user types an English question. Your embedding model is a bi-encoder and the articles are German. Why can we embed the 11k articles once and never again, but could not do the same with a cross-encoder?
2. Phase 2: the Haiku reranker lifted English recall@10 from 14% (A) only to 25% (B), while query translation (C) reached 71%. Using the table above, explain why the reranker could not do more.
3. If the fine-tuned embeddings reach, say, 60% on English with no LLM, what exactly would you compare it against to say whether "closing the gap" worked, and on which two other axes besides recall@10?

Write your answers into `docs/learning/01_answers.md` (or tell me in chat). I will correct them, not hand you mine first.
