# LinkedIn post (draft, not posted)

Numbers copied from the rendered post (`docs/writeup/post.md`, built from the harness outputs). Re-check them against the post if either changes. Link target after publication: https://franciscomira.github.io/fedlex-rag-benchmark/post.html

---

I wanted to answer one question with measurements, not opinions: can a small fine-tuned model replace an LLM call in cross-lingual legal retrieval?

The setting: Swiss federal law in German, French and Italian (8 acts, 4,073 articles), questions in four languages, answers that must cite articles, with code validating every citation.

What I found, on a test set of 164 questions with 95% intervals:

- A 118M-parameter embedding model, fine-tuned on a free GPU (the whole phase cost about 2 USD in API calls), finds the right article in the top 10 for 81% of questions [71, 90], against 55% [46, 63] untuned. For English questions it goes from 24% to 80%, with no LLM call.
- An LLM-translation pipeline gets 87% [79, 94]. I can't tell the two apart on that metric, but the LLM pipeline orders results better.
- A full pipeline on the tuned model answers as well as the LLM-translation pipeline as far as I can measure (fact recall 0.80 against 0.82, not distinguishable), at 20% lower cost and 2.2 times lower latency.

What went wrong: a padding bug in my own embedding code made every first-round retrieval number too low. I found it, corrected the numbers and published an errata.

What I would not claim: the questions were written by an LLM from the article text, the test set has about 110 independent questions, nobody who reads German, French and Italian reviewed the training questions, and no Swiss-law expert has looked at any of it. There is no live demo.

Post, code, model and benchmark:
https://franciscomira.github.io/fedlex-rag-benchmark/post.html

#NLP #RAG #LegalTech #Switzerland #Evaluation
