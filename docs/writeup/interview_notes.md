# How to talk about this project (for interviews)

Private prep notes, not for publication. Numbers are from the rendered post (`docs/writeup/post.md`); every one comes from the evaluation harness with a 95% interval.

## The 90-second version

I measured whether a small fine-tuned embedding model can replace an LLM call in cross-lingual retrieval over Swiss federal law. Retrieval was the bottleneck, not reranking, so I fine-tuned the embedder (118M parameters, a free GPU, about 2 USD of API calls for the whole phase). On the test set it finds the right article in the top ten for 81% of questions against 55% untuned; English questions go from 24% to 80%. An LLM-translation pipeline gets 87%, and I can't tell the two apart on that metric. A full pipeline on the tuned model answers as well as that pipeline within the noise (fact recall 0.80 against 0.82), 20% cheaper and 2.2 times faster. I also found a bug in my own embedding code that had made my first-round numbers too low, and I published the correction.

## What I can defend

- **Process.** Hypotheses written before every test run, each frozen configuration run once, paired bootstrap intervals that resample question groups, one command per number, a stop rule that actually fired and was handled (the int8 model lost ranking quality on test, I stopped and asked before spending more).
- **Honesty about results.** I say which differences are not claimable, I report the hypothesis I missed (fact recall came out above the range I wrote), and I kept all ten recorded demo answers, including a refusal of a question the law answers in part.
- **Engineering.** Code validates every citation against the retrieved articles; the model never produces ids. Leakage tests between training and benchmark. A serving path that is the same object the harness measured, with a check that the model file is the measured one.

## What I would not claim

- That the fine-tuned model "beats" or "matches" the LLM pipeline: the intervals allow a difference of several points either way on finding the article, and the LLM pipeline orders results better.
- That the gain is cross-lingual retrieval and not partly the model learning the style in which an LLM wrote the questions.
- Anything about real users or real lawyers: no real questions, no Swiss-law expert, no German/French/Italian reader reviewed the training questions.
- Anything about fine-tuning a reranker, which was my original question.

## Weak points a reviewer found in my own draft (know them before they ask)

- The tuned embedder vs the LLM translation pipeline on recall@10: the interval for the difference runs from about 15 points worse to about 2 better. "About as often" is a statement about the point estimate only.
- Answer quality: no detectable difference, but the tuned pipeline's fact recall could be about 10 points lower, and the point estimates for fact recall, faithfulness and traps all favour the LLM pipeline. Judge noise is not in the intervals.
- The first-round agent (D) scored nominally higher than the served pipeline and was never re-run with the corrected first stage.
- My own stop rule on the int8 model fired and I chose to continue after asking; the plain search (no reranker) serves the int8 ordering, which is 0.055 MRR worse than fp32.
- Citation precision is only about 0.58: part of it is citing every article the model used, but I did not separate that from wrong citations.
- The latency comparison comes from two runs made at the same time on one laptop.

## Questions to expect, and honest answers

1. **"Isn't your benchmark too easy?"** For rerankers, yes: the questions were written from the article text, so once the right article is among the candidates, any decent reranker finds it. That is why I tuned the first stage, where the benchmark does separate systems.
2. **"The same LLM family wrote the training and the test questions."** Yes. I used six deliberately different styles and a different prompt, and the held-out validation set shows the same shape as the benchmark, but that cannot rule it out. Real user questions would.
3. **"Why trust your judge?"** I don't trust small differences. It agreed with one non-expert's labels on 81% of 63 facts; I use one judge for all systems and report differences as unclaimable when the interval includes zero.
4. **"What was the bug?"** Padding tokens were averaged into the vectors of every batched passage, so the index was subtly degraded while single queries were fine. I found it when the same untrained model scored much better in a separate script. The untuned hybrid went from 0.49 to 0.55 recall@10; I added an errata section and marked what I could not rerun.
5. **"Why int8, if it loses ranking quality?"** It is 118 MB instead of 470 MB and runs on a CPU. On test it loses 0.055 MRR (claimable) and 0.02 recall@10 (not claimable) against fp32; with a reranker on top the pipeline's MRR was 0.84 against 0.87 for the full-precision pipeline, overlapping. I asked before keeping it.
6. **"What is the cost?"** About 0.0024 USD and 5.4 s per question for the full pipeline; the retrieval stage alone is free and about 0.05 to 0.1 s on a laptop CPU. Phase 3 cost 2.16 USD in API calls, Phase 4 about 4 USD.
7. **"Why no live demo?"** Hugging Face only runs a server-side app on a paid plan. I did not want to pay or put a paid model call behind a public endpoint, so there is a static page of recorded answers and the demo and MCP server run locally.
8. **"What would you do with another month?"** Real questions and an expert review of the gold answers, harder negatives from neighbouring articles, and fine-tune the reranker I started with.

## Numbers to have ready (test set, 164 answerable questions, 95% intervals)

- Recall@10: untuned hybrid 0.55 [0.46, 0.63]; tuned embedder alone 0.81 [0.71, 0.90]; tuned plus reranker 0.89 [0.80, 0.97]; LLM translation 0.87 [0.79, 0.94].
- English recall@10: 0.24 untuned, 0.80 tuned.
- Full pipeline (P2): recall@10 0.86, fact recall 0.80 [0.71, 0.89], faithfulness 0.84, traps 9 of 15, 0.0024 USD, 5.4 s.
- Not distinguishable: tuned embedder alone against LLM translation on recall@10 (-0.064 [-0.149, +0.016]); P2 against P3 on fact recall (-0.021 [-0.099, +0.051]).
