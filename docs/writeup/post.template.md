# Fine-tuning a small embedding model for Swiss federal law: what worked, what didn't, and a bug in my own numbers

*Francisco Mira. Code, benchmark, model and every number below: [github.com/franciscomira/fedlex-rag-benchmark](https://github.com/franciscomira/fedlex-rag-benchmark). The model is on [Hugging Face](https://huggingface.co/franciscomira/fedlex-e5-small).*

I wanted to answer one question with measurements instead of opinions: **can a small fine-tuned model replace an LLM call in a cross-lingual legal retrieval pipeline, at lower cost and latency?** The answer I got is "mostly yes for finding the right article, and I can't tell it apart from the LLM pipeline on answer quality", with a list of reasons not to trust that too far. This post is the argument, the numbers, and the places where I was wrong.

## The setting

Swiss federal law is published officially in German, French and Italian, as parallel texts, and the same article keeps the same number in all three. That makes a clean experiment: the content stays fixed and only the language changes. I took {{f corpus_acts}} federal acts (the Code of Obligations, the Civil Code and six others), {{f corpus_articles}} articles in three languages, and wrote a benchmark of {{f benchmark_items}} items: questions in English, German, French and Italian, each with the one article (sometimes two) that answers it, plus questions that the law does not answer, where the right behaviour is to say so.

English questions are the hard case on purpose. No article is in English, so an English question over German, French and Italian text is cross-lingual in every configuration.

Two rules shaped everything. First, **measure before changing**: every change to the pipeline is judged by an evaluation harness, never by looking at a few outputs. Second, **the model never touches what must be exact**: article ids, SR numbers and citations are produced and validated by code. The model may rank passages and draft an answer; code checks every cited id against the articles that were actually retrieved, and an id the model invents is dropped and counted.

I split the benchmark into a development part and a test part. I chose configurations on the development part, froze them, and ran each frozen configuration on the test part once. The test set has {{f test_answerable}} answerable questions in about {{f test_groups}} independent groups (a question and its translations count once), so intervals are wide. Every number below has a 95% bootstrap interval that resamples those groups, and I say which differences I cannot claim.

## Where this sits

Two pieces of earlier work are close. SCALE (Stern et al.) is a multi-task Swiss legal benchmark whose retrieval task takes the facts of a court case as the query and looks for the laws it cites. ZHAW's JuRAG tested twelve multilingual embedding models on court decisions, with twenty-six questions in four languages, and found the embedding model to be the biggest lever. My setting differs: short natural-language questions answered by one statute article, cited answers scored for faithfulness and for refusing when the law is silent, and cost and latency per query. Their finding about embeddings fits what I saw. I read both from their published task descriptions and blog post, not their results tables, and I don't compare numbers with theirs.

## What I thought the problem was, and what it was

I started with a different question: whether a small fine-tuned *reranker* could match a frontier LLM reranker. The first baseline made that question uninteresting. On the development questions, a small and cheap LLM reranker and a much larger one brought the right article into the top ten equally often, at prices that differed by about twenty times. Once the right article is among the candidates, any decent reranker seems to find it. The benchmark can't tell rerankers apart, partly because the questions were written from the article text.

What the baseline did show is that **the first search step is the bottleneck**. A plain hybrid search (keywords plus a multilingual embedding model) found the right article in the top ten for only {{v A2 recall@10 en}} of English questions, against {{v A2 recall@10 de}} for German (measured with an embedder bug that I describe below; corrected, English is {{v A3 recall@10 en}}). Asking a small LLM to translate the question into German, French and Italian before searching lifted English to {{v C2 recall@10 en}}. That works, but it costs an LLM call and several seconds per question.

So I changed the question: **can a small fine-tuned embedding model close the gap that LLM translation closes, with no LLM call at query time?** I did not fine-tune a reranker. The original question stays open.

## A bug in my own measurements

Before the fine-tuning results, the correction, because it changes how much of the earlier numbers you should believe.

My embedding code averaged padding tokens into the vectors of texts that were embedded in a batch. A question embedded alone was right; every passage of the index was embedded in a batch and was subtly wrong. I found it by accident: the same untrained model scored much better in a separate validation script than in my pipeline. After the fix, the untuned hybrid search on the test questions went from {{f bug_old_recall}} to {{f bug_new_recall}} recall@10 (paired difference {{d A3 A2 recall@10}}), and English from {{v A2 recall@10 en}} to {{v A3 recall@10 en}}. Every retrieval number from my first round that used embeddings had been too low, and I had written conclusions on top of them. I re-ran what I could, added an errata section to the earlier report, and marked what I could not re-run: the first round's answer-quality numbers were measured on the broken first stage, and the agent experiment was never repeated.

## Fine-tuning

I fine-tuned `multilingual-e5-small` ({{f model_params}} parameters). For training data I had a small LLM write four questions for each of {{f pool_articles}} articles that are not in the benchmark: two in English and two in German, French or Italian, in six deliberately different styles (a lay description of a situation, legal wording, keywords, a third party asking, and so on). That gave {{f train_questions}} training questions, plus a validation split, for {{f question_generation_usd}} USD. I removed every development and test answer article, every article next to one, and near-duplicates of them, and wrote tests that fail if any of them leaks in.

Training used a standard in-batch ranking loss, three epochs, on a free Kaggle GPU in {{f training_minutes}} minutes. The whole phase, with data generation, audits and test runs, cost {{f phase3_spend_usd}} USD in API calls. Two findings I did not expect: {{f quarter_data_note}} of the data gives most of the gain, and mining hard negatives, the textbook improvement, gave no claimable gain ({{f hardneg_gain_dev}} recall@10 on {{f dev_items}} development questions) and made each training step {{f hardneg_slowdown}} slower, so I dropped it.

## Results on the test set

Retrieval only, {{n T1}} answerable questions, each frozen configuration run once.

{{table phase3}}

{{figure fig_recall}}

- The tuned embedder alone (T1: no LLM call, {{lat T1}} median) reaches recall@10 of {{v T1 recall@10}}, against {{v A3 recall@10}} for the untuned hybrid search with the corrected embedder. The difference is {{d T1 A3 recall@10}}: {{claim T1 A3 recall@10}}.
- The pipeline that asks an LLM to translate the question first (C′) reaches {{v Cp recall@10}} at {{cost Cp}} USD and {{lat Cp}} per question. T1 minus C′ is {{d T1 Cp recall@10}} on recall@10 ({{claim T1 Cp recall@10}}), so I cannot say which is better at *finding* the article within the top ten. But the LLM pipeline puts the right article first more often: on MRR, T1 minus C′ is {{d T1 Cp mrr}} ({{claim T1 Cp mrr}}).
- With a cheap LLM reranker on top (T2), recall@10 is {{v T2 recall@10}} and the English questions are at {{v T2 recall@10 en}}, against {{v Cp recall@10 en}} for the translation pipeline. T2 minus C′ is {{d T2 Cp recall@10}} ({{claim T2 Cp recall@10}}): not distinguishable, at {{lat T2}} against {{lat Cp}}.

{{figure fig_english}}

The English cell is where the fine-tuning pays off: {{v A3 recall@10 en}} for the untuned search and {{v T1 recall@10 en}} with the tuned model, with no LLM call.

{{table by_language}}

Cells hold about forty questions each, so read the per-language numbers as directions.

## Do the answers get better?

Retrieval metrics say how often the right article is found. They do not say whether the answers are good, so I also measured answers with the pipeline that I would actually serve (P2): the tuned embedder as an int8 file ({{f int8_mb}} MB instead of {{f fp32_mb}} MB, runs on a CPU), a small LLM reranker over the best candidates, and a drafted answer whose citations are validated by code, with an explicit "not in the sources" path. Answers were scored by a separate judge model from another family, which agreed with my hand labels {{f judge_agreement}} of the time on {{f calib_facts}} facts, which is good enough to compare systems and not good enough to read small differences.

{{table phase4}}

- Fact recall, the share of the gold facts that the answer states, is {{v P2 fact_recall}} for the pipeline I serve (P2) and {{v P3 fact_recall}} for the LLM-translation pipeline with the same answer step (P3). P2 minus P3 is {{d P2 P3 fact_recall}}: {{claim P2 P3 fact_recall}}. Recall@10 and faithfulness are likewise not distinguishable.
- P2 costs {{cost P2}} USD and takes {{lat P2}} per question, against {{cost P3}} USD and {{lat P3}} for P3: {{pct_cheaper P2 P3}} cheaper and {{x_faster P2 P3}} faster, both clearly.
- The int8 file is not free: against the full-precision model, MRR on the test set is {{d P1 T1 mrr}} ({{claim P1 T1 mrr}}) and recall@10 is {{d P1 T1 recall@10}} ({{claim P1 T1 recall@10}}). I kept it because a reranker sits on top in the pipeline I serve.
- Faithfulness stays near {{v P2 faithfulness}}: about one claim in six is not supported by the cited text, whatever the retrieval. On the trap questions, which the law does not answer, P2 handled {{traps P2}} correctly and P3 {{traps P3}}. That is too few questions to rank anything, and it is still the weakest part of every system I built.

{{figure fig_cost_latency}}

So the honest summary is: **a fine-tuned small model, trained on a free GPU for a phase that cost about two dollars in API calls, finds the right article in the top ten about as often as an LLM translation pipeline ({{m T1 recall@10}} against {{m Cp recall@10}}, a difference I cannot resolve either way), and a pipeline built on it answers as well as that LLM pipeline at lower cost and latency.** "Answers as well as" means "I found no difference larger than the noise", with an interval of roughly plus or minus eight points on fact recall.

## What I would not claim

- **The benchmark rewards what the model learned.** Every benchmark question was written by an LLM from the text of one article, and so were my training questions, by the same family of model. Part of the gain may be the model learning that writing style rather than cross-lingual retrieval. Real users' questions would be the real test, and I have none.
- **No expert has looked at any of it.** Nobody on the project reads German, French and Italian well enough to have reviewed the training questions, and no Swiss-law expert reviewed the benchmark, the gold facts or any answer. The single human reviewer reads English only and has no legal training.
- **The test set is small.** About {{f test_groups}} independent questions. Only large differences are claimable; I report the ones that are not as "not distinguishable", never as ties.
- **I changed several things at once.** Compared with my first-round pipeline, the retrieval stage differs in four ways at once: fine-tuning, the padding fix, no keyword search, and int8. The improvement in answer quality over my first-round system is real in the data and I can't say which change produced it.
- **Parts of the first round were never repeated:** the agent system and the trap results with the corrected first stage. They are marked as such on the results page.
- **The leakage check has a known gap.** I excluded near-duplicates of the test articles symmetrically, so a short training article that is contained in a long test article can slip through. I measured the effect: the gain of the tuned model over the untuned one holds on the test questions that are not touched by such articles, but the tuned model's absolute level on the touched ones may be flattered.
- **A reranker was never fine-tuned**, which was my original question.

## What I would do next

Put real questions from people who work with Swiss law into the test set, and have a lawyer who reads German, French and Italian review the gold answers and a sample of the training questions. Build harder negatives from neighbouring articles, because this benchmark is easy for rerankers by construction. And finally fine-tune the reranker I started with, now that the first stage is no longer the bottleneck.

## Try it, check it, break it

The [results page](https://franciscomira.github.io/fedlex-rag-benchmark/) rebuilds every table from the committed run folders with one command, and says for each number whether reproducing it needs API keys or a GPU. There is [a page of ten recorded answers](https://franciscomira.github.io/fedlex-rag-benchmark/examples.html), all ten kept, including one where the system refused a question the law answers in part. There is a {{f mcp_tools}}-tool MCP server for search, exact article lookup and cited answers; several Fedlex MCP servers already exist, and what this one adds is retrieval measured on a benchmark and citations validated by code, over only {{f corpus_acts}} acts. There is no live demo: Hugging Face only runs a server-side app (which a search needs) on a paid plan, so the search demo runs locally.

The last phase, which covers the measurements of the pipeline above and shipping it, cost {{f phase4_month_usd}} USD of API calls.
