# Related work: what was measured and what was not

Read on 2026-10-15 from the arXiv HTML version of SCALE and the ZHAW blog post. Summaries are from those pages only. The SCALE results section (6.3) was not read, so SCALE retrieval numbers are not quoted here. Re-read it before the write-up.

## SCALE: One Law, Many Languages (Stern et al., arXiv:2306.09237)

[Paper](https://arxiv.org/abs/2306.09237). CC BY-SA.

**Measured**
- A multi-task Swiss legal benchmark in five languages (German, French, Italian, Romansh, English). Tasks include Information Retrieval, Court View Generation, Leading Decision Summarization, Citation Extraction and eight classification tasks.
- The retrieval task (Doc2Doc IR, section 4.5): the query is the facts of a Federal Supreme Court case, used as a proxy for a lawyer's appeal. The target is the laws and leading decisions cited in the considerations. The corpus is Swiss legislation and leading decisions (about 10K documents). Queries are in DE/FR/IT and the laws are in DE/FR/IT, so cross-lingual pairs occur.
- Metrics: NDCG and capped Recall@k. Models: BM25 and multilingual SBERT (also with hard negatives). Cross-encoders were left out because of cost on long texts.
- Stated limitations: BM25 handles context and multiple languages poorly, SBERT degrades on long inputs because of truncation, and performance is expected to fall as the corpus grows.

**Not measured** (as far as the pages read show)
- Short natural-language questions answered from a single statute article. Queries are long case facts.
- Answer generation with article-level citations, citation faithfulness, and refusing when the answer is not in the sources.
- Cross-encoder or LLM rerankers, and fine-tuning a small reranker against an LLM reranker.
- Frontier LLMs compared head to head.
- Cost or latency per query.

## ZHAW JuRAG (blog post, 2026-06-02)

[Post](https://blog.zhaw.ch/artificial-intelligence/2026/06/02/can-ai-bridge-switzerlands-legal-language-gap/).

**Measured**
- Retrieval over 165,556 Federal Supreme Court decisions (2000 to 2025), split into about 2.4 million passages.
- 26 legal questions, each in German, French, Italian and English (104 query instances). Ground truth is the relevant decision and the passage ("Erwägungen") with the reasoning.
- Compared keyword search, 12 multilingual embedding models, reranking as a second stage, hybrid search, language deconfounding and LLM query paraphrasing.
- Main metric: the correct passage in the top 10. The best configuration reached 73%, against 19% for keyword search. The embedding model was the biggest lever. Reranking added a useful second-stage gain, hybrid search helped only marginally, and LLM paraphrasing made retrieval worse. German and English queries did better than French and Italian.
- Stated limitations: the collection and evaluation set are skewed to German, French and Italian evaluation needs improvement, and no test with practising lawyers yet.

**Not measured** (as far as the page shows)
- Statutes: the corpus is court decisions.
- Answer faithfulness or citation checking.
- Cost per query.
- Which reranker was used and how it compares to an LLM reranker is not stated on the page. The page reports reranking as a second-stage gain only.
- Only 26 questions, so differences between configurations may be noisy.

## Where this project differs

| Axis | SCALE | JuRAG | This project |
| --- | --- | --- | --- |
| Corpus | Statutes and leading decisions | Court decisions | Statutes, article level (8 acts) |
| Query | Case facts | 26 lawyer questions | About 150 questions, each answerable from one article |
| Output scored | Retrieved citations | Retrieved passages | Retrieved articles and cited answers |
| Citation faithfulness and refusal | No | No | Yes |
| Rerankers | No cross-encoders | Reranking as a stage | Fine-tuned small reranker against an LLM reranker |
| Frontier LLMs head to head | No | No | Claude, GPT, Gemini |
| Cost per query | No | No | Yes |

The overlap to cite openly: both prior works show cross-lingual retrieval over Swiss law is hard and that the embedding model matters. This project does not repeat that finding as its main claim. The open question it adds is whether a small tuned reranker matches an LLM reranker on cost and quality.
