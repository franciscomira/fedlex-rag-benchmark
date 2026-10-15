# Fedlex Cross-Lingual RAG Benchmark

How well do frontier LLMs answer questions about Swiss federal law across languages, and can a small fine-tuned reranker match an LLM reranker at a fraction of the cost?

Swiss federal law (Fedlex) is published officially in German, French and Italian as parallel texts. This makes a controlled experiment possible: the content stays the same and only the language changes.

## What this project measures

- **Cross-lingual retrieval:** the question in one language, the law in another.
- **Grounded answers:** Claude vs. GPT vs. Gemini on cited, article-level answers.
- **Faithfulness:** whether citations support the claims, and whether the model refuses when the answer is not in the sources.
- **Cost and latency:** a fine-tuned small reranker vs. a frontier LLM reranker, per query.

## Status

Phase 0: scope and data. See [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md).

## Layout

```
data/raw/          source downloads (not committed)
data/processed/    aligned, chunked articles
benchmark/         questions, gold answers, gold articles
src/fedlex_bench/  retrieval, reranking, answering
eval/              harness and results
finetune/          reranker fine-tuning
notebooks/         exploration
docs/              plan and write-up
```

## Related work

- [SCALE: One Law, Many Languages](https://arxiv.org/abs/2306.09237)
- [ZHAW JuRAG: cross-lingual retrieval over Swiss court decisions](https://blog.zhaw.ch/artificial-intelligence/2026/06/02/can-ai-bridge-switzerlands-legal-language-gap/)
- [SwiLTra-Bench](https://arxiv.org/abs/2503.01372v1)

## Licence

Code: TBD. Data: see source licences (Fedlex terms, dataset licences), to be confirmed in Phase 0.
