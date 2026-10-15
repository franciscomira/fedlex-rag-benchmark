# Fedlex Cross-Lingual RAG Benchmark

How well do frontier LLMs answer questions about Swiss federal law across languages, and can a small fine-tuned reranker match an LLM reranker at a fraction of the cost?

Swiss federal law (Fedlex) is published officially in German, French and Italian as parallel texts. This makes a controlled experiment possible: the content stays the same and only the language changes.

## What this project measures

- **Cross-lingual retrieval:** the question in one language, the law in another.
- **Grounded answers:** Claude vs. GPT vs. Gemini on cited, article-level answers.
- **Faithfulness:** whether citations support the claims, and whether the model refuses when the answer is not in the sources.
- **Cost and latency:** a fine-tuned small reranker vs. a frontier LLM reranker, per query.

## Status

Phase 0: scope and data, done. See [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) and [docs/phase0_report.md](docs/phase0_report.md).

## Reproduce the corpus

Requires [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run python -m fedlex_bench.chunk      # download from Fedlex, write data/processed/articles.parquet
uv run python -m fedlex_bench.alignment  # DE/FR/IT alignment report in docs/phase0_alignment.txt
uv run pytest
```

`data/manifest.json` pins the Fedlex consolidation, source URLs and file checksums.

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

What each one measured, and what this project adds: [docs/related_work.md](docs/related_work.md).

## Licence

Code: [MIT](LICENSE). The licence covers the code only, not the Fedlex texts.

Data: the statute texts come from Fedlex (Swiss Federal Chancellery) open data via the SPARQL endpoint and filestore. Fedlex permits reuse for commercial and non-commercial purposes under the conditions of Art. 49 para. 1 of the Publikationsverordnung (SR 170.512.1), as described at <https://www.fedlex.admin.ch/de/broadcasters> and <https://fedlex.data.admin.ch/de-CH/home/intro>. Key conditions: the content must not be altered, it must be visibly distinct from any commentary, and it must not suggest an official publication.

**This is not an official publication. Only the publication by the Federal Chancellery is authoritative.** (German original: "Dies ist keine amtliche Veröffentlichung. Massgebend ist allein die Veröffentlichung durch die Bundeskanzlei.")

Processing notes: article text is taken from the published Akoma Ntoso XML, with footnotes moved into a separate `footnotes` column rather than dropped. Repealed articles are kept with empty text and a flag. Article ids and chunk boundaries are derived by this project. Raw Fedlex files and the processed Parquet are not committed to this repository.

The Swiss Case Law Open Dataset on Hugging Face (CC0) was checked and contains court decisions only, so it is not used for statutes.
