# Fedlex Cross-Lingual RAG Benchmark

How well do frontier LLMs answer questions about Swiss federal law across languages, and can a small fine-tuned model close the cross-lingual retrieval gap that an LLM closes, at a fraction of the cost?

Swiss federal law (Fedlex) is published officially in German, French and Italian as parallel texts. This makes a controlled experiment possible: the content stays the same and only the language changes.

## What this project measures

- **Cross-lingual retrieval:** the question in one language (English, German, French or Italian), the law in another.
- **Grounded answers:** cited, article-level answers with a code-validated citation list and an explicit "not in the sources" path.
- **Faithfulness:** whether citations support the claims, and whether the system refuses when the answer is not in the sources.
- **Cost and latency:** per query, from a dated price table.

## Status

| Phase | State |
| --- | --- |
| 0 Scope and data | done: 8 acts, 4,073 articles aligned in DE/FR/IT (12,219 rows). [Report](docs/phase0_report.md) |
| 1 Benchmark | done: `benchmark/v1`, 242 items (150 answerable questions + 20 traps, plus 72 translations). [Report](docs/phase1_report.md), spec [benchmark_spec.md](docs/benchmark_spec.md) |
| 2 Baseline harness, providers, agent | done: hybrid retrieval, LLM reranker, three providers, ADK agent, evaluation harness. [Report](docs/phase2_report.md), spec [eval_spec.md](docs/eval_spec.md) |
| 3 Fine-tuning | done: a fine-tuned multilingual-e5-small reaches test recall@10 0.81 with no LLM call; retrieval only. [Report](docs/phase3_report.md), [data](data/finetune/README.md), model: [franciscomira/fedlex-e5-small on Hugging Face](https://huggingface.co/franciscomira/fedlex-e5-small) ([model card source](docs/model_card.md)). Found and corrected an embedder bug that affected the Phase 2 retrieval numbers: see the errata in the [Phase 2 report](docs/phase2_report.md) |
| 4 Ship | done except the write-up: the int8 model and the full pipeline were run once each on test (below); [results page](https://franciscomira.github.io/fedlex-rag-benchmark/) live; MCP server and a local retrieval-only demo in the repository; no live demo (see below). Log: [phase4_progress.md](docs/phase4_progress.md) |
| 5 Write-up | planned. [Project plan](docs/PROJECT_PLAN.md) |

### Phase 2 results in one table

> **Errata (2026-10-28).** The retrieval numbers in this table were measured with an embedder that averaged padding tokens into batched vectors, which understated every system that uses embeddings. Retrieval-only reruns with the corrected embedder are in the next section and in the [Phase 2 report](docs/phase2_report.md); the fact-recall numbers were not re-measured. The table is kept as published.

Test split, 164 answerable items, Claude Haiku 5.5 everywhere, judge gpt-6.1-sol, 95% group-bootstrap intervals. Each configuration was run once on test, after being frozen on dev.

| System | Recall@10 | Fact recall | Cost per question |
| --- | --- | --- | --- |
| Hybrid retrieval (BM25 + multilingual-e5-small), no model | 0.49 [0.40, 0.58] | n/a | 0 |
| + LLM rerank + cited answer (single-shot) | 0.65 [0.56, 0.73] | 0.61 [0.52, 0.69] | $0.0021 |
| Translate the question into DE/FR/IT, then retrieve per language | 0.81 [0.73, 0.89] | 0.76 [0.68, 0.84] | $0.0028 |
| Google ADK agent (planner, per-language retrievers, citation checker) | 0.92 [0.87, 0.97] | 0.81 [0.75, 0.86] | $0.0026 |

The gains of the last two rows over single-shot are concentrated in English questions; the agent does not beat the simple translation control on answer quality (+0.04, interval -0.05 to +0.14). Limitations, including a ceiling effect of this benchmark for strong rerankers and a judge that agrees 81% with the hand labels, are in the [Phase 2 report](docs/phase2_report.md).

### Phase 3 results in one table

Test split, 164 answerable items, retrieval only (no answer or judge), 95% group-bootstrap intervals; each frozen configuration was run once. Hypotheses were written before the runs; none failed. Details, limitations and the paired differences: [Phase 3 report](docs/phase3_report.md).

| System | Recall@1 | Recall@10 | English recall@10 | Cost per question | Latency p50 |
| --- | --- | --- | --- | --- | --- |
| Hybrid retrieval, corrected embedder, no model | 0.25 | 0.55 [0.46, 0.63] | 0.24 | 0 | 0.1 s |
| **Fine-tuned embedder, no LLM** | 0.48 | **0.81 [0.71, 0.90]** | **0.80** | **0** | **0.1 s** |
| **Fine-tuned embedder + Haiku reranker** | **0.79** | **0.89 [0.80, 0.97]** | 0.89 | $0.0016 | 1.6 s |
| LLM translation + hybrid + Haiku reranker (corrected) | 0.76 | 0.87 [0.79, 0.94] | 0.86 | $0.0024 | 8.0 s |

The tuned embedder alone gains +0.26 recall@10 [+0.17, +0.35] over the untuned hybrid and is within 0.06 of the LLM-translation pipeline on recall@10 (difference not claimable), but orders worse (MRR 0.63 against 0.84). With a cheap reranker on top it is not distinguishable from the translation pipeline (+0.015 recall@10 [-0.05, +0.07]) at a fifth of the latency. Training used 11.5k synthetic questions (0.87 USD) and under 15 minutes on a free GPU; hard negatives did not help. The benchmark's questions were written by an LLM from the article text, so these numbers describe finding that article, not answer quality. The weights (fp32 safetensors and an int8 ONNX file) are published at [huggingface.co/franciscomira/fedlex-e5-small](https://huggingface.co/franciscomira/fedlex-e5-small); the training data stays in this repository.

### Phase 4 results: the pipeline that is served

Test split, 164 answerable items and 15 traps, judge gpt-6.1-sol, 95% group-bootstrap intervals; each frozen configuration was run once, hypotheses written before. The served pipeline (P2) is the tuned embedder as an int8 ONNX file (118 MB, CPU), the Haiku reranker over the top 60 of 150 candidates, and a cited answer whose article ids are validated by code, with an explicit "not in the sources" path. The serving code builds the same system the harness measured.

| System | Recall@10 | Fact recall | Faithfulness | Traps correct | Cost per question | Latency p50 |
| --- | --- | --- | --- | --- | --- | --- |
| P1 int8 embedder alone (retrieval only) | 0.79 [0.69, 0.88] | n/a | n/a | n/a | 0 | 0.05 s |
| **P2 served pipeline** (int8 + reranker + cited answer) | **0.86 [0.78, 0.94]** | **0.80 [0.71, 0.89]** | 0.84 [0.80, 0.87] | 9 of 15 | $0.0024 | 5.5 s |
| P3 LLM translation + reranker + cited answer (corrected embedder) | 0.87 [0.79, 0.94] | 0.82 [0.75, 0.89] | 0.85 [0.83, 0.88] | 11 of 15 | $0.0030 | 11.9 s |

P2 and P3 are not distinguishable on recall@10, fact recall or faithfulness (fact recall P2 minus P3 -0.02 [-0.10, +0.05]); P2 is cheaper and about twice as fast, both claimable. The int8 file loses 0.02 recall@10 against fp32 (not claimable) and 0.055 MRR (claimable) on test; with the reranker on top the MRR is 0.84 [0.75, 0.92] for P2 against 0.87 [0.78, 0.95] for the fp32 equivalent (T2, retrieval only), overlapping broadly. Fact recall is higher by +0.20 [+0.10, +0.29] than Phase 2's single-shot system, which used the buggy first stage; the first stage changed in four ways at once (fine-tuning, padding fix, no BM25, int8), so this does not say which helped. Trap counts are too small to rank. Fact recall came out above the range I wrote down beforehand for both P2 and P3 (details in the log). Limits: the questions were written by an LLM from the article text, nobody who reads German, French or Italian reviewed the training questions, no Swiss-law expert reviewed anything, and system D was not re-measured with the corrected embedder.

### Demo, MCP server, results page

- **Results page:** `uv run python -m fedlex_bench.site build` writes `site/index.html` from the committed run folders (every number is read from the harness output). Published at <https://franciscomira.github.io/fedlex-rag-benchmark/> by a GitHub Actions workflow; see [results_page_notes.md](docs/results_page_notes.md).
- **Demo app (local):** `uv run --group serve python -m fedlex_bench.serve.web` serves a retrieval-only search (no LLM, no key, no stored questions) with the three-language texts, plus recorded full-pipeline examples. There is **no live public demo**: Hugging Face only hosts Docker Spaces on a paid plan, and a free host with enough memory was not found. A Docker image of the demo builds and runs ([demo/README.md](demo/README.md)). Ten recorded full-pipeline answers are on the [results site](https://franciscomira.github.io/fedlex-rag-benchmark/examples.html).
- **MCP server:** `uv run --group serve python -m fedlex_bench.serve.mcp_server` (tools `list_acts`, `search_articles`, `get_article`, `answer_with_citations`). Several Fedlex MCP servers exist. What this one adds: retrieval measured on a benchmark with intervals, citations validated by code, the article text in three languages side by side. What it does not: it covers 8 federal acts, not all of Fedlex, and it is not an official source. See [docs/mcp.md](docs/mcp.md).

## Reproduce

Requires [uv](https://docs.astral.sh/uv/). API keys go in `.env` (copy `.env.example`); nothing in this repository needs a key to read or to run the tests.

```
uv sync
uv run pytest                                        # offline, uses fake model clients

# Phase 0: the corpus (downloads from Fedlex, writes data/processed/articles.parquet)
uv run python -m fedlex_bench.chunk
uv run python -m fedlex_bench.alignment              # DE/FR/IT alignment report

# Phase 3: fine-tuning data and checks (training itself runs on a free Kaggle GPU: notebooks/README.md)
uv sync --group finetune
uv run python -m fedlex_bench.finetune.exclusions    # training pool: dev/test gold, neighbours, twins excluded
uv run python -m fedlex_bench.finetune.dataset       # split by article
uv run python -m fedlex_bench.eval report 2026-10-28_test_aec4726ef0 2026-10-28_test_ae60b9e8c1 2026-10-28_test_033e9b80b2

# Phase 4: rebuild the table above from the committed runs (free, no key)
uv run python -m fedlex_bench.eval report 2026-11-03_test_35611dd1aa 2026-11-03_test_d6a30cf69a 2026-11-03_test_716222848a
uv run python -m fedlex_bench.eval compare 2026-11-03_test_d6a30cf69a 2026-11-03_test_716222848a
uv run python -m fedlex_bench.site build             # the results page

# Phase 2: indexes and evaluation
uv run python -m fedlex_bench.retrieval build        # embeds the corpus (CPU, about 15 minutes)
uv run python -m fedlex_bench.eval run --split dev --method hybrid --config all_lang
uv run python -m fedlex_bench.eval report <run ids>  # results table with 95% intervals
uv run python -m fedlex_bench.eval compare <run a> <run b>   # paired comparison
```

The numbers in the Phase 2 report are in `eval/results/<run id>/` (`config.json`, `per_item.jsonl`, `metrics.json`); `uv run python -m fedlex_bench.eval report 2026-10-18_test_78df4b6f59 2026-10-18_test_be2a67022b 2026-10-18_test_89c5213cec 2026-10-18_test_89a1a1a0ad` rebuilds the table from them. Re-running a system from scratch needs the provider keys and costs money; providers are not bit-deterministic, so a fresh run will differ slightly. See [eval/README.md](eval/README.md) for the test-split rules and the cache.

`data/manifest.json` pins the Fedlex consolidation, source URLs and file checksums. `benchmark/v1/SHA256SUMS` pins the benchmark files.

## Layout

```
data/raw/             source downloads (not committed)
data/processed/       aligned, chunked articles and indexes (not committed)
benchmark/v1/         frozen benchmark (read-only); earlier folders are the Phase 1 build trail
src/fedlex_bench/     ingest, chunk, alignment; benchmark/ (build tools); retrieval/; llm/ and answer/;
                      agent/ (Google ADK); eval/ (metrics, judge, runner, CLI); finetune/ (Phase 3);
                      serve/ (serving pipeline, demo app, MCP server); site/ (results page builder)
site/                 the built results page
demo/                 Dockerfile and notes for a Hugging Face Space, recorded examples
tests/                pytest, offline
eval/                 results, frozen test configurations, test-run log, price table, judge calibration
data/finetune/       Phase 3 training data (pool, synthetic questions, pairs, audit) and its README
notebooks/            Kaggle fine-tuning notebook, kit builder, set-up guide
models/               trained checkpoints (not committed)
docs/                 plan, specs, phase reports and progress logs
.claude/agents/       role files of the subagents used to build the project
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

Processing notes: article text is taken from the published Akoma Ntoso XML, with footnotes moved into a separate `footnotes` column rather than dropped. Repealed articles are kept with empty text and a flag. Article ids and chunk boundaries are derived by this project. Raw Fedlex files, the processed Parquet and the embedding indexes are not committed to this repository. The committed `eval/results/` folders hold model outputs (answers, traces, scores) and article ids, not article texts. The benchmark questions and gold facts were written by language models and only partly reviewed by a human who is not a lawyer; this is not legal advice. The committed Phase 3 training questions (`data/finetune/`) are also model-written, contain no article text, and were audited by a second model family only; nobody who reads German, French and Italian reviewed them.

The Swiss Case Law Open Dataset on Hugging Face (CC0) was checked and contains court decisions only, so it is not used for statutes.
