# Fedlex Cross-Lingual RAG Benchmark

How well do frontier LLMs answer questions about Swiss federal law across languages, and can a small fine-tuned reranker match an LLM reranker at a fraction of the cost?

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
| 3 Fine-tuned reranker | next |
| 4 Ship, 5 write-up | planned. [Project plan](docs/PROJECT_PLAN.md) |

### Phase 2 results in one table

Test split, 164 answerable items, Claude Haiku 5.5 everywhere, judge gpt-6.1-sol, 95% group-bootstrap intervals. Each configuration was run once on test, after being frozen on dev.

| System | Recall@10 | Fact recall | Cost per question |
| --- | --- | --- | --- |
| Hybrid retrieval (BM25 + multilingual-e5-small), no model | 0.49 [0.40, 0.58] | n/a | 0 |
| + LLM rerank + cited answer (single-shot) | 0.65 [0.56, 0.73] | 0.61 [0.52, 0.69] | $0.0021 |
| Translate the question into DE/FR/IT, then retrieve per language | 0.81 [0.73, 0.89] | 0.76 [0.68, 0.84] | $0.0028 |
| Google ADK agent (planner, per-language retrievers, citation checker) | 0.92 [0.87, 0.97] | 0.81 [0.75, 0.86] | $0.0026 |

The gains of the last two rows over single-shot are concentrated in English questions; the agent does not beat the simple translation control on answer quality (+0.04, interval -0.05 to +0.14). Limitations, including a ceiling effect of this benchmark for strong rerankers and a judge that agrees 81% with the hand labels, are in the [Phase 2 report](docs/phase2_report.md).

## Reproduce

Requires [uv](https://docs.astral.sh/uv/). API keys go in `.env` (copy `.env.example`); nothing in this repository needs a key to read or to run the tests.

```
uv sync
uv run pytest                                        # offline, uses fake model clients

# Phase 0: the corpus (downloads from Fedlex, writes data/processed/articles.parquet)
uv run python -m fedlex_bench.chunk
uv run python -m fedlex_bench.alignment              # DE/FR/IT alignment report

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
                      agent/ (Google ADK); eval/ (metrics, judge, runner, CLI)
tests/                pytest, offline
eval/                 results, frozen test configurations, test-run log, price table, judge calibration
finetune/             Phase 3 (empty so far)
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

Processing notes: article text is taken from the published Akoma Ntoso XML, with footnotes moved into a separate `footnotes` column rather than dropped. Repealed articles are kept with empty text and a flag. Article ids and chunk boundaries are derived by this project. Raw Fedlex files, the processed Parquet and the embedding indexes are not committed to this repository. The committed `eval/results/` folders hold model outputs (answers, traces, scores) and article ids, not article texts. The benchmark questions and gold facts were written by language models and only partly reviewed by a human who is not a lawyer; this is not legal advice.

The Swiss Case Law Open Dataset on Hugging Face (CC0) was checked and contains court decisions only, so it is not used for statutes.
