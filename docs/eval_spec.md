# Evaluation specification (Phase 2, draft for approval)

Status: draft written 2026-10-17, before any harness code. Benchmark: `benchmark/v1/` (read-only, tag `benchmark-v1`). Nothing here may be changed after the first `test.jsonl` run without a dated amendment at the bottom.

## 0. Ground rules

- Iterate on `dev.jsonl` only. `test.jsonl` is run once per frozen configuration (section 11) and every test run is appended to `eval/test_runs.log`.
- Every reported number is rebuilt from a cached run with one command: `uv run python -m fedlex_bench.eval report <run_id>`.
- Budget: 25 USD per month, hard cap in `config.toml`. A run projected above 5 USD (20%) needs explicit approval. The harness refuses to start a paid run when the projection exceeds the remaining budget.
- Code handles everything exact: ids, citations, validation, scoring arithmetic. Models only draft answers, rerank, and judge.

## 1. Units and splits

- Item = one line of the benchmark file. Answerable items (`qtype` single or cross_ref) feed retrieval and answer metrics. Traps (`is_trap`) feed trap accuracy only.
- Dev: 63 items = 40 answerable originals + 5 traps + 18 translations. Test: 179 = 110 + 15 + 54.
- Translations share gold with a parent. For statistics, the unit of resampling is the **group** (parent id; an original without translations is its own group), so a question and its translations are never treated as independent.
- Headline numbers use all answerable items. Every table also gives the originals-only number, so translations cannot inflate a result.

## 2. Retrieval metrics

Computed on article ids (`<SR>:<article>`), the same id in DE, FR and IT.

- The ranked list the retriever returns is first mapped to article ids and de-duplicated (first occurrence wins). `k` counts unique articles, never language copies. A hit is any language version of a gold article.
- **recall@k** for k = 1, 5, 10: fraction of the item's gold articles in the top k. For single items this is 0/1. For cross_ref items it is 0, 0.5 or 1; the table also reports "all gold in top k".
- **MRR**: reciprocal rank of the first gold article, 0 if absent from the returned list (list length 20).
- Reported **per query language** (EN, DE, FR, IT), per question type, and per **index configuration**:
  - `same_lang`: DE/FR/IT query against the index of its own language. Not defined for EN.
  - `single_lang_de`, `single_lang_fr`, `single_lang_it`: any query against one fixed language index (cross-lingual for three of four query languages).
  - `all_lang`: one index with all three language versions.
- English queries are cross-lingual in every configuration. The same-language control is 8 to 11 items per language on dev (26 on all of v1): report it, do not draw conclusions from it.
- Traps are excluded from retrieval metrics.

## 3. Answer metrics

The answerer returns a structured object, parsed and validated by code (invalid output is retried once, then counted as a failure and reported):

```json
{"status": "answered" | "refused",
 "answer": "text in the question language",
 "citations": ["220:257d", "..."],
 "pointers": ["281.1:16"]}
```

- `citations` = articles the answer relies on. `pointers` = articles named only as "where to look"; used on refusals. The model picks ids from the retrieved candidate list given in the prompt; code checks each id against that list. An id outside the list is **invalid**: it is dropped, counted as `n_invalid_citations`, and reported as a failure rate. Article text in the output (quotes) is never trusted as a citation.
- **Fact recall** (judge, section 5): per gold fact, does the answer state it (supported / not). Item score = supported facts / all facts. Answerable items only. A refusal scores 0.
- **Citation precision** = |valid citations ∩ gold| / |valid citations| (undefined and counted 0 if there are no citations on an answered item). **Citation recall** = |valid citations ∩ gold| / |gold|. Pure code, article level, language independent.
- **Faithfulness** (judge): the judge sees the answer and the text of the **cited** articles (all language versions returned by retrieval for those ids) and lists the answer's factual claims with a supported / unsupported flag. Item score = supported / all claims. Refusals and answers with no claims are excluded from the faithfulness mean (count reported).
- **Over-refusal rate** = share of answerable items with `status == "refused"`.
- **Trap accuracy**: section 4.
- **Answer language**: the answer is requested in the question language. Judging works across languages because gold facts are English.

## 4. Trap rule (closes the open point from Phase 1)

A trap item is **correct** iff all three hold:

1. `status == "refused"`;
2. `citations` is empty (an article presented as supporting an answer fails the item);
3. the judge finds that the `answer` text does not assert the specific information the question asks for (it is told the `trap_reason`, and checks for a substantive answer slipped into the refusal text).

`pointers` are allowed and not penalised, because a good refusal can say "article X of the Federal Act says the Federal Council sets the tariff". They are reported separately (average pointers per refusal) so a system cannot hide behind them: if the pointer is the closest article recorded in `critic_notes`, that is noted in the item table, not scored.

Test cases (written as unit tests before the scorer):

| Case | status | citations | pointers | answer text | Result |
| --- | --- | --- | --- | --- | --- |
| clean refusal | refused | [] | [] | "The sources do not answer this." | correct |
| refusal with pointer | refused | [] | ["281.1:16"] | "Not stated; art. 16 says the Federal Council sets the tariff." | correct |
| refusal that leaks the answer | refused | [] | [] | "Not in the law, but the fee is 300 CHF." | wrong (criterion 3) |
| answered with an article | answered | ["220:1"] | [] | any | wrong |
| answered, no citation | answered | [] | [] | any | wrong |
| refused but cites | refused | ["220:1"] | [] | "Not answered." | wrong (criterion 2) |
| malformed output | parse failure | | | | wrong, counted as failure |

Dev has 5 traps and test 15: trap accuracy is given as counts (k of n) with the exact interval, never as a headline percentage claim.

## 5. Judge

- **Roles**: (a) fact judge, (b) faithfulness judge, (c) trap-leak judge. One prompt each, stored in `src/fedlex_bench/eval/prompts/`, hashed into `config.json`. Temperature 0, JSON output, the model id pinned with its date at the first run and never changed within a phase (a change means a new calibration and rerun).
- **Family rule**: the judge is from a different family than the answerer. Primary judge: a small GPT-class model. For GPT answers the judge is a small Gemini-class model. The exact model ids are chosen by the llm-integrator from the providers' current docs on the day of pinning; the table of ids and prices with the date lives in `eval/prices.json`.
- **Self-preference check**: on a fixed random sample of 40 dev answers from each answerer, score with both judges; report agreement per answerer family and the mean score gap (same-family judge minus other-family judge). A gap whose bootstrap CI excludes 0 is reported next to every comparison between families.
- **Calibration**: I draw 30 dev answers (seed 20261017, stratified: about 20 answerable across query languages and fact counts, 5 traps, 5 over-refusal or no-citation cases) from the first cheap single-shot run. The user labels each gold fact supported / not supported and each trap correct / wrong. **The user reads English only**, so the sheet shows the answer plus a machine English gloss (labelled as such, produced by the pinned judge model), next to the English gold facts. This is a limitation: the user's labels test fact matching, not wording nuances in German, French or Italian. Agreement is reported as raw agreement and Cohen's kappa, per judge role. **Stop and ask if raw agreement is below 80%.**
- Calibration labels are never used to tune the judge prompt on the same 30 answers after the first agreement number; if the prompt changes, new answers are drawn.
- The judge never sees which system or model produced an answer.

## 6. Cost and latency

- Tokens (input, cached input, output) are logged per call. Cost per query = sum of token counts times the unit prices in `eval/prices.json` (dated table, price source URL, date retrieved). Reports state the price-table date.
- Cost per query counts the whole pipeline: reranker calls, answerer calls, agent steps. Judge cost is reported separately (it is evaluation overhead, not system cost).
- Latency = wall-clock seconds from query to final answer, p50 and p95 per configuration. Responses are cached (key: provider, model id, prompt hash, params, input hash); a cache hit replays the stored latency and cost, so reports are stable. Latency is a measurement from one machine and network on the recorded date, not a benchmark of the provider.
- Local models (embeddings): wall time on the recorded CPU/GPU, cost 0.

## 7. Statistics

- 95% bootstrap confidence intervals, percentile method, 10,000 resamples, seed 20261017, resampling **groups** (section 1).
- **Paired comparisons**: for two systems on the same items, bootstrap the per-group difference of the metric; report the difference with its CI. Claim a difference only if the CI excludes 0. No multiple-comparison correction is applied, so with many comparisons treat a lone borderline result as exploratory; the experiment log marks which comparisons were planned.
- **Resolution with this benchmark** (rough, binary metric near 50%): about 40 answerable groups on dev gives a 95% interval of about ±15 percentage points for one system; about 110 groups on test about ±9 points. Paired differences are tighter when systems agree on most items, but expect that differences below roughly 8 to 10 points on test and 15 points on dev cannot be claimed. Per-language and per-act cells (8 to 40 items) can show only large gaps. Trap accuracy (5 / 15 items) supports no ranking.
- Report the CI width next to every table so readers see what can be resolved.

## 8. Run layout and reproducibility

```
eval/results/<run_id>/
  config.json     # system config, dated model ids, prompt hashes, git sha, benchmark SHA256SUMS hash, split, seed, price-table date
  per_item.jsonl  # one line per item: retrieved ids, answer object, per-metric scores, tokens, cost, latency, errors
  metrics.json    # aggregates with CIs, per language / type / config
eval/cache/       # provider response cache (git-ignored)
eval/test_runs.log  # every test-split run: date, run_id, config hash
```

- `run_id` = `<date>_<split>_<short config hash>`. The harness refuses to run on a dirty git tree for test runs.
- Determinism: temperature 0 where supported, fixed seeds, sorted inputs. Providers are not bit-deterministic; the cache is what makes a reported number reproducible, and `config.json` says which cache entries were used. A rerun without cache is expected to move by noise, and I will measure that noise on one configuration (three repeats on dev) and state it.
- Raw cache is not committed. Committed to git: `config.json`, `metrics.json`, and `per_item.jsonl` for runs cited in the report (answers are model output about public law, not raw corpus; article text is not stored in them, only ids).

## 9. Experiments covered (Step 4)

1. Retrieval only: BM25, embeddings, hybrid RRF, each with and without the LLM reranker, across the index configurations of section 2.
2. Single-shot answering with Claude, GPT and Gemini on the same retrieved context (best retrieval config chosen on dev).
3. ADK agent vs single-shot: quality, steps, cost, latency, tool-selection accuracy, failure taxonomy from traces.
4. Chunking ablation (article vs paragraph vs fixed window) on recall, then faithfulness, latency and cost for the best two.

Chunks map back to article ids; retrieval scoring is always on article ids.

## 10. Failure taxonomy (agent traces)

Assigned by a script from traces, with an LLM-free rule where possible: no gold retrieved in any step; gold retrieved but not cited; cited an invalid id; wrong language retriever chosen; stopped too early; looped without new evidence; refused answerable; answered trap; malformed output. Categories are fixed before looking at test results.

## 11. Freezing and the test run

Up to four configurations are written into `docs/phase2_progress.md` (full config and hash) **before** any test number is produced. Each is run once on `test.jsonl`. A bug fix that changes a frozen configuration requires a new entry and a note in the report that test was used more than once.

## 12. Open decisions for the user

1. Primary judge family choice (small GPT-class, Gemini-class for GPT answers) is the plan; say so if you prefer another pairing.
2. Calibration sheet uses machine English glosses because you read English only; the alternative is calibrating on English-question answers only (about 11 dev items, too few).
3. Whether traps should also be required to give no pointers at all (stricter); the draft allows them.

## Amendments

(none yet)
