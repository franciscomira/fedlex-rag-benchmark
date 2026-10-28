# Fine-tuning data (Phase 3)

Training data for fine-tuning `intfloat/multilingual-e5-small` as a cross-lingual retriever over Swiss federal law. Everything here is built by code from the corpus in `data/processed/articles.parquet` and the frozen benchmark in `benchmark/v1/`. The questions are synthetic (written by an LLM from the article text); no benchmark question appears in the training data. This is not official Fedlex material; the statute texts come from Fedlex, see the repository README for licensing.

## Files

| File | What | In git |
| --- | --- | --- |
| `train_pool_ids.txt` | the 3,201 article ids that may be used for training | yes |
| `excluded_ids.json` | the excluded ids by reason (gold, neighbours, twins) | yes |
| `queries.jsonl` | 12,800 synthetic questions (article id, slot, language, style) | yes |
| `queries_failed.json` | the one article that failed validation | yes |
| `pairs.jsonl` | the questions with train/validation split and passage language | yes |
| `hardneg.jsonl` | up to 3 hard-negative article ids per training question | yes |
| `review_hardneg.md`, `review_hardneg_en.md` | a 20-example read-through sheet for the negatives, German and with English machine glosses | yes |
| `audit.json`, `audit_rows.jsonl` | the LLM audit of negatives and positives (see below) | yes |
| `mined.jsonl`, `mined_sample.jsonl` | the 30 best embedding and BM25 candidates per question (about 15 MB) | no, rebuild with `mine` |
| `queries_sample.jsonl` | the 20-article pilot | yes |

## How it was built

1. **Training pool** (`src/fedlex_bench/finetune/exclusions.py`). Start from the 3,663 retrieval-eligible article ids (each exists in DE, FR and IT). Remove the 172 ids in `benchmark/v1/dev_gold_articles.txt` and `test_gold_articles.txt`; their 288 neighbours (the previous and next eligible article of the same act, in natural order, so `257d` and `257e` count as separate articles); and 3 articles whose German text has a character 5-gram Jaccard of at least 0.5 with a gold article (the benchmark's own twin threshold was 0.7 and found none). Result: **3,201 articles**.
2. **Synthetic questions** (`synth.py`). One call to `claude-haiku-5-5` per article returns four questions: two English and two in two of DE/FR/IT. The language left out is the one the article is shown in, so no question is written in the language of the text the model read. Each question has one of six styles (keywords, short question, legal wording, lay situation, third party, confused); four distinct styles per article. The prompt forbids article numbers; code checks the JSON, length, article/SR number patterns and duplicates, retries once, and records failures. The first 20 pilot articles led to prompt changes (a rule against adding facts the article lacks, and an instruction for the third-party style to use the role named in its slot and not to default to "my business partner"). **Correction (found in review):** the code that was meant to name a role per slot was never applied (a failed edit, later removed as dead code), so the prompt's "role named in the slot" had nothing to refer to and the third-party questions got roles chosen by the generator, often a client or a business partner. Seed 20261028; the response cache makes a rerun free and identical. Result: **12,800 questions for 3,200 articles** (one article, `272:163`, failed twice on an article-number leak). Languages: English 6,400, German 2,130, French 2,162, Italian 2,108. Cost 0.87 USD.
3. **Split and passage language** (`dataset.py`). 10% of the articles per act (324 articles, 1,296 questions) go to validation with all their questions; the other 2,876 articles (11,504 questions) train. The split is by article and decided by a hash, so no article appears on both sides. Each question gets one passage language (DE, FR or IT) by hash, so a question is never expanded into several rows of the same article.
4. **Hard negatives** (`mine.py`, `hardneg.py`). For each question the base e5-small and BM25 each list the 30 best articles of the training pool (never a gold, neighbour or twin, and never the positive; the other language versions of the positive are the same article). The filter `pick_hard_negatives` then drops candidates that are the positive's neighbours (same act, within 3 pool positions) or share at least 30% of its character 5-grams, drops the 3 best candidates when the base model ranks the positive below them (likely false negatives), drops articles shorter than 100 characters, and lets an article serve as a negative for at most 25 questions. Candidates found by both embedding and BM25 come first. The draft was written by Claude; the thresholds are to be reviewed by the project owner (see `docs/phase3_progress.md`).

## Counts

| | Questions | Articles |
| --- | --- | --- |
| Train | 11,504 | 2,876 |
| Validation | 1,296 | 324 |
| Hard negatives: 3 / 2 kept per training question | 11,502 / 2 | |

Training questions by style: confused 1,852, keywords 1,907, lay situation 1,951, legal wording 1,954, short question 1,936, third party 1,904.

Observations on the base model (corrected embedder; see the correction below). Among the 3,201 pool articles it ranks the labelled positive first for 48% of the questions, in the top 10 for 75% and in the top 30 for 84%. By question language, top 10: English 64%, German 87%, French 82%, Italian 86%. By style, top 10: legal wording 94%, short question 85%, keywords 83%, third party 66%, confused 61%, lay situation 58%. Cosine scores are compressed, so a score margin cannot separate safe from unsafe negatives. With the filter, 2,675 distinct articles serve as first negative, the 100 most used fill 11% of the first slots, and the median first negative has 421 characters (pool median 377).

**Correction (2026-10-28).** The first version of this data was mined with an embedder that averaged padding tokens into batched vectors (a bug of the Phase 2 code, see `docs/phase2_report.md`, errata). Everything above was rebuilt with the corrected embedder. The earlier conclusions that short generic articles act as hubs (49% of first slots) and that keyword questions are hardest for the base model (top 10: 13%) were artifacts of that bug; the hub cap (25 uses) and the minimum length (100 characters) in the filter stay in place but now bind little.

## Leakage check

`uv run pytest tests/test_finetune_leakage.py` fails on any of: a gold article, a gold neighbour or a twin of gold in the pool; a positive outside the pool; an article in both train and validation; a hard negative outside the pool, equal to the positive, or a gold article. A model's `config.json` data hashes must match the tested files, and the twin check is repeated with an independent implementation. Result on the files above: **all pass, no overlap** (17 tests with `test_finetune_train.py` and `test_finetune_synth.py`). A deliberately poisoned pool made the gold and twin checks fail. **Known gap (found in review):** the twin criterion is a symmetric Jaccard of at least 0.5, which does not catch a short pool article contained in a long gold article: 6 pool articles are at least 80% contained in a gold article, 37 at least 60% (touching 18 gold articles), 84 at least 50%. They were not removed (the pool would change and the model would need retraining). On test, the tuned model's gain over the untuned system is unchanged on the 142 answerable items whose gold article is not touched at the 60% level (+0.30 [+0.20, +0.40] recall@10 in group means), see `docs/phase3_report.md`, limitation 10. The check covers article ids; it does not cover paraphrase leakage of benchmark questions into synthetic ones (the synthetic prompt never saw a benchmark question).

## Audit by a second model family

Nobody on the project reads German, French and Italian well enough to review the data, so a model from another family than the generator (`gpt-6-luna`, the Phase 2 sensitivity judge, known to be lenient) judged 330 (question, article) pairs: does the article contain what is needed to answer the question (`yes`, `partly`, `no`). Cost 0.04 USD. 100 sampled training questions, the same questions in every set (`audit.py`, seed 20261028), rerun on the corrected negatives:

| Candidate article | n | yes | partly | no |
| --- | --- | --- | --- | --- |
| first hard negative chosen by the filter | 100 | 1 (95% CI 0.00 to 0.05) | 24 | 75 |
| best wrong article of the base model, no filter | 100 | 4 (0.02 to 0.10) | 34 | 62 |
| the labelled positive | 100 | 80 (0.71 to 0.87) | 20 | 0 |
| a random pool article (control) | 30 | 0 | 0 | 30 |

- **False negatives are rare**: 1 of 100 for the filtered negative, 4 of 100 without the filter (intervals overlap, so the filter's benefit on this measure is suggestive, not shown). 24% of the chosen negatives are "partly" related, which is what a hard negative should be.
- **About one question in five is only partly answered by its own article** (20 of 100). The judge is lenient, so the true share is probably not lower. The positive labels are noisy but mostly point the right way, and the in-batch loss will treat a partial match as the correct answer.
- The judge was not calibrated on this task, and nobody checked a sample of its verdicts against the legal text. The false negatives themselves were not inspected after the rebuild.

## Limitations

- The questions are model-written, and no human who reads German, French or Italian, nor a Swiss-law expert, reviewed them. The only check is the LLM audit above (about 1 in 5 questions only partly answered by its article; 1% of the chosen hard negatives also answer the question).
- English dominates (50%); German, French and Italian have 16 to 17% each.
- Same drafting model family as the benchmark's questions (Claude); the style variety is meant to limit, not remove, the shared drafting style.

## Reproduce

```
uv run python -m fedlex_bench.finetune.exclusions
uv run python -m fedlex_bench.finetune.synth run --yes      # free from the cache; needs the cache or an Anthropic key
uv run python -m fedlex_bench.finetune.dataset
uv run python -m fedlex_bench.finetune.mine
uv run python -m fedlex_bench.finetune.hardneg
```

SHA-256 prefixes of the committed files are in `KIT_INFO.json` of the Kaggle kit and in every training `config.json` (`data_sha256`).
