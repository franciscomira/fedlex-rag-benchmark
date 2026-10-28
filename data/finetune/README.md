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
2. **Synthetic questions** (`synth.py`). One call to `claude-haiku-5-5` per article returns four questions: two English and two in two of DE/FR/IT. The language left out is the one the article is shown in, so no question is written in the language of the text the model read. Each question has one of six styles (keywords, short question, legal wording, lay situation, third party with a fixed role, confused); four distinct styles per article. The prompt forbids article numbers; code checks the JSON, length, article/SR number patterns and duplicates, retries once, and records failures. The first 20 pilot articles led to two prompt changes (fixed roles for the third-party style, a rule against adding facts the article lacks). Seed 20261028; the response cache makes a rerun free and identical. Result: **12,800 questions for 3,200 articles** (one article, `272:163`, failed twice on an article-number leak). Languages: English 6,400, German 2,130, French 2,162, Italian 2,108. Cost 0.87 USD.
3. **Split and passage language** (`dataset.py`). 10% of the articles per act (324 articles, 1,296 questions) go to validation with all their questions; the other 2,876 articles (11,504 questions) train. The split is by article and decided by a hash, so no article appears on both sides. Each question gets one passage language (DE, FR or IT) by hash, so a question is never expanded into several rows of the same article.
4. **Hard negatives** (`mine.py`, `hardneg.py`). For each question the base e5-small and BM25 each list the 30 best articles of the training pool (never a gold, neighbour or twin, and never the positive; the other language versions of the positive are the same article). The filter `pick_hard_negatives` then drops candidates that are the positive's neighbours (same act, within 3 pool positions) or share at least 30% of its character 5-grams, drops the 3 best candidates when the base model ranks the positive below them (likely false negatives), drops articles shorter than 100 characters, and lets an article serve as a negative for at most 25 questions. Candidates found by both embedding and BM25 come first. The draft was written by Claude; the thresholds are to be reviewed by the project owner (see `docs/phase3_progress.md`).

## Counts

| | Questions | Articles |
| --- | --- | --- |
| Train | 11,504 | 2,876 |
| Validation | 1,296 | 324 |
| Hard negatives: 3 / 2 / 1 / 0 kept per training question | 7,538 / 826 / 1,133 / 2,007 | |

Training questions by style: confused 1,852, keywords 1,907, lay situation 1,951, legal wording 1,954, short question 1,936, third party 1,904.

Observations on the base model that shaped the filter: its cosine scores are compressed (positive 0.874 on average, best wrong article 0.892, 30th-best wrong article 0.873); it ranks the positive first for 20% of the questions and in the top 10 for 39% (legal wording 76%, keywords 13%, English 29%). Without the hub cap and the length rule, 100 short generic articles (3% of the pool) filled 49% of the first-negative slots.

## Leakage check

`uv run pytest tests/test_finetune_leakage.py` fails on any of: a gold article, a gold neighbour or a twin of gold in the pool; a positive outside the pool; an article in both train and validation; a hard negative outside the pool, equal to the positive, or a gold article. Result on the files above: **all pass, no overlap** (11 tests with `test_finetune_synth.py`). A deliberately poisoned pool made the gold and twin checks fail. The check covers article ids; it does not cover paraphrase leakage of benchmark questions into synthetic ones (the synthetic prompt never saw a benchmark question).

## Audit by a second model family

Nobody on the project reads German, French and Italian well enough to review the data, so a model from another family than the generator (`gpt-6-luna`, judge of Phase 2's sensitivity check, known to be lenient) judged 330 (question, article) pairs: does the article contain what is needed to answer the question (`yes`, `partly`, `no`). Cost 0.04 USD. 100 sampled training questions, the same questions in every set (`audit.py`, seed 20261028):

| Candidate article | n | yes | partly | no |
| --- | --- | --- | --- | --- |
| first hard negative chosen by the filter | 100 | 3 (95% CI 0.01 to 0.08) | 16 | 81 |
| best wrong article of the base model, no filter | 100 | 2 (0.01 to 0.07) | 18 | 80 |
| the labelled positive | 100 | 79 (0.70 to 0.86) | 20 | 1 |
| a random pool article (control) | 30 | 0 | 0 | 30 |

- **False negatives are rare in both sets (about 3%), and the filter does not measurably reduce them** (3 against 2 of 100, on largely the same questions). What the filter changed is hubness, not false negatives. All 3 filtered false negatives are the same pattern: the Code of Obligations repeats a rule for another company form (stock corporation versus limited-liability company, for example `220:710` against `220:896`, term of office). The 0.3 text-overlap rule does not catch these; a rule on same-title articles in the same act would.
- **About one question in five is only partly answered by its own article** (20 of 100, `lay_situation` 48% of its 23 questions, `legal_wording` 12%, `short_question` and `keywords` none; per-style cells hold 5 to 23 questions, so read them as hints). One was not answered (an adoption article for a naturalisation question). The judge is lenient, so the true share is probably not lower. The training signal is still mostly right (the article is the best place), but the positive labels are noisy, and the in-batch loss will treat a partial match as the correct answer.
- The judge was not calibrated on this task, and nobody checked a sample of its verdicts against the legal text.

## Limitations

- The questions are model-written, and no human who reads German, French or Italian, nor a Swiss-law expert, reviewed them. The only check is the LLM audit above (about 1 in 5 questions only partly answered by its article; about 3% of hard negatives also answer the question).
- 17% of the training questions have no hard negative after filtering, most of them `keywords` and English questions, where the base model's candidate list is mostly generic hubs.
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

SHA-256 prefixes of the committed files: `pairs.jsonl` 9d1ee3e1, `hardneg.jsonl` c4e17b5b, `queries.jsonl` 9f5e919d, `train_pool_ids.txt` 5c7be002.
