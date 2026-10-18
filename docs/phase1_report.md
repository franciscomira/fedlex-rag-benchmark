# Phase 1 report: the benchmark

Date: 2026-10-16. Frozen as `benchmark/v1/` (git tag `benchmark-v1`, pushed to origin). Spec: [benchmark_spec.md](benchmark_spec.md). Progress log with the decisions in order: [phase1_progress.md](phase1_progress.md).

## Summary

- **242 items** in `benchmark/v1/`: 170 originals (150 answerable questions, 20 not-in-corpus traps) and 72 translations of 24 of the answerable questions into the three other query languages.
- Questions are in English, German, French and Italian and are answered by one article (128) or by two linked articles (22) of eight Swiss federal acts. 82.7% of the answerable questions are cross-lingual by construction (question language different from the language of the article text the drafter saw).
- Dev has 45 originals, test has 125. No gold article is in both splits.
- **Review coverage, honestly: 34 of 242 items were checked by a human; 208 were accepted on the approval of an automated judge.** The human reads only English and has no legal training. No Swiss-law expert has reviewed anything. Details below.
- First BM25 baseline on dev (never tuned): recall@10 is 25% to 45% for German, French and Italian questions against their own language, 0% for English questions, 30% overall against an index with all three languages.

## Format

JSONL, one item per line, keys sorted, LF endings. Files in `benchmark/v1/`:

| File | Content |
| --- | --- |
| `dev.jsonl`, `test.jsonl` | the items (63 and 179 including translations) |
| `dev_gold_articles.txt`, `test_gold_articles.txt` | sorted gold article ids per split; Phase 3 fine-tuning data must exclude both |
| `REVIEW_COVERAGE.json` | human-reviewed and machine-approved-only counts per split, and the machine-only ids |
| `SHA256SUMS` | checksums of the five files above; `sha256sum -c SHA256SUMS` verifies |

Item fields (full table in the spec, section 5): `id`, `split`, `qtype` (`single`, `cross_ref`, `trap`), `is_trap`, `query_lang`, `question`, `source_lang` (language of the article text the drafter saw), `gold_article_ids` (`<SR>:<article>`, copied by code), `gold_facts` (1 to 4 short facts in English, each tied to one gold article), `source_act`, `difficulty`, `lexical_overlap` and `bm25_rank` (computed by code), `trap_type`/`trap_reason`, `translation_of`, `provenance`, `review_status`.

Gold article ids, acts, splits and ranks are never typed by a model. Writers saw a labelled article text (A, B) and a slot id; code attached everything else.

How gold answers will be scored later (decided, not built): retrieval recall on article ids (a hit if any language version is retrieved); answer correctness as fact recall by a pinned LLM judge calibrated on hand-labelled facts; citation precision/recall by code; faithfulness by the judge; traps correct if the system says the sources do not answer and presents no article as answering it. The exact trap citation rule is open (see weaknesses).

## Counts (originals unless stated)

By act and split (answerable questions):

| SR | Act | Dev | Test | Total |
| --- | --- | --- | --- | --- |
| 220 | Code of Obligations | 24 | 66 | 90 |
| 210 | Civil Code | 4 | 10 | 14 |
| 272 | Civil Procedure Code | 3 | 7 | 10 |
| 281.1 | Debt Enforcement and Bankruptcy Act | 3 | 7 | 10 |
| 235.1 | Data Protection Act | 2 | 6 | 8 |
| 221.229.1 | Insurance Contract Act | 2 | 6 | 8 |
| 241 | Unfair Competition Act | 1 | 4 | 5 |
| 221.214.1 | Consumer Credit Act | 1 | 4 | 5 |
| | Total | 40 | 110 | 150 |

Plus 20 traps (5 dev, 15 test, 5 per question language).

- **Type:** 128 single, 22 cross_ref, 20 trap.
- **Question language (answerable):** EN 39, DE 37, FR 37, IT 37. EN is one above the 38/37/38/37 plan because one backfilled item replaced a French item with an English one.
- **Question language x article language (answerable):** de-de 9, de-fr 14, de-it 14; fr-de 15, fr-fr 8, fr-it 14; it-de 14, it-fr 14, it-it 9; en-de 13, en-fr 14, en-it 12. The 26 same-language items (9, 8, 9) are a control group too small for conclusions per language.
- **Difficulty:** medium 111, hard 22 (exactly the cross_ref items), easy 17. Difficulty therefore duplicates question type; do not read a result split by difficulty as independent information.
- **Same-language lexical overlap (BM25 rank of the gold article, non-English questions, n=111):** rank 1 to 3 for 37 (33%), 4 to 20 for 24, above 20 for 50. English questions have no defined value (the three ranks are stored).
- **Trap types:** repealed 6, out of area 5, non-statute 5, unspecified detail 4.
- **Translations:** 24 groups of 3, 18 per target language, 18 dev and 54 test; parents are SR 220 single questions, 6 per original language. They share gold, facts and split with their parent.
- Dev answerable questions per language: EN 11, DE 8, FR 10, IT 11. The validator warns about this; per-language dev numbers rest on 8 to 11 items.
- Gold articles: 172 distinct ids; no article is used by two originals.

## How the questions were produced

Everything below ran as Claude Code subagents inside this project; no API key was used. The drafter and critic model is the `opus` alias set in `.claude/agents/*.md`. The alias resolves at run time and the harness does not report a dated model id, so the drafts cannot be regenerated bit for bit. The agent file hash, plan hash, brief hash and date are stored in each item's `provenance`.

1. **Corpus and eligibility** (Phase 0, reviewed twice by an independent reviewer agent): 3,663 in-force main-text articles per language; benchmark pool of 2,865 articles after requiring at least 200 characters in every language and removing 30 hand-listed transitional or entry-into-force provisions that sit in the main body; 59 articles with a near-identical twin (5-gram Jaccard 0.7 or more in German) were excluded as gold.
2. **Plan by code**, seed 20261015: split by hash of the article id (27% dev bucket), quotas per act and split, query and source language balance, cross_ref pairs found from literal article references in the text. 258 slots (about 50% more than needed). Plan sha256 `511bb7ec...`.
3. **Pilot** (10 items, twice): the first pilot exposed that briefs lacked act and structure context (a question about a limited liability company was written for a cooperative article), that the difficulty rubric was too generous, and that BM25 ranks for zero-score articles and for English are not informative. All were fixed. A code review at that point also found selection drift when items are dropped, drafts not bound to their slot, critic verdicts not bound to draft text, twin articles, article numbers visible in briefs, and cross_ref pairs dominated by catalogue articles; all were fixed.
4. **Drafting**: question-writer agents, one brief per group of about 25 slots, output not containing ids.
5. **Critic round 1** (`question-critic`, checks every item against the corpus in all three languages): of 258 drafts, 219 keep, 39 fix, 0 drop. Fix reasons: facts not supported, facts not asked, or other (16), competing article (7), ambiguous (5), copied wording (5), answerable trap (4), outside knowledge (2). All 39 were revised and re-checked in rounds 2 to 5.
6. **Selection by code** (`select`) from the kept slots, 170 items, with reserve backfill.
7. **Review aids and independent checks before human review:**
   - English renderings of questions and gold articles and a supporting sentence per gold fact (Sonnet). No gold fact was found unsupported. These are working aids, not published.
   - A blind solver (Sonnet, given only the question and the corpus) found the gold article for 149 of 150 answerable questions and judged all 20 traps unanswerable.
   - A fresh-context judge (`question-judge`, Opus, sees only the item and the three-language article text): 161 approved, 9 rejected at first pass. The 9 were revised, re-judged and approved. One item flagged by the human for an expert was replaced from the reserve and judged.
8. **Translations** by Sonnet agents (prompt in `benchmark/translations/translator_prompt.md`), each checked for fidelity by the judge; all 72 approved.
9. **Final independent data review** (`eval-reviewer`) before freeze: no split leakage, no questions quoting articles, no answerable traps; it found two freeze defects (item order; judge verdicts tied to ids, not to text) that were fixed, plus the points under "weaknesses".

The cost was not measured. Estimated token use was about 6 million input and 1 million output tokens for drafting and criticism at the first estimate; the later review aids, blind solver, judge and translations added to that. The estimate in the spec has not been replaced by a measured number.

## What is checked by a human, and what is not

- **Human (34 items):** the reviewer reads only English and has no legal training; the review app showed English renderings, supporting sentences and risk flags. Reviewer background is stored per review (`provenance.review_basis`). The 34 are the 30-item random calibration sample plus 4 more. All 34 were accepted.
- **Calibration:** the judge agreed with the human on 27 of 30 (90%). The three disagreements (v1-0039, v1-0050, v1-0149) were judge rejections for concrete text defects that the human had accepted. The human found no defect among the 27 items the judge approved, which bounds the judge's miss rate only loosely (zero in 27; roughly up to 10% at the usual confidence).
- **Machine only (208 items):** approved by the fresh-context judge, whose verdicts are bound by a content hash to the exact question and facts it saw. This includes the 9 items revised after a judge rejection and the one backfilled item (by the reviewer's explicit decision), all 72 translations, and every other item not among the 34. By split: dev 10 human and 53 machine-only, test 24 human and 155 machine-only (`REVIEW_COVERAGE.json` lists the ids).
- **Not checked by anyone:** whether a subtle legal exception changes an answer; whether the German, French and Italian versions differ in meaning in ways the facts do not cover. Where the three versions visibly differ the facts were worded to hold in all three (examples noted by the checkers: 220:874, 220:1182).

## Baseline sanity check

BM25 recall@10 on dev, 40 answerable originals (`benchmark/baseline_dev.json`), no tuning, test never read:

| Question language | German index | French index | Italian index | All three languages | n |
| --- | --- | --- | --- | --- | --- |
| EN | 0.0% | 0.0% | 0.0% | 0.0% | 11 |
| DE | 25.0% | 0.0% | 0.0% | 43.8% | 8 |
| FR | 0.0% | 45.0% | 10.0% | 45.0% | 10 |
| IT | 0.0% | 0.0% | 36.4% | 36.4% | 11 |
| Overall | 5.0% | 11.3% | 12.5% | 30.0% | 40 |

The benchmark is neither trivial nor impossible for a lexical baseline: same-language and all-language recall sits between 25% and 45%; English questions score 0% because BM25 cannot cross languages. The single-language overall figures are low only because most questions are in another language than the index. Cells have 8 to 11 items. With the 18 dev translations included (`benchmark/baseline_dev_with_translations.json`) overall all-language recall is 36%.

## Known weaknesses

1. **Review depth.** 208 of 242 items rest on one model's approval. The drafter, critic, blind solver and judge are all Claude models, so shared blind spots are possible. A cross-family judge (GPT or Gemini) and a spot check by a Swiss law student are not done; the plan lists both.
2. **Judge validation is thin.** 30 calibration items, no bad item among them, so the judge's rate of approving bad items is not measured.
3. **Gold facts.** v1-0149 fact 1 mixes two rules (extract sent to creditors, full list laid open for ten days) and the Italian wording differs; it passed the judge but is the weakest known item. Many gold facts are model-written paraphrases; facts that differ between language versions are worded to hold in all three, which makes some of them less specific.
4. **Lexical overlap is common.** A third of the non-English questions are found by BM25 in rank 1 to 3 in their own language, mostly through unavoidable legal terms. Lexical baselines will look stronger on the same-language items than on the cross-lingual ones.
5. **Difficulty labels** equal question type (all cross_ref are hard, no single is hard).
6. **Small cells.** Small acts have 5 to 10 questions; the same-language control group has 8 or 9 questions per language; dev has 8 German questions. Differences between such groups will be noisy.
7. **Neighbouring articles across splits.** 25 dev/test gold pairs lie within one article number of each other (for example 220:795a in dev and 220:795b in test). The rule prevents the same article in both splits but not close neighbours; Phase 3 should exclude neighbours of test gold articles as well as the gold lists.
8. **Pointer articles in traps.** For about 25 of 30 traps the corpus has an article that only says where to look (for example that the Federal Council sets a fee tariff). A strict "any citation fails" rule would punish good refusals. The scoring rule is decided in Phase 2, before any test number is produced.
9. **Twin articles.** 59 articles with near-identical twins were excluded as gold, but the twins remain in the corpus; a retriever that returns a twin of a gold article is marked wrong. SR 281.1 has many procedural twins (composition, bankruptcy, moratorium).
10. **Model alias not pinned.** Drafting cannot be reproduced exactly; everything from the drafts onward is reproducible with the commands below.
11. **Snapshot.** Ids of transitional provisions are positional and not used as gold. The corpus is one consolidation per act (dates in the Phase 0 report); later law changes are not tracked.
12. **Licence.** The benchmark contains questions and facts written by models, not Fedlex text; the English article renderings and the corpus are not committed. The Fedlex non-official-publication notice applies wherever article text is shown.

## Reproduce

```bash
uv run python -m fedlex_bench.chunk                       # build data/processed/articles.parquet
uv run python -m fedlex_bench.benchmark plan --out slots_check.jsonl    # same sha256 as benchmark/plan/slots.jsonl
uv run python -m fedlex_bench.benchmark validate --dir benchmark/v1
(cd benchmark/v1 && sha256sum -c SHA256SUMS)
uv run python -m fedlex_bench.benchmark baseline --dir benchmark/v1    # dev only
uv run pytest
```

Drafts, critic and judge verdicts, assembled candidates and revisions are committed under `benchmark/` (`drafts/`, `critic/`, `machine_review/`, `assembled/`, `translations/`, `archive/pilot_round0/`), so the path from slot to final text is auditable.

## Next

- Phase 2 baseline harness: done (the trap citation rule was decided in [eval_spec.md](eval_spec.md) section 4; results in [phase2_report.md](phase2_report.md)). Never use `test.jsonl` or the test gold list for tuning.
- Cross-family judge pass: done in Phase 2 (Gemini 3.6 Flash approved all 242 items, weak evidence; see phase2_report.md). Still optional: a student spot check of the item ids flagged as unsure by the checkers; if any item changes, make a v2, never edit v1.
- Phase 3: exclude `dev_gold_articles.txt` and `test_gold_articles.txt`, and articles adjacent to them, from fine-tuning pairs.
