# Benchmark specification (v1, draft for approval)

Status: proposal, nothing generated yet. Date: 2026-10-15.

This document fixes the design of the Phase 1 benchmark before any question is written. Everything that must be exact (article ids, acts, splits, ranks, hashes) is produced by code. The model writes only the question text, the gold facts and a difficulty label.

## 1. Size and mix

| | Final | Drafted (about +50%) |
| --- | --- | --- |
| Answerable questions | 150 | 225 |
| Not-in-corpus traps | 20 | 30 |
| Total | 170 | 255 |

Each gold article is used by at most one item in the whole benchmark, as primary or as secondary gold. The eligible pool is large (see section 3), so reuse is never needed.

**Mix by act (answerable, final).** About 60% from SR 220, the rest spread so that every act is represented.

| SR | Act | Questions |
| --- | --- | --- |
| 220 | Code of Obligations | 90 |
| 210 | Civil Code | 14 |
| 272 | Civil Procedure Code | 10 |
| 281.1 | Debt Enforcement and Bankruptcy Act | 10 |
| 235.1 | Data Protection Act | 8 |
| 221.229.1 | Insurance Contract Act | 8 |
| 241 | Unfair Competition Act | 5 |
| 221.214.1 | Consumer Credit Act | 5 |

The quotas are not proportional to act size on purpose. SR 210 and SR 220 together are 70% of the pool, and a proportional mix would leave the small acts with one or two questions each, too few to say anything about them. Results by act for the small acts will still be noisy (5 to 10 items) and the report will say so.

**Question types** (`qtype`):

- `single`: answered by exactly one article. About 128 items.
- `cross_ref`: the gold article explicitly refers to a second article of the same act ("Artikel N", "article N", "articolo N") and the full answer needs both. Two gold ids. About 22 items (15%), found by code from the references in the article text and resolved with `ids.resolve_gold`. Both articles must be eligible and fall in the same split bucket.
- `trap`: no article of the 8 acts answers it. No gold article.

## 2. Languages

**Query language** (`query_lang`): EN, DE, FR, IT, balanced by code: 38 / 37 / 38 / 37 answerable items, and 5 traps each. Balanced within each act where the quota allows.

**Source language** (`source_lang`): the language of the article text the drafter saw. It is a provenance field, not an evaluation setting: the gold is an article id and is the same in all three languages.

- EN questions: source language DE, FR, IT in equal parts (always cross-lingual).
- DE, FR, IT questions: 75% drafted from an article in one of the other two languages (cross-lingual), 25% from the same language (control).
- Result: about 81% of answerable items are cross-lingual by construction, and every query language has a same-language control group of about 9 items. That is too small for a per-cell conclusion; it is a sanity check.

Translations are not used to produce the main 150. Each question is written natively in its query language. A parallel subset is proposed in section 9.

**Corpus configurations for evaluation (Phase 2 and later).** All use the same gold ids; a retrieved chunk counts as a hit if its article id is a gold id, in any language.

- A. Same-language index: DE, FR or IT question against the index of its own language. Not defined for EN.
- B. Cross-lingual: question against a single-language index of another language (for example EN against DE only, FR against DE only).
- C. All languages: question against an index with the three language versions of every article.

## 3. Eligibility of gold articles (enforced by code)

Implemented in `fedlex_bench.eligibility.benchmark_eligible(df, min_chars=200)`:

1. `kind == "article"`, not repealed, not historical, non-empty text.
2. Present and eligible in DE, FR and IT.
3. At least 200 characters in all three languages (judgement call; fewer characters rarely hold an answer).
4. Not a transitional, final or entry-into-force provision, even when it sits in the main body (a hand-reviewed list of 30 ids, `TRANSITIONAL_IN_BODY` in `eligibility.py`; not reviewed by a Swiss-law expert).

Pool size at 200 characters, articles per language: 2,865 after rule 4 (2,877 before it; 12 articles are removed by the hand-reviewed list). By act: SR 220: 1,243; SR 210: 822; SR 281.1: 304; SR 272: 296; SR 221.229.1: 85; SR 235.1: 61; SR 221.214.1: 34; SR 241: 20. The smallest pool is SR 241, with 20 articles, for 5 questions plus about 8 drafts; if a change to the rules shrinks any act below twice its draft need I stop and ask.

Ranges, transitional ids (`uN`) and footnote-only text are never gold.

## 4. Split design

The split is decided by a rule on the article id, before anything is drafted.

```
bucket(article_id) = int(sha256("fedlex-bench-v1:" + article_id).hexdigest(), 16) % 100
split(article_id)  = "dev" if bucket < 27 else "test"
```

- Each eligible article gets its split from the rule, nothing else. An item's split is the split of its primary gold article; for `cross_ref` items the secondary article must have the same split, so no article of one split is gold in the other.
- Per-cell quotas (act x split) are filled by sampling from the articles of that split, with `random.Random(20261015)`, over inputs sorted by `natural_key`. The same seed and corpus give the same plan. Answerable targets: 40 dev, 110 test. Traps: 5 dev, 15 test, assigned by the rank of a hash of the trap slot id, not by any property of the text.
- The test split is never used to tune prompts, retrieval settings or models. Dev is for that. The full benchmark is run on test once per phase end.
- `benchmark/v1/test_gold_articles.txt` lists every gold id of the test split. Phase 3 fine-tuning data must exclude these ids, and also the dev gold ids, so the dev numbers stay an honest held-out estimate.
- Known weakness: neighbouring articles of one act often cover the same topic, so an article next to a test gold article is not independent of it. The rule prevents the same article from appearing in both splits, not near-duplicates. Phase 3 should therefore also leave out articles within two article numbers of a test gold article when it builds fine-tuning pairs; the exact rule is decided in Phase 3 and recorded there.
- The bucket rule is applied to each eligible article, so a 27/73 dev/test share of the pool results. Realised item counts per split come from the quotas above, not from the hash.

## 5. Item schema (JSONL, one object per line)

Files: `benchmark/v1/dev.jsonl`, `benchmark/v1/test.jsonl`. Keys sorted, UTF-8, `\n` line endings, items sorted by `id`.

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | str | `v1-0001` ..., assigned by code after selection |
| `split` | `dev` or `test` | from the rule in section 4 |
| `qtype` | `single`, `cross_ref`, `trap` | |
| `is_trap` | bool | equals `qtype == "trap"`; the validator checks it |
| `query_lang` | `en`, `de`, `fr`, `it` | |
| `question` | str | the question, in `query_lang` |
| `source_lang` | `de`, `fr`, `it`, null | language of the text the drafter saw; null for traps |
| `gold_article_ids` | list of str | `"<SR>:<article>"`, empty for traps; copied by code |
| `gold_facts` | list of `{text, article_id}` | 1 to 3 atomic facts in English (up to 4 for `cross_ref`); `article_id` says which gold article supports it; empty for traps |
| `source_act` | str or null | SR of the primary gold article; null for traps |
| `difficulty` | `easy`, `medium`, `hard` | drafter label, confirmed by the critic: easy = one sentence of the article states the answer, medium = needs a condition, a number or several paragraphs, hard = needs an exception, a cross-reference, or a calculation or comparison (a number that is only read off the text is not hard). Pilot finding: the first rubric called every numeric rule hard (5 of 8 pilot items), so it was tightened |
| `lexical_overlap` | int | see section 6 |
| `bm25_rank` | `{de, fr, it}` ints | rank of the best gold article in each language index |
| `trap_type` | str or null | traps only, see section 7 |
| `trap_reason` | str or null | traps only: one sentence on why the 8 acts do not answer it |
| `translation_of` | str or null | id of the original, for parallel-subset variants |
| `provenance` | object | `drafter`, `model`, `agent_file_sha256`, `plan_sha256`, `drafted_on`, `critic_verdict`, `critic_rounds`, `critic_notes` |
| `review_status` | `unreviewed`, `accepted`, `edited`, `rejected` | set only by the review app, never by an agent |
| `reviewed_by`, `reviewed_at`, `review_note` | str or null | set with `review_status` |

Gold ids, acts, languages, splits and ranks are produced by code and never typed by the model.

**Pipeline files.**

- `benchmark/plan/slots.jsonl`: one slot per draft, from `benchmark plan`. A slot holds `slot_id`, `split`, `act`, `qtype`, `query_lang`, `source_lang`, the gold ids, and for traps the `trap_type`. Committed, since it contains ids only.
- `benchmark/plan/briefs/*.md`: the text the drafters read (articles in `source_lang`). Git-ignored, because it contains Fedlex text; regenerated by `benchmark brief`.
- `benchmark/drafts/<group>.jsonl`: writer output (draft record, below). One file per writer.
- `benchmark/critic/<group>.jsonl`: critic verdicts, written by the main session (the critic is read-only).
- `benchmark/reserve/`: items the critic kept but that exceed a cell's quota, kept for backfill if a human rejects an item in the same cell.

**Draft record** (what a writer returns, one per slot, no ids):

```json
{"slot_id": "s-0042", "skip": false, "skip_reason": null,
 "question": "...", "gold_facts": [{"text": "...", "article": "A"}],
 "difficulty": "medium", "competing_candidates": ["advisory text only"], "note": ""}
```

For `cross_ref` slots the articles are labelled A (primary) and B. For traps: `question`, `trap_reason`, and no facts. `benchmark assemble` merges drafts with the plan into items; a slot with `skip: true` is dropped and its reason logged.

## 6. Lexical overlap

`lexical_overlap` measures how much a purely lexical search would help. It is computed by code after drafting, never by the model.

- BM25 (k1 = 1.5, b = 0.75), own implementation in the package. Tokens: lowercase, `\w+` with at least 2 characters, no stemming, no stop words. Index: every retrieval-eligible article of one language, one document per article, title included.
- `bm25_rank[lang]` = 1-based rank of the best-ranked gold article of the item in the `lang` index, for the question text as it is.
- A gold article that scores 0 (no shared token with the question) has no rank: `bm25_rank[lang]` is null for it. The pilot showed that a rank assigned to zero-score articles only reflects article order in the corpus and carries no information.
- `lexical_overlap` = `bm25_rank` of the item's own query language (DE, FR, IT); null when that rank is null. For EN it is null (not applicable): English function words such as "on", "or" and "in" are also tokens in the FR, DE and IT indexes, which gave the pilot EN items mid ranks by accident. The three ranks are still stored for EN items. A low value means the question reuses the article's words. Reported in bands: 1 to 3, 4 to 20, above 20, no signal (null).
- Because there is no stemming and no stop-word list, function words decide the order among weak matches, so ranks in the hundreds or thousands mean "no lexical signal", not a defect.
- Cross-lingual items will often rank poorly in the other languages' indexes. That is the cross-lingual effect, not a defect, and it is why all three ranks are stored.

The critic sees the ranks and flags questions that copy distinctive terms of the article. No item is dropped automatically for a low rank number; the distribution is reported, and the dev BM25 recall@10 check in step 5 of the kickoff guards the overall difficulty.

## 7. Traps

20 final, 30 drafted, 5 per query language. The quota across types is about equal:

- `out_of_area`: a legal area none of the 8 acts regulates (tax, criminal, social insurance and similar). Not areas the Code of Obligations does cover, such as tenancy or employment.
- `repealed`: asks for content of a provision that is repealed in the snapshot, so no in-force text answers it.
- `unspecified_detail`: sounds answerable from a covered act but asks for a figure or procedure the acts do not state.
- `non_statute`: needs case law, cantonal rules, fees or practice rather than the statute.

A trap must be one a retrieval system could wrongly "answer" with a plausible article. The critic must check, for each trap, that the answer is absent from all 8 acts in all 3 languages, by search, and names the closest article it found (stored in `critic_notes`). Correct behaviour in scoring: refuse and cite nothing.

## 8. How gold answers are scored later (decided now, built in Phase 2)

- **Retrieval:** recall@k, MRR and all-gold@k on article ids; hit if any language version of a gold article is retrieved. Ranges are mapped with `ids.resolve_gold`.
- **Answer correctness:** for each gold fact an LLM judge returns supported or not supported by the system answer. Score is fact recall. The judge model and prompt are pinned in the repo, run at temperature 0, and checked against 30 facts I label by hand; the agreement rate is reported with the results.
- **Citations:** ids cited by the system are copied by code from the retrieved context, never generated. Precision and recall against `gold_article_ids` are computed by code.
- **Faithfulness:** the judge checks each claim of the answer against the text of the article the system cited.
- **Traps:** correct if the system says the sources do not answer it and does not present an article as answering it. Open point from the pilot: a good refusal may cite a pointer article (for example 281.1:16, which says the Federal Council sets the fee tariff), and a strict "any citation fails" rule would punish it. The critic records the closest article for each trap in `critic_notes`; the exact rule was decided in Phase 2 before any test number was produced: see [eval_spec.md](eval_spec.md) section 4 (refused, no citations, no invalid ids, and the judge finds no specific answer asserted; pointers are allowed).

No scorer is written in Phase 1. (The scorer and its final definitions are in [eval_spec.md](eval_spec.md); where they differ from this section, that file wins.)

## 9. Parallel translated subset (decision for you)

The project's thesis is that only the language changes. Original questions differ in content between languages, so they cannot show that. Proposal (as built: parents are single, non-flagged originals, human review of the parent not required): translate 24 items (6 per query language, 6 dev and 18 test, all `single`, SR 220 first) into the 3 other query languages. That gives 24 groups of 4 variants, 72 extra items (`translation_of` set), same gold, same split. The critic checks that meaning is kept, and you review them. Cost: 72 extra short reviews. If you decline, the benchmark still works; the language comparison is then confounded with the question content, and the report will say so.

## 10. Validator (`uv run python -m fedlex_bench.benchmark validate`)

Exit code is non-zero on any error. Checks: schema and types; `is_trap` consistent with `qtype`; ids sorted and unique; every gold id exists in the parquet in all three languages and is benchmark-eligible; `gold_facts[].article_id` is one of the item's gold ids; split equals the rule applied to the primary gold id and to every gold id; no gold id in both splits; no gold id used by two items (translations excepted); question language balance within tolerance (each query language 22% to 28%); act quotas within 1 of the table; no duplicate or near-duplicate question text (normalised exact match, plus a character 5-gram Jaccard above 0.8 inside one query language); traps have no gold ids and a reason; questions longer than 400 characters; when freezing, every item must be `accepted` or `edited`. Warnings (not errors): same-language BM25 rank 1 for more than 40% of items; fewer than 20 items in a language pair cell.

## 11. Provenance and reproducibility of drafting

- Drafters and critics are Claude Code subagents (`question-writer`, `question-critic`, role files in `.claude/agents/`, model set to the `opus` alias). No API key is used and no `.env` is read. The model alias resolves at run time and the harness does not pin or report a dated model id. So reproducibility of drafting is limited to: the agent file hash, the plan hash, the brief hash and the date, all stored in `provenance`. The drafted text cannot be regenerated bit for bit. What is reproducible by one command is everything downstream of the draft files: assemble, ranks, split, validate, freeze, checksums.
- Every draft and critic file is committed, so the path from slot to final text is auditable.
- The writer role file was rewritten before the pilot (slot labels instead of typed ids, no lexical overlap by the model).
- Pilot finding: the brief showed only the article title, so the writer could not tell which company form or chapter an article belongs to and wrote a GmbH question for a cooperative article. Briefs now carry the act name and the structural position (division, title, chapter) taken from the raw XML headings.

## 12. Versioning and freeze

`benchmark/v1/` holds `dev.jsonl`, `test.jsonl`, `test_gold_articles.txt` and `SHA256SUMS` (checksums of those files). After the freeze commit (git tag `benchmark-v1`) the files are never edited; corrections become v2. A freeze needs: validator clean, every item accepted or edited by you, no item marked reviewed by an agent.

## 13. Cost and effort estimate for drafting

Drafting and criticism run as subagents inside this session, so there is no separate API bill; usage comes from your Claude plan. For comparison, at API rates for Claude Opus 5.5 ($4 per million input tokens, $20 per million output tokens, list prices from the claude-api reference, 2026-10-06):

| Step | Items | Input tokens | Output tokens | At API rates |
| --- | --- | --- | --- | --- |
| Pilot (10 slots, writer + critic) | 10 | 0.3 M | 0.05 M | about $2 |
| Writers, full run | 255 | 2.0 M | 0.4 M | about $16 |
| Critic round 1 (searches the corpus) | 255 | 2.6 M | 0.4 M | about $18 |
| Fix pass and critic round 2 (about 30%) | 80 | 0.9 M | 0.15 M | about $7 |
| Parallel subset translation and critic (optional) | 72 | 0.3 M | 0.1 M | about $3 |
| Total | | about 6 M | about 1.1 M | about $45, plausible range $25 to $90 |

These are estimates from item counts and article lengths, not measurements; the pilot gives the first real number, and I will report actual usage after it.

## 14. Stop conditions

I stop and ask you if: the critic drops more than 40% of drafts; fewer than 150 answerable items survive without reusing an article; any act's pool is under twice its draft need; dev BM25 recall@10 is above 90% or below 20%; the pilot shows a systematic problem (for example questions that copy article wording).

## 15. Review process (amended 2026-10-16, after the pilot and drafting)

The original plan was that one person hand-checks every item. The reviewer reads only English and has no Swiss legal training, so the process was changed. This section records what was decided and why; the Phase 1 report will give the actual counts.

- **Aids for the reviewer.** For every item, an English rendering of the question, an English rendering of the gold article(s), and for each gold fact the supporting sentence (or "not found"). They are working aids, generated by a model, kept local and git-ignored (`data/processed/review_aids/`), and labelled unofficial in the review app. The German, French and Italian texts remain the authoritative ones.
- **Independent checks before human review.**
  - A blind solver (a separate agent given only the question and the corpus) tried to find the answering article. Result: it found the gold article for 149 of 150 answerable items and judged all 20 traps unanswerable.
  - A fresh-context judge (`question-judge`, no access to drafts or critic notes) approved or rejected each item from the item and its article texts. First pass: 161 approved, 9 rejected. The 9 were revised, re-judged and approved.
  - Judge verdicts live in sidecar files (`benchmark/machine_review/`), never in `review_status`, which only the human review app sets.
- **Human review.** The reviewer checks, without legal training: that each fact matches its supporting sentence, that the question asks what the facts answer, that it reads naturally in its language, that it does not copy the article, and that traps are plausible. Anything the reviewer cannot judge is marked "Flag for expert". The reviewer's background is stored with each review (`provenance.review_basis`).
- **Calibration.** A seeded random sample of 30 items (4 traps, 26 answerable) was reviewed by the human with the judge's verdict hidden. Result: the human accepted all 30; the judge agreed on 27 (90%). The three disagreements were judge rejections for concrete text defects. The sample cannot measure how often the judge approves a bad item, because the human found no bad item among the 27 the judge approved (so the miss rate is bounded only loosely, roughly up to 10%).
- **What counts as reviewed at freeze.** The reviewer personally reviewed 34 items (the 30-item calibration sample and a few more, in risk order). By the reviewer's explicit decision, the 9 items revised after a judge rejection, the one item backfilled from the reserve after an expert flag, the 72 translations and all other unreviewed items are accepted on the judge's approval (`freeze --accept-machine-approved`). The 3 calibration items whose text was revised after the reviewer accepted them (v1-0039, v1-0050, v1-0149) therefore count as machine-approved only. The report states exact counts per split of human-checked and machine-approved-only items. An item that is human-rejected or expert-flagged is never accepted on machine approval. A cross-family judge (GPT or Gemini) is deferred until API keys are available.
- **Limits.** The drafter, critic, blind solver and judge are all Claude models, so shared blind spots are possible. No Swiss-law expert has reviewed any item; items flagged for an expert stay as a list for a future spot check.
