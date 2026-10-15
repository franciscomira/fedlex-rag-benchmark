# Phase 0 report: data foundation

Date: 2026-10-15. Scope: eight acts in German, French and Italian. The three acts from the kickoff (SR 220 Code of Obligations, SR 210 Civil Code, SR 235.1 Data Protection Act) are the alignment check; five related acts were added afterwards to form the corpus slice (section 2).

**Recommendation: GO**, with the adjustments listed at the end. Id alignment is 100% (stop threshold was 95%), the pipeline is deterministic, and 107 tests pass. The 100% figure is largely structural, so the content-level checks below matter more than the headline number.

## 1. Data source and licence

**Source used:** Fedlex open data. SPARQL endpoint `https://fedlex.data.admin.ch/sparqlendpoint` for discovery, filestore for Akoma Ntoso XML. For each act the pipeline picks the consolidation applicable on a fixed as-of date (2026-10-15) and records its URL and consolidation date.

**Source rejected:** the Swiss Case Law Open Dataset on Hugging Face (`voilaj/swiss-caselaw`, CC0). It contains court decisions only. Its 121 data files include no statute files, so it cannot provide article-level federal law.

**Licence.** Fedlex states on <https://fedlex.data.admin.ch/de-CH/home/intro> that texts and metadata are provided under the terms on <https://www.fedlex.admin.ch/de/broadcasters>, which allow reuse for commercial and non-commercial purposes. Conditions come from Art. 49 para. 1 of the Publikationsverordnung (SR 170.512.1):

- content must not be altered;
- it must be visibly distinct from commentary or additions;
- it must carry the notice "Dies ist keine amtliche Veröffentlichung. Massgebend ist allein die Veröffentlichung durch die Bundeskanzlei.";
- it must not suggest an official publication;
- it may only be redistributed for a fee in refined form (condition h).

Consequences for this repo: the README carries the notice, and raw files and the Parquet are not committed. Chunk text drops footnotes, which could arguably count as altering content. This is documented in the README and is an open decision (see section 6). I did not find a Fedlex dataset entry on opendata.swiss that assigns one of its term categories, and the Fedlex "legal notice" page could not be read (JavaScript only). The broadcasters page is treated as the operative source.

## 2. Consolidations used

| SR | Act | Consolidation | Applicable |
| --- | --- | --- | --- |
| 210 | Civil Code | 2026-07-01 | open-ended |
| 220 | Code of Obligations | 2026-10-01 | to 2027-06-30 |
| 235.1 | Data Protection Act (2020 revision) | 2025-07-07 | open-ended |
| 221.214.1 | Consumer Credit Act | 2023-09-01 | open-ended |
| 221.229.1 | Insurance Contract Act (VVG) | 2024-01-01 | open-ended |
| 241 | Unfair Competition Act | 2025-01-01 | open-ended |
| 272 | Civil Procedure Code | 2026-07-01 | open-ended |
| 281.1 | Debt Enforcement and Bankruptcy Act | 2026-01-01 | open-ended |

The old 1992 Data Protection Act (repealed 2023-08-31) is excluded by selection.

The five added acts were chosen as the statutes most often applied together with the Code of Obligations: contract and consumer law (221.214.1, 241), insurance contracts (221.229.1, the same domain as the private Confluence RAG work this project replays publicly), and how claims are enforced (272, 281.1). They give a corpus with a clear core (the CO) and neighbouring acts that questions can legitimately cross into. This is a judgement call, not a measured choice, and the list can still change before Phase 1.

## 3. Counts (identical for DE, FR and IT)

| SR | Chunks | Main-text articles | Transitional/final | Repealed or elided | Historical | In force |
| --- | --- | --- | --- | --- | --- | --- |
| 210 | 1,277 | 1,099 | 178 | 40 | 74 | 1,163 |
| 220 | 1,690 | 1,602 | 88 | 91 | 0 | 1,599 |
| 221.214.1 | 46 | 46 | 0 | 0 | 0 | 46 |
| 221.229.1 | 112 | 112 | 0 | 14 | 0 | 98 |
| 235.1 | 77 | 77 | 0 | 0 | 0 | 77 |
| 241 | 33 | 33 | 0 | 3 | 0 | 30 |
| 272 | 428 | 428 | 0 | 1 | 0 | 427 |
| 281.1 | 410 | 400 | 10 | 6 | 0 | 404 |
| **Total per language** | **4,073** | **3,797** | **276** | **155** | **74** | **3,844** |

`articles.parquet` has 12,219 rows (4,073 × 3), 11 columns: the seven required (`sr`, `article_id`, `lang`, `title`, `text`, `source_url`, `consolidation_date`) plus `kind`, `is_repealed`, `is_historical`, `block_heading`.

- *Repealed* chunks are kept with empty text and `is_repealed=True`, not dropped. The flag also covers elided ("…") and obsolete provisions.
- *Historical* are 74 articles in SR 210 (block "Wortlaut der früheren Bestimmungen", the pre-1988 marital property text). They are flagged `is_historical=True` and should be excluded from retrieval, otherwise they compete with current law.
- Transitional chunks include 5 SR 220 blocks without `<article>` elements, chunked whole.

## 4. Alignment

| SR | Aligned ids | Union | Rate |
| --- | --- | --- | --- |
| 210 | 1,277 | 1,277 | 100.00% |
| 220 | 1,690 | 1,690 | 100.00% |
| 235.1 | 77 | 77 | 100.00% |
| 221.214.1 | 46 | 46 | 100.00% |
| 221.229.1 | 112 | 112 | 100.00% |
| 241 | 33 | 33 | 100.00% |
| 272 | 428 | 428 | 100.00% |
| 281.1 | 410 | 410 | 100.00% |
| All | 4,073 | 4,073 | 100.00% |

In-force chunks only: 3,844 of 3,844. No id is missing in any language.

**What this does and does not show.** Ids come from Fedlex eIds, which Fedlex derives from the document structure. So 100% shows that the three XML files share the same structure. It does not show that the text behind an id is the same provision. An independent reviewer made this point, and the report now includes content-level checks:

- Number check: the eId-derived number matches the printed number in 12,185 of 12,186 articles. The exception is `220:697l-697m` in Italian, a markup quirk.
- Paragraph and item counts differ across languages for 5 of 4,073 ids: `210:94` (IT has an extra elided paragraph), `220:644` (IT lacks paragraph 2), `220:1033` (DE has paragraph 2 that FR and IT lack), `221.229.1:95c` (IT has 4 items, DE and FR 3) and `281.1:219` (DE has 14 items, FR and IT 15). The reviewer's check of the first three suggested they are differences in the official texts rather than parsing errors. The last two have not been checked against the raw XML.
- Length ratio above 1.6 (longest over shortest of DE/FR/IT, in-force, longest over 200 chars): 10 of 3,844 (six in SR 281.1, plus `220:1075`, `220:644`, `210:336`, `210:339`). They look like ordinary translation differences but have not been read pair by pair.
- Parser fix found while adding the new acts: IT `221.229.1:48-49` has no body and a footnote that says only "Introdotto dal ...", not "Abrogato". It was first flagged in force in IT but repealed in DE and FR. The data smoke test caught the flag mismatch. Any article with no body text is now treated as repealed, with a test.
- Five random aligned articles (seed 42) printed side by side read as correct translations. They are in `docs/phase0_alignment.txt`.

Full output is in [phase0_alignment.txt](phase0_alignment.txt). Regenerate with `uv run python -m fedlex_bench.chunk` and `uv run python -m fedlex_bench.alignment`.

## 5. Reproducibility

- Re-running the chunker gives a byte-identical Parquet (sha256 `9b7e404de8a86e09e66716f79a3a18141c13e68064ba58916971613aae81a491`).
- The discovery manifest `data/manifest.json` is tracked in git and holds the source URLs, consolidation dates and a sha256 per raw file. Raw files are verified against it on load. Raw file names include the filestore revision.
- The as-of date is fixed, so results do not drift with the day the pipeline runs.
- Caveat: the recorded hashes come from the first download. They prove the files have not changed since, not that they match Fedlex's current content. A fresh clone with no `data/raw` re-downloads whatever the filestore serves for the pinned URLs.
- Python is pinned in `.python-version` (3.13), dependencies in `uv.lock`.

## 6. Known issues

1. **Footnotes dropped.** Possible conflict with the "no alteration" condition. Decide whether to keep them (as a separate column) before the repo goes public.
2. **Range ids.** Repealed ranges such as `210:135-149` are one chunk, so a lookup of `210:140` finds nothing. All such chunks are repealed and have no text, so retrieval is unaffected today. Citation parsing will need range expansion.
3. **Transitional ids are positional.** `uN` is the ordinal of the `disp_` block in the snapshot. A new block shifts later ids, so gold labels that cite transitional provisions are only valid for this snapshot. Prefer main-text articles for benchmark questions. The `block_heading` column gives context.
4. **Annexes not chunked.** SR 235.1 annexes 1 and 2 (about 50 to 300 characters each per language) are left out.
5. **Unreviewed translation differences.** The three structural mismatches and four length outliers in section 4 have not been read by someone who knows Swiss law.
6. **Source URL deep links.** `source_url` is the filestore URL plus `#<eId>`. Whether the fragment resolves on fedlex.admin.ch has not been checked.
7. **Marginal titles.** 6 (SR 210), 20 to 21 (SR 220) and 2 (SR 235.1) in-force chunks have an empty title.

## 7. Review process

The code was reviewed by a read-only reviewer agent, working from the eval-reviewer role (the project agent was not registered at the time, so a general-purpose agent got its rules inline). It reported 16 findings, including two high-severity ones: 222 superseded SR 210 rows counted as in force, and a manifest ignored by git. All were fixed except the range lookup. The fixes were verified by re-running tests (107 pass), ruff and the chunker. The fixed code has not had a second independent review.

## 8. Recommendation: GO, with adjustments

Go: the source is usable, alignment passes the threshold, and the data is reproducible.

Adjustments before Phase 1:

- **Corpus slice.** The plan targets the Code of Obligations plus 3 to 5 related acts. The eight acts here give 3,844 in-force articles per language (11,532 chunks across the three languages). Confirm the slice before writing questions. The pipeline takes new SR numbers as input (`--acts`), so changing it is cheap.
- **Exclude** `is_historical` and `is_repealed` chunks from the retrieval index.
- **Benchmark questions** should cite main-text article ids only.
- **Resolve the footnote question** (issue 1) and add the non-official notice wherever texts are displayed (demo, results page).
- **Still open from the Phase 0 task list:** the public GitHub repo (not created, by instruction), the code licence (README still says TBD), and the SCALE / JuRAG reading list.
