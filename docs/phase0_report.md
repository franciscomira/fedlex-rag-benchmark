# Phase 0 report: data foundation

Date: 2026-10-15. Scope: three acts (SR 220 Code of Obligations, SR 210 Civil Code, SR 235.1 Data Protection Act) in German, French and Italian.

**Recommendation: GO**, with the adjustments listed at the end. Id alignment is 100% (stop threshold was 95%), the pipeline is deterministic, and 106 tests pass. The 100% figure is largely structural, so the content-level checks below matter more than the headline number.

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

The old 1992 Data Protection Act (repealed 2023-08-31) is excluded by selection.

## 3. Counts (identical for DE, FR and IT)

| SR | Chunks | Main-text articles | Transitional/final | Repealed or elided | Historical | In force |
| --- | --- | --- | --- | --- | --- | --- |
| 210 | 1,277 | 1,099 | 178 | 40 | 74 | 1,163 |
| 220 | 1,690 | 1,602 | 88 | 91 | 0 | 1,599 |
| 235.1 | 77 | 77 | 0 | 0 | 0 | 77 |
| **Total per language** | **3,044** | **2,778** | **266** | **131** | **74** | **2,839** |

`articles.parquet` has 9,132 rows (3,044 × 3), 11 columns: the seven required (`sr`, `article_id`, `lang`, `title`, `text`, `source_url`, `consolidation_date`) plus `kind`, `is_repealed`, `is_historical`, `block_heading`.

- *Repealed* chunks are kept with empty text and `is_repealed=True`, not dropped. The flag also covers elided ("…") and obsolete provisions.
- *Historical* are 74 articles in SR 210 (block "Wortlaut der früheren Bestimmungen", the pre-1988 marital property text). They are flagged `is_historical=True` and should be excluded from retrieval, otherwise they compete with current law.
- Transitional chunks include 5 SR 220 blocks without `<article>` elements, chunked whole.

## 4. Alignment

| SR | Aligned ids | Union | Rate |
| --- | --- | --- | --- |
| 210 | 1,277 | 1,277 | 100.00% |
| 220 | 1,690 | 1,690 | 100.00% |
| 235.1 | 77 | 77 | 100.00% |
| All | 3,044 | 3,044 | 100.00% |

In-force chunks only: 2,839 of 2,839. No id is missing in any language.

**What this does and does not show.** Ids come from Fedlex eIds, which Fedlex derives from the document structure. So 100% shows that the three XML files share the same structure. It does not show that the text behind an id is the same provision. An independent reviewer made this point, and the report now includes content-level checks:

- Number check: the eId-derived number matches the printed number in 9,116 of 9,117 articles. The exception is `220:697l-697m` in Italian, a markup quirk.
- Paragraph and item counts differ across languages for 3 of 3,044 ids: `210:94` (IT has an extra elided paragraph), `220:644` (IT lacks paragraph 2) and `220:1033` (DE has paragraph 2 that FR and IT lack). These are differences in the official texts, not parsing errors.
- Length ratio above 1.6 (longest over shortest of DE/FR/IT, in-force, longest over 200 chars): 4 of 2,839 (`220:1075`, `220:644`, `210:336`, `210:339`). They look like ordinary translation differences but have not been read pair by pair.
- Five random aligned articles (seed 42) printed side by side read as correct translations. They are in `docs/phase0_alignment.txt`.

Full output is in [phase0_alignment.txt](phase0_alignment.txt). Regenerate with `uv run python -m fedlex_bench.chunk` and `uv run python -m fedlex_bench.alignment`.

## 5. Reproducibility

- Re-running the chunker gives a byte-identical Parquet (sha256 `bc310d686ece0ffb94d6b620e5d32ca251a1e5c6ef3173d5160580d22dff20b4`).
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

The code was reviewed by a read-only reviewer agent, working from the eval-reviewer role (the project agent was not registered at the time, so a general-purpose agent got its rules inline). It reported 16 findings, including two high-severity ones: 222 superseded SR 210 rows counted as in force, and a manifest ignored by git. All were fixed except the range lookup. The fixes were verified by re-running tests (106 pass), ruff and the chunker. The fixed code has not had a second independent review.

## 8. Recommendation: GO, with adjustments

Go: the source is usable, alignment passes the threshold, and the data is reproducible.

Adjustments before Phase 1:

- **Corpus slice.** The plan targets the Code of Obligations plus 3 to 5 related acts. The three acts here give 2,839 in-force articles per language. Choose the extra acts (for example the Civil Procedure Code, the Criminal Code or the Consumer Information Act) before writing questions. The pipeline takes new SR numbers as input.
- **Exclude** `is_historical` and `is_repealed` chunks from the retrieval index.
- **Benchmark questions** should cite main-text article ids only.
- **Resolve the footnote question** (issue 1) and add the non-official notice wherever texts are displayed (demo, results page).
- **Still open from the Phase 0 task list:** the public GitHub repo (not created, by instruction), the code licence (README still says TBD), and the SCALE / JuRAG reading list.
