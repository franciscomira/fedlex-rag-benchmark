# Phase 0 report: data foundation

Date: 2026-10-15. Scope: eight acts in German, French and Italian. The three acts from the kickoff (SR 220 Code of Obligations, SR 210 Civil Code, SR 235.1 Data Protection Act) are the alignment check; five related acts were added afterwards to form the corpus slice (section 2).

**Recommendation: GO**, with the adjustments listed at the end. Id alignment is 100% (stop threshold was 95%), the pipeline is deterministic, and 338 tests pass (120 from the original pipeline, 218 added for id resolution, eligibility, entity handling and report-input checks). The 100% figure is largely structural, so the content-level checks below matter more than the headline number.

## 1. Data source and licence

**Source used:** Fedlex open data. SPARQL endpoint `https://fedlex.data.admin.ch/sparqlendpoint` for discovery, filestore for Akoma Ntoso XML. For each act the pipeline picks the consolidation applicable on a fixed as-of date (2026-10-15) and records its URL and consolidation date.

**Source rejected:** the Swiss Case Law Open Dataset on Hugging Face (`voilaj/swiss-caselaw`, CC0). It contains court decisions only. Its 121 data files include no statute files, so it cannot provide article-level federal law.

**Licence.** Fedlex states on <https://fedlex.data.admin.ch/de-CH/home/intro> that texts and metadata are provided under the terms on <https://www.fedlex.admin.ch/de/broadcasters>, which allow reuse for commercial and non-commercial purposes. Conditions come from Art. 49 para. 1 of the Publikationsverordnung (SR 170.512.1):

- content must not be altered;
- it must be visibly distinct from commentary or additions;
- it must carry the notice "Dies ist keine amtliche Veröffentlichung. Massgebend ist allein die Veröffentlichung durch die Bundeskanzlei.";
- it must not suggest an official publication;
- it may only be redistributed for a fee in refined form (condition h).

Consequences for this repo: the README carries the notice, and raw files and the Parquet are not committed. Footnotes are kept in a separate column so that no Fedlex content is dropped (section 6, issue 1). I did not find a Fedlex dataset entry on opendata.swiss that assigns one of its term categories, and the Fedlex "legal notice" page could not be read (JavaScript only). The broadcasters page is treated as the operative source.

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

The five added acts were chosen as the statutes most often applied together with the Code of Obligations: contract and consumer law (221.214.1, 241), insurance contracts (221.229.1, the same domain as the private Confluence RAG work this project replays publicly), and how claims are enforced (272, 281.1). They give a corpus with a clear core (the CO) and neighbouring acts that questions can legitimately cross into. This is a judgement call, not a measured choice.

**Corpus decision (2026-10-15): keep all eight acts.** SR 220 (Code of Obligations) is the core. The other seven (210, 221.214.1, 221.229.1, 235.1, 241, 272, 281.1) are neighbouring acts that questions may legitimately cross into. This was decided at the start of Phase 1, not by an experiment, so it should be revisited if Phase 1 shows that the neighbouring acts add distractors without adding answerable questions.

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

`articles.parquet` has 12,219 rows (4,073 × 3), 12 columns: the seven required (`sr`, `article_id`, `lang`, `title`, `text`, `source_url`, `consolidation_date`) plus `kind`, `is_repealed`, `is_historical`, `block_heading`, `footnotes`.

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
- Paragraph and item counts differ across languages for 5 of 4,073 ids: `210:94` (IT has an extra elided paragraph), `220:644` (IT lacks paragraph 2), `220:1033` (DE has paragraph 2 that FR and IT lack), `221.229.1:95c` (IT has 4 items, DE and FR 3) and `281.1:219` (DE has 14 items, FR and IT 15). The reviewer's check of the first three suggested they are differences in the official texts rather than parsing errors. The last two were checked against the raw XML afterwards (manifest sha256 verified first). Neither is a parsing error in our code:
  - `221.229.1:95c` (IT): the official Italian XML contains the item `art_95_c/para_3/lbl_c` twice, with identical eId and identical text. The extra item is a duplicated element in Fedlex's source file, not a fourth item of the law. Our parser faithfully keeps it, so the IT chunk text repeats its last line ("c. sono autorizzate ad utilizzare la cosa assicurata."). It is the only duplicate of this kind inside an article in the corpus. Nothing was changed in the code: content must not be altered, and removing the duplicate would be a data-cleaning decision. A question built on this article should not rely on the IT item count.
  - `281.1:219` (DE 14 items, FR and IT 15): the FR and IT files put one sentence (the privilege of claims of persons whose assets were under parental administration applies only if bankruptcy was declared during parental authority or within a year) in a separate unnumbered item (`bull_u2`) under class 2, letter a. The DE file has the same sentence inside letter a itself. The content is equivalent in all three languages and nothing is lost. This is a markup difference between the official language versions.
  - Side finding from the same check: the literal string `[tab]` appears in the raw Fedlex XML itself (`272:250` DE and FR, `281.1:219` IT; 3 chunks in total). It is a source artefact that our text keeps.
- Length ratio above 1.6 (longest over shortest of DE/FR/IT, in-force, longest over 200 chars): 10 of 3,844 (six in SR 281.1, plus `220:1075`, `220:644`, `210:336`, `210:339`). They look like ordinary translation differences but have not been read pair by pair.
- Parser fix found while adding the new acts: IT `221.229.1:48-49` has no body and a footnote that says only "Introdotto dal ...", not "Abrogato". It was first flagged in force in IT but repealed in DE and FR. The data smoke test caught the flag mismatch. Any article with no body text is now treated as repealed, with a test.
- Printed number of a range versus a conjunction (second review). Fedlex gives `Art. 50 und 51` and `Art. 135 bis 149` the same kind of eId (`art_50_51`, `art_135_149`), and the three languages print the same eId differently (DE `Art. 463–464`, FR `Art. 463 et 464`, IT `Art. 463 e 464`; IT prints `Art. 697l` alone for `697l-697m`). The id therefore comes from the eId and stays one chunk, such as `210:50-51`; this is deliberate, because splitting by the printed text would give DE, FR and IT different ids for the same eId. All 21 chunks whose printed number is a conjunction in at least one language are adjacent pairs (for example 50 and 51, 663a and 663b), so a range and a conjunction cover the same articles there. What did change in `ingest.py`: the number normaliser no longer turns `und`, `et` and `e` into a range. `normalize_article_numbers("Art. 50 und 52")` gives `["50", "52"]`, never `50-52` (which would claim 51), and an article without an `art_` eId whose printed number is a conjunction becomes one chunk per cited article. The corpus has no such article, so the parquet did not change (the sha256 in section 5 is identical before and after). In the eId check, a printed `50,51` compares equal to the eId-derived `50-51`.
- Five random aligned articles (seed 42) printed side by side read as correct translations. They are in `docs/phase0_alignment.txt`.

Full output is in [phase0_alignment.txt](phase0_alignment.txt). Regenerate with `uv run python -m fedlex_bench.chunk` and `uv run python -m fedlex_bench.alignment`.

## 5. Reproducibility

- Re-running the chunker gives a byte-identical Parquet (sha256 `e6a8ddac5bcabe3b5eb3600b05fc18e1370af3f03310de8ce10c925f3646b75c`). A test rebuilds it from the raw files with the acts and languages passed in reverse order and with the rows shuffled (two seeds) and sorted again, and checks that all four hashes are equal and equal to the file on disk. It is skipped when `data/raw` is missing. The hash did not change when the ingest code was edited for the second review.
- `fedlex_bench.alignment` reports only on the acts in the parquet, and refuses to run (exit code 2) unless the parquet belongs to the manifest: every `source_url` must start with the manifest URL, the consolidation date must agree, the raw file must exist, and its sha256 must equal the manifest value (checked before the raw XML is parsed again).
- The discovery manifest `data/manifest.json` is tracked in git and holds the source URLs, consolidation dates and a sha256 per raw file. Raw files are verified against it on load. Raw file names include the filestore revision.
- The as-of date is fixed, so results do not drift with the day the pipeline runs.
- Caveat: the recorded hashes come from the first download. They prove the files have not changed since, not that they match Fedlex's current content. A spot check on 2026-10-15 re-fetched the five German filestore files used in the link check (SR 220, 210, 221.229.1, 272, 281.1, section 7): all five served bytes match the manifest sha256. The other 19 files were not re-fetched. A fresh clone with no `data/raw` re-downloads whatever the filestore serves for the pinned URLs.
- Python is pinned in `.python-version` (3.13), dependencies in `uv.lock`.

## 6. Known issues

1. **Footnotes** (resolved). They were first dropped from the text, which could conflict with the "no alteration" condition. They are now kept in a separate `footnotes` column (1,665 DE, 1,694 FR and 1,660 IT chunks have one). Whether a footnote exists differs between languages for 97 aligned ids, which looks like language-specific editorial notes, not parsing errors. Notes can be long (up to about 4,900 characters), so do not embed them with the text without checking retrieval quality.
2. **Range ids** (resolved). Repealed ranges such as `210:135-149` are one chunk (42 range ids per language, all repealed, all without text). A lookup of `210:140` used to find nothing. `fedlex_bench.ids` now parses ids, defines how suffixed numbers sort, and resolves `210:140` to `210:135-149` with `resolve_id(id, known_ids)` (or `IdResolver` for many lookups). The number grammar is strict: digits, then at most one letter and/or one Latin ordinal (`257d`, `6bis`, `6abis`, `12ater`); `135ff` or `140xyz` are malformed. All 2,163 distinct numbers in the corpus parse. A range covers only what it names: `135-149` covers the integers 135 to 149, not `140a`, `140z` or `140bis` (the repealed block does not say whether an inserted 140a belongs to it, so such an id resolves to `None`); `226a-226d` covers the letters a to d; `274-274g` covers 274 and 274a to 274g. Unresolvable ids return `None`, and an id covered by two ranges is ambiguous and also returns `None`. Passing one string instead of a list of known ids is a `TypeError`. Retrieval itself was never affected, because range chunks have no text. A corpus-level test checks every range id (every member resolves to it, no member is a chunk of its own, no other chunk lies anywhere in the natural-order span of a range). **Resolving is not the same as being usable as gold**, see section 9.
3. **Transitional ids are positional.** `uN` is the ordinal of the `disp_` block in the snapshot. A new block shifts later ids, so gold labels that cite transitional provisions are only valid for this snapshot. Prefer main-text articles for benchmark questions. The `block_heading` column gives context.
4. **Annexes not chunked.** SR 235.1 annexes 1 and 2 (about 50 to 300 characters each per language) are left out.
5. **Unreviewed translation differences.** Of the five structural mismatches in section 4, `221.229.1:95c` and `281.1:219` were traced to the raw XML (a duplicated source element and a markup difference). The other three rest on the earlier reviewer check. The ten length outliers have not been read pair by pair, and nobody who knows Swiss law has reviewed any of them.
6. **Source URL deep links** (checked, with a caveat). `source_url` is the filestore XML URL plus `#<eId>`. It links to the raw XML file, not to the readable page. See section 7 for the check and the recommendation.
7. **Marginal titles.** In-force chunks with an empty title, the same in DE, FR and IT: 6 (SR 210), 20 (SR 220), 2 (SR 221.214.1), 0 (SR 221.229.1), 2 (SR 235.1), 1 (SR 241), 23 (SR 272) and 2 (SR 281.1), 56 per language in total. The count includes in-force transitional chunks. (An earlier draft gave only three of the acts and "20 to 21" for SR 220.)
8. **Transitional and final provisions in the main text.** The chunker marks only `disp_` blocks as transitional. Several acts put their own transitional, repeal and entry-into-force articles into the main text, under a "Schlussbestimmungen" part. They are in force and stay retrievable, but they are taken out of the benchmark pool by a hand-reviewed list (section 7).
9. **Duplicated item and `[tab]` in the source.** See section 4: `221.229.1:95c` (IT) repeats one item, and three chunks contain a literal `[tab]`. Both come from Fedlex's XML and are kept unchanged.

## 7. Link check and eligibility helper

**Link check (2026-10-15).** Five in-force main-text articles, one from SR 220 and one each from four other acts: `220:321`, `210:480`, `221.229.1:28`, `272:55`, `281.1:143a`. They were picked ad hoc with a throwaway script that is not kept in the repo, so the selection cannot be reproduced from the repo and a sample of five says nothing about the other articles. All requests were read-only GETs and nothing was saved to the repo.

- The `source_url` base (filestore XML) returned 200 with `application/xml` for all five (German). The served bytes match the manifest sha256, and each file contains an `<article>` with the expected eId (`art_321`, `art_480`, `art_28`, `art_55`, `art_143_a`).
- The fragment does not help there. The URL returns plain XML, and `eId` is an ordinary attribute, not `xml:id`, so a viewer has nothing to jump to. Verdict: the link works as a **provenance link to the source file**, but it is not a human-readable deep link.
- The readable page is `https://www.fedlex.admin.ch/eli/<eli path>/<lang>`, for example `https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de#art_321`. Plain HTTP returns 200 for it, but the body is the same 77,119-byte JavaScript shell for every act, so HTTP alone proves nothing about the fragment. I opened all five pages in the built-in browser: the rendered page has an element whose `id` equals the eId (for example `<article id="art_321">`) with the article text, so the eId is a valid fragment target. But the page did not scroll to it in my test (scroll position stayed at 0 after the content had loaded), so a reader lands at the top of the act. I did not test the FR and IT pages, and I did not verify that the readable page shows the same consolidation as our snapshot (the URL without a date shows the current one).
- Recommendation, not implemented because `chunk.py` is out of scope for this change: add a second field with the `fedlex.admin.ch` page URL plus eId for display, and keep the filestore URL as provenance. In the UI, cite the article by SR number and article number, not by link alone.

**Eligibility helper.** `fedlex_bench.eligibility` defines which chunks may be retrieved and which articles may anchor a question. Retrieval-eligible means `kind == "article"`, not repealed, not historical, non-empty text. That gives 3,663 chunks per language, identical in DE, FR and IT (the 3,844 in-force chunks minus 181 in-force transitional or final provisions). Benchmark-eligible means retrieval-eligible in all three languages, at least `min_chars` characters in every language (default 200, a judgement call, not tuned), and not on `TRANSITIONAL_IN_BODY`, a hand-reviewed list of 30 transitional, final and repeal provisions that sit in the main text (below). The pool is defined once on article ids, so it is identical across languages by construction, `pool_rows` raises if a pool id has no eligible row in a language, and `eligibility_mismatches` lists any id that is eligible in only some languages (currently none). Run `uv run python -m fedlex_bench.eligibility` for the counts.

Pool size (articles per act, the same in each language; the three right-hand columns apply the length rule and the exclusion list):

| SR | Retrieval-eligible | min 100 chars | min 200 chars | min 300 chars |
| --- | --- | --- | --- | --- |
| 210 | 1,065 | 1,016 | 822 | 582 |
| 220 | 1,526 | 1,478 | 1,243 | 960 |
| 221.214.1 | 46 | 42 | 34 | 29 |
| 221.229.1 | 98 | 94 | 85 | 74 |
| 235.1 | 77 | 68 | 61 | 50 |
| 241 | 30 | 28 | 20 | 15 |
| 272 | 427 | 378 | 296 | 213 |
| 281.1 | 394 | 375 | 304 | 232 |
| **Total** | **3,663** | **3,479** | **2,865** | **2,155** |

At the default, 798 of the 3,663 retrieval-eligible articles (about a fifth) drop out, 12 of them through the exclusion list. The small acts lose the most in relative terms: SR 241 keeps 20 of 30 and SR 235.1 keeps 61 of 77. The first version of this table (3,502 / 2,877 / 2,162) had no exclusion list; a test now pins the full table.

**Excluded as transitional or final provisions in the main text.** `TRANSITIONAL_IN_BODY` was built by listing, in the raw XML, every main-text article that sits under a part, chapter, section or title called "Schlussbestimmungen" / "Dispositions finales" / "Disposizioni finali" or "Uebergangsbestimmung(en)" (DE, FR and IT headings agree), and then reading each one. The 12 that would otherwise be in the pool at 200 characters (title in German, shortest language in characters):

| Id | Title | Chars |
| --- | --- | --- |
| 221.229.1:103a | Uebergangsbestimmung zur Aenderung vom 19. Juni 2020 | 223 |
| 221.229.1:104 | Inkrafttreten des Gesetzes | 266 |
| 235.1:69 | Uebergangsbestimmungen betreffend laufende Bearbeitungen | 205 |
| 235.1:70 | Uebergangsbestimmung betreffend laufende Verfahren | 248 |
| 235.1:71 | Uebergangsbestimmung betreffend Daten juristischer Personen | 542 |
| 235.1:72 | Uebergangsbestimmung betreffend die Wahl und die Beendigung der Amtsdauer | 365 |
| 272:404 | Weitergelten des bisherigen Rechts | 288 |
| 272:407 | Schiedsgerichtsbarkeit | 571 |
| 272:407b | (no title; transitional provision for the amendment of 20 March 2015) | 430 |
| 272:407c | (no title; amendment of 19 June 2015) | 434 |
| 272:407f | (no title; amendment of 17 March 2023) | 477 |
| 281.1:351 | Inkrafttreten | 370 |

The list has 18 more ids that are below the length limit today (for example 235.1:68, 235.1:73, 235.1:74, 241:28, 241:29, 272:402, 272:403, 272:408, 281.1:352), so it does not depend on `min_chars`. It is deliberately conservative. Not excluded although they sit under "Schlussbestimmungen": `221.229.1:100` and `:101` (relation to the Code of Obligations, excluded legal relationships), `272:400`, `:401` and `:401a` (implementing rules, pilot projects, statistics), and `220:406e`, whose title contains the word "Inkrafttreten" but which is an ordinary contract rule. A test fails if a pool article has a title in any language that starts like a transitional or entry-into-force provision and is neither listed nor reviewed as ordinary. The list belongs to this snapshot. No reviewer who knows Swiss law has checked it.

## 8. Review process

The code was reviewed by a read-only reviewer agent, working from the eval-reviewer role (the project agent was not registered at the time, so a general-purpose agent got its rules inline). It reported 16 findings, including two high-severity ones: 222 superseded SR 210 rows counted as in force, and a manifest ignored by git. All were fixed; the range lookup was fixed afterwards in `ids.py`. The original fixes were verified by re-running tests (120 passed at that time), ruff and the chunker.

A second independent review of `ids.py`, `eligibility.py`, the report and the surrounding code followed. Its findings and what was done:

- Resolving an id was treated as enough to use it as gold: `resolve_gold` now also requires pool membership (section 9).
- The id grammar accepted any letters after the digits: it is now strict (section 6, issue 2), and integer ranges no longer claim suffixed numbers such as `140z`.
- The alignment script reported on every act of the manifest and did not check that the parquet matched it: it now reports on the acts in the parquet and verifies URLs, consolidation date and raw sha256 first.
- Stale empty-title counts, a pool that contained transitional provisions from the main text, an unreproducible sample claim and an unpinned pool table: fixed in sections 6 and 7, and the table is pinned by a test.
- The external-entity test did not test anything. A real test showed that a document with an entity reference crashed the parser (the article text was serialised and parsed a second time). It now copies the element, entities are not expanded, and the test checks that.
- A printed conjunction was normalised to a range: fixed (section 4).
- Smaller test and robustness points: a corpus test that rebuilds from shuffled input and compares the parquet hash, `pool_rows` checking its output, a bare string as known ids rejected, numpy integers accepted as `min_chars`, and the range test asserting that no range member is a chunk of its own and that no chunk lies inside the order span of a range.

The fixes of this second round have not had a third independent review.

## 9. Recommendation: GO, with adjustments

Go: the source is usable, alignment passes the threshold, and the data is reproducible.

Adjustments before Phase 1:

- **Corpus slice.** Decided: keep all eight acts (section 2). The plan targeted the Code of Obligations plus 3 to 5 related acts, so this is wider than planned: seven neighbouring acts. They give 3,844 in-force chunks per language (11,532 across the three languages), of which 3,663 are main-text articles. The pipeline takes new SR numbers as input (`--acts`), so changing the slice is cheap.
- **Exclude** `is_historical`, `is_repealed` and transitional chunks from the retrieval index: use `retrieval_eligible` in `fedlex_bench.eligibility`.
- **Benchmark questions** should take their gold articles from `benchmark_eligible` (main-text articles in force, at least 200 characters in all three languages, not a listed transitional or final provision). A cited id must go through **both steps, in one call**: `fedlex_bench.eligibility.resolve_gold(id, pool_ids, known_ids)` resolves it (normalisation, ranges) and returns it only if the result is in `benchmark_eligible(df)["article_id"]`. `resolve_id` alone is not enough: `210:140` resolves to the repealed range chunk `210:135-149`, and repealed, too short, transitional or listed articles also resolve, but none of them may be a gold label. Pass all chunk ids as `known_ids` so that ranges are found.
- **Add the non-official notice** wherever texts are displayed (demo, results page).
- **Closed since the first draft:** code licence (MIT), footnotes (separate column), and the SCALE / JuRAG reading ([related_work.md](related_work.md)). The SCALE results section was not read, so quote no SCALE numbers until it is.
