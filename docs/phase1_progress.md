# Phase 1 progress notes (working file, not the report)

Last updated: 2026-10-16. Purpose: let a fresh session continue without the long history. The report for readers will be `docs/phase1_report.md`.

## Where things stand

- Spec approved: `docs/benchmark_spec.md`. Code in `src/fedlex_bench/benchmark/` (plan, brief, assemble, select, validate, bm25, structure, twins). 721 tests passed at last full run.
- Plan: `benchmark/plan/slots.jsonl` (258 slots, sha256 511bb7ec...), twins in `benchmark/plan/twins.jsonl`, brief indexes `benchmark/plan/index*.json`. Briefs are git-ignored; regenerate with `uv run python -m fedlex_bench.benchmark brief --slots benchmark/plan/slots.jsonl` (and `--pilot`).
- Round-0 pilot (old plan) archived in `benchmark/archive/pilot_round0/`. Void.
- Round-1 drafts for all 258 slots: `benchmark/drafts/*.jsonl` (11 files). Assembled: `benchmark/assembled/candidates.jsonl`; critic inputs in `benchmark/assembled/critic_in/`.
- Critic round 1: `benchmark/critic/round1-chunk-{1..6}.jsonl`. Result: 219 keep, 39 fix, 0 drop (fix reasons: facts_unsupported 8, other 8, competing_article 7, ambiguous 5, lexical_copy 5, trap_answerable 4, outside_knowledge 2).
- Fix pass in progress: revised records go to `benchmark/drafts_fix/round1-{A,B}.jsonl`; slot lists in `benchmark/critic/fix{A,B}.json`.

## Next steps

1. Merge revised records into the base drafts (replace by slot id), re-assemble, run critic round 2 on the 39 revised slots only (verdicts must carry the new `candidate_sha256` and `"round": 2`).
2. Pilot items: 10 pilot slots are part of the 258; they are not special any more.
3. `select` with `--slots` to build `benchmark/selected/{dev,test}.jsonl` (backfills drops from surplus), then `validate`.
4. Translations for the parallel subset (24 accepted items, 72 translations) only after the human accepts the originals.
5. Review app (Streamlit), human review, freeze to `benchmark/v1/`, BM25 recall@10 on dev, `docs/phase1_report.md`.

## Findings to carry into the report

- Pilot showed briefs lacked act and structure context (fixed), difficulty rubric too generous (fixed), BM25 ranks for zero-score articles and for EN are not informative (set to null).
- Code review found select drift, plan/draft binding, critic/draft binding, twins, article numbers in briefs, cross_ref sampling skew (all fixed; see commits).
- Critic round 1 systemic problems: gold facts the question does not ask for; facts that differ between language versions; neighbouring provisions with a different legal route; many same-language BM25 rank 1 to 3 (about a third; watch the dev BM25 recall check); traps with a half the statute answers; pointer-article citations on traps (Phase 2 scoring rule still open); cross_ref second article not always needed.
- Weaknesses to state: model alias for drafting is not pinned; all drafting and criticism by the same model family; Swiss-law expert has not reviewed anything; dev language counts uneven (de 8); small acts have 5 to 10 items.
