# Phase 1 progress notes (working file, not the report)

Last updated: 2026-10-16. Purpose: let a fresh session continue without the long history. The report for readers will be `docs/phase1_report.md`.

## Where things stand (2026-10-16)

- Spec approved: `docs/benchmark_spec.md`. Package `src/fedlex_bench/benchmark/` (plan, brief, assemble, select, validate, bm25, structure, twins, review_app, freeze, baseline). Full test suite passes, ruff clean.
- Plan `benchmark/plan/slots.jsonl` (258 slots, sha256 511bb7ec...). Drafts in `benchmark/drafts/`, critic verdicts in `benchmark/critic/` (rounds 1 to 5; round 1: 219 keep, 39 fix, 0 drop; all fixes resolved by round 5).
- Selected benchmark candidates: `benchmark/selected/{dev,test}.jsonl` = 170 items (150 answerable + 20 traps; dev 45, test 125), validator clean, all `review_status: unreviewed`. Reserve of 88 kept-but-unselected items in `benchmark/reserve/`.
- BM25 recall@10 on dev (first run): same-language DE 25%, FR 45%, IT 36%; EN 0%; all-language index 30% overall (n=40).

## Next steps

1. Human review in the Streamlit app: `uv run --group review streamlit run src/fedlex_bench/benchmark/review_app.py`. Rejections need backfill from `benchmark/reserve/` in the same cell (not automated yet).
2. After acceptance: parallel translated subset (24 accepted single items, 6 per query language, 6 dev + 18 test, 72 translations, critic check on meaning, human review).
3. eval-reviewer pass on the data (leakage, verbatim quoting, unsupported gold, imbalance), then `freeze` to `benchmark/v1/`, tag `benchmark-v1`, write `docs/phase1_report.md`.

## Findings to carry into the report

- Pilot showed briefs lacked act and structure context (fixed), difficulty rubric too generous (fixed), BM25 ranks for zero-score articles and for EN are not informative (set to null).
- Code review found select drift, plan/draft binding, critic/draft binding, twins, article numbers in briefs, cross_ref sampling skew (all fixed; see commits).
- Critic round 1 systemic problems: gold facts the question does not ask for; facts that differ between language versions; neighbouring provisions with a different legal route; many same-language BM25 rank 1 to 3 (about a third; watch the dev BM25 recall check); traps with a half the statute answers; pointer-article citations on traps (Phase 2 scoring rule still open); cross_ref second article not always needed.
- Weaknesses to state: model alias for drafting is not pinned; all drafting and criticism by the same model family; Swiss-law expert has not reviewed anything; dev language counts uneven (de 8); small acts have 5 to 10 items.
