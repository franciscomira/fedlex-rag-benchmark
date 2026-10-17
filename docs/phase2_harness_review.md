# Harness review before the test run (eval-reviewer, 2026-10-17)

The reviewer (read-only) reported 22 findings: 6 high, 9 medium, 7 low. No gold leakage to the systems under test was found, and ordering and seeds are deterministic. Findings are listed with their disposition; the code fixes are in the commit that follows this file.

| ID | Finding | Disposition |
| --- | --- | --- |
| H1 | MRR used the whole ranked list, not the top 20 of the spec | fixed: MRR on the top 20, test added |
| H2 | parse failures counted as over-refusals, not reported | fixed: over-refusal excludes parse failures; parse-failure and invalid-citation rates reported |
| H3 | judge failures silently dropped from means; judge JSON strict, 1024 output tokens, identical retry | fixed: tolerant JSON extraction, 2000 tokens, different retry, failures counted per judge role and shown |
| H4 | faithfulness excluded answered items without citations (gameable) | fixed: scored 0 and counted; spec amended |
| H5 | test split runnable more than once (`--limit`, `--judge`, `--run-id`, no frozen-list check, log written before run) | fixed: test needs a frozen config hash from `eval/frozen_configs.json`, refuses `--limit`, `--run-id` reuse and existing result folders; log has started/completed |
| H6 | invalid (made-up) citation ids dropped before the trap check | fixed: any invalid id makes a trap fail (also covers invalid pointers, since the answerer merges them; stricter than the spec) |
| M1 | config.json and run hash missed prompt hashes, benchmark checksum, corpus hash, price date, seed | fixed |
| M2 | faithfulness text truncated twice, cutting FR/IT | fixed |
| M3 | latency not reproducible from cache (local time measured live) | documented in the spec: local retrieval time is measured live, model-call latency is replayed |
| M4 | system with an answerer returning no answer silently dropped | fixed: counted as a parse failure |
| M5 | citation P/R mixed refusals in as zeros | fixed: computed on answered items only; spec amended |
| M6 | `compare` did not check split, judge or overlap | fixed |
| M7 | spec items without code (all-gold@k, pointers per refusal, invalid-citation rate, parse-failure rate, CI width column) | implemented the first four; CI shown as an interval in every table cell, spec wording amended |
| M8 | reranker could drop candidates | fixed: missing candidates are appended in their original order |
| M9 | judge retry identical | fixed with H3 |
| L1 | fact verdict indices not validated | fixed: indices must be exactly 0..n-1, otherwise the judge call counts as failed |
| L2 | calibration drops judge failures | open, minor (calibration finished on all 30 items, 0 failures) |
| L3 | stale help text, unused parameter | fixed |
| L4 | n column includes traps | fixed: report shows answerable n |
| L5 | translation group assumes parent in same split | open, minor (checked by the benchmark validator) |
| L6 | division by zero on empty input | fixed in judge; selfpref left |
| L7 | missing tests | added for the fixes above |
