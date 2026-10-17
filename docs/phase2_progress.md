# Phase 2 progress log

Plan: `docs/prompts/phase2_kickoff.md`. Spec: `docs/eval_spec.md` (approved 2026-10-17).

## Status

- Step 0 housekeeping: done (repo already public with tag `benchmark-v1`; budget cap 25 USD in `config.toml`, confirm threshold 5 USD).
- Step 1 eval spec: done, approved with the draft's choices (judge pairing, English gloss for calibration, pointers allowed on refusals).
- Step 2 build: in progress. Lead part done (`src/fedlex_bench/eval/`: metrics, bootstrap, judge roles, runner, guards; interfaces in `src/fedlex_bench/types.py`). Three subagents building `retrieval/`, `llm/` + `answer/`, `agent/` against the interfaces; no live API calls until `.env` exists.
- Still to do in the lead part: CLI (`python -m fedlex_bench.eval run|report`), results table, calibration sheet generator.
- Blocked on the user: Anthropic API key (small prepaid), OpenAI key, Gemini key into `.env`.

## Experiment log

(empty: no experiment run yet)
