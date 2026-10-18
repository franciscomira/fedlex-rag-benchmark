# eval/

Everything an evaluation run reads or writes outside the code. Method and metric definitions: [docs/eval_spec.md](../docs/eval_spec.md). Results and discussion: [docs/phase2_report.md](../docs/phase2_report.md).

| Path | What it is | In git |
| --- | --- | --- |
| `results/<run id>/` | `config.json` (system, judge, prompt and data hashes, git sha), `per_item.jsonl`, `metrics.json`; agent runs also `traces/traces.jsonl` | yes |
| `results/selfpref*.json`, `results/sensitivity_*.json` | judge self-preference and second-judge checks | yes |
| `frozen_configs.json` | the configurations (by hash) allowed on the test split, written by `eval freeze` before any test run | yes |
| `test_runs.log` | every test run: `started` and `completed` lines | yes |
| `prices.json` | dated per-million-token prices used for cost | yes |
| `calibration/` | the judge calibration: sample, labels (first pass and revised), agreement reports | yes |
| `cache/` | provider response cache and the monthly spend ledger | no |

Run ids: dev runs are named by hand (`dev_final_B`, ...); test runs are `<date>_test_<config hash>` and cannot be named.

## Commands

```
uv run python -m fedlex_bench.eval run --split dev --method hybrid --config all_lang \
    --rerank claude-haiku-5-5 --retrieve-k 150 --rerank-k 60 --answer claude-haiku-5-5
uv run python -m fedlex_bench.eval run ... --translate claude-haiku-5-5     # translate-then-retrieve
uv run python -m fedlex_bench.eval run ... --agent claude-haiku-5-5         # ADK agent
uv run python -m fedlex_bench.eval report <run ids>
uv run python -m fedlex_bench.eval compare <run a> <run b>                  # paired bootstrap
uv run python -m fedlex_bench.eval freeze --name B <same arguments as run>  # before a test run
uv run python -m fedlex_bench.eval.rejudge --judge gpt-6-luna <run ids>     # second judge, no system rerun
uv run python -m fedlex_bench.eval.selfpref <run ids>                       # judge self-preference
uv run python -m fedlex_bench.eval.calibration sample|label|agree|flag       # judge calibration
uv run python -m fedlex_bench.benchmark.cross_family --model gemini-3.6-flash --dry-run   # Step 3 check
uv run python -m fedlex_bench.agent.failures --traces <traces.jsonl> --items benchmark/v1/test.jsonl
```

## Rules for the test split

`--split test` runs only a frozen configuration, on the whole split, with an automatic run id, never into an existing folder, on a clean git tree (results and the test log are ignored by that check). A configuration that has a `completed` line in `test_runs.log` is refused. A run that crashed or was stopped before completing may be repeated. Never tune on test; dev is for choices.

## Money

Keys come from `.env`. `config.toml` sets the monthly cap (25 USD) and the amount above which a planned run needs confirmation (5 USD). Every paid call is checked against the spend ledger in `cache/ledger.jsonl` first. Responses are cached on disk, so repeated runs of a configuration cost nothing, but failed calls are not cached and are retried.
