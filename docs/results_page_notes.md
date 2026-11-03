# Results page: how it is built

`site/index.html` is generated, then committed. Rebuild with:

```
uv run python -m fedlex_bench.site build          # add --strict to fail on a missing run folder
uv run pytest tests/site
```

The output is `site/index.html`, `site/style.css` and `site/results_data.json` (a copy of every number shown). No script, font or tracker is loaded from anywhere; the page works offline. Building twice gives identical bytes.

## The rule

Every number on the page is read by code from committed harness outputs: `eval/results/<run>/metrics.json` and `per_item.jsonl`, `eval/test_runs.log`, `eval/frozen_configs.json`, `eval/prices.json`, `eval/calibration/agreement_first_pass.json` and `config.toml`. Nothing is copied from the markdown reports. Paired differences use `paired_bootstrap` from `fedlex_bench.eval.stats` with the item selection of `eval compare`, so each cell equals the line that command prints (a test checks this). Prose is written with template fields; a test fails if a digit is typed into it by hand. Trap counts get an exact binomial interval. Latency percentiles are single values and carry no interval, and the page says so.

## The manifest

`site/results_manifest.json` lists the runs (key, run id, label, name), the sections with the runs and paired comparisons each shows, the links, and the reproduce table (commands with `{runs:section}` and `{compares:section}` placeholders, expanded from the manifest). To fill the Phase 4 rows, put the run ids of P1, P2 and P3 into `runs` and rebuild: a run with no id, or whose folder is missing, is shown as "pending" (a missing folder also prints a warning); a test run without a `completed` line in `eval/test_runs.log` is also pending. Comparisons and reproduce commands appear as soon as both runs exist. Paired differences are refused, as in `eval compare`, when two runs differ in split or judge setting; the page shows that instead of a number. The demo link text and URL are in `links.demo`.

## Layout

`src/fedlex_bench/site/`: `data.py` (loading, statistics, formatting), `build.py` (the report, reproduce commands, data dump), `render.py` (HTML), `assets/style.css`, `__main__.py`. Tests: `tests/site/` (numbers against `metrics.json` for a fake project root and for the real data, pending rows, notices, no external URLs, every reproduce command names an existing module, subcommand, option and run folder).

The notices come from `fedlex_bench.serve.notices`. The page shows no statute or answer text.

## Publishing

`.github/workflows/pages.yml` uploads the committed `site/` folder to GitHub Pages and builds nothing. It does nothing until Pages is enabled for the repository (Settings, Pages, Source: GitHub Actions).
