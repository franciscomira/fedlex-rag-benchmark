"""Command line: ``uv run python -m fedlex_bench.benchmark <command> [options]``.

Commands (docs/benchmark_spec.md):

- ``plan``      write ``benchmark/plan/slots.jsonl`` (deterministic, seed 20261015)
- ``brief``     write the writer briefs for the slots (all, or ``--pilot``)
- ``assemble``  merge writer drafts with the plan into candidate items
- ``select``    critic verdicts + candidates -> dev / test / reserve files with final ids
- ``validate``  run every check of spec section 10 (``--freeze`` adds the freeze conditions)
- ``freeze``    write ``benchmark/v1`` (dev, test, test gold ids, SHA256SUMS)
- ``baseline``  BM25 recall@10 on the dev split, per query language and index
- ``calibration-sample``  pick the 30 calibration items for the machine judges
- ``calibration``         human vs machine judge agreement on the calibration sample
- ``machine-status``      per judge: approve / reject counts, items without a verdict

The review app is ``review_app.py`` (Streamlit): ``uv run --group review streamlit run
src/fedlex_bench/benchmark/review_app.py``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from fedlex_bench.benchmark import (
    assemble,
    baseline,
    brief,
    freeze,
    machine_review,
    plan,
    select,
    validate,
)

COMMANDS: dict[str, tuple[Callable[[list[str]], int], str]] = {
    "plan": (plan.run, "write benchmark/plan/slots.jsonl"),
    "brief": (brief.run, "write the briefs the question writers read"),
    "assemble": (assemble.run, "merge writer drafts with the plan into candidates"),
    "select": (select.run, "apply critic verdicts, assign final ids, fill dev/test/reserve"),
    "validate": (validate.run, "check a benchmark folder against spec section 10"),
    "freeze": (freeze.run, "write benchmark/v1 from the reviewed selection (validate --freeze)"),
    "baseline": (baseline.run, "BM25 recall@10 on the dev split (never the test split)"),
    "calibration-sample": (machine_review.run_sample, "pick the 30 calibration items (seeded)"),
    "calibration": (machine_review.run_calibration, "human vs machine judges on the sample"),
    "machine-status": (machine_review.run_status, "verdict counts per machine judge"),
}


def main(argv: list[str] | None = None) -> int:
    description = __doc__.splitlines()[0] if __doc__ else None
    epilog = "\n".join(f"  {name:<19} {text}" for name, (_, text) in COMMANDS.items())
    ap = argparse.ArgumentParser(
        prog="python -m fedlex_bench.benchmark",
        description=description,
        epilog="commands:\n" + epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("command", choices=list(COMMANDS), metavar="command")
    ap.add_argument(
        "args", nargs=argparse.REMAINDER, help="options of the command (try: <command> --help)"
    )
    ns = ap.parse_args(sys.argv[1:] if argv is None else argv)
    return COMMANDS[ns.command][0](ns.args)
