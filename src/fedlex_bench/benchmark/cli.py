"""Command line: ``uv run python -m fedlex_bench.benchmark <command> [options]``.

Commands (docs/benchmark_spec.md):

- ``plan``      write ``benchmark/plan/slots.jsonl`` (deterministic, seed 20261015)
- ``brief``     write the writer briefs for the slots (all, or ``--pilot``)
- ``assemble``  merge writer drafts with the plan into candidate items
- ``select``    critic verdicts + candidates -> dev / test / reserve files with final ids
- ``validate``  run every check of spec section 10 (``--freeze`` adds the freeze conditions)

``freeze`` and the review app are not part of this package yet.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from fedlex_bench.benchmark import assemble, brief, plan, select, validate

COMMANDS: dict[str, tuple[Callable[[list[str]], int], str]] = {
    "plan": (plan.run, "write benchmark/plan/slots.jsonl"),
    "brief": (brief.run, "write the briefs the question writers read"),
    "assemble": (assemble.run, "merge writer drafts with the plan into candidates"),
    "select": (select.run, "apply critic verdicts, assign final ids, fill dev/test/reserve"),
    "validate": (validate.run, "check a benchmark folder against spec section 10"),
}


def main(argv: list[str] | None = None) -> int:
    description = __doc__.splitlines()[0] if __doc__ else None
    epilog = "\n".join(f"  {name:<9} {text}" for name, (_, text) in COMMANDS.items())
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
