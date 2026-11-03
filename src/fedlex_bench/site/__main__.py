"""Command line: ``uv run python -m fedlex_bench.site build [--root DIR] [--out DIR]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fedlex_bench.site.build import DEFAULT_MANIFEST, PROJECT_ROOT, build_report, dump_data
from fedlex_bench.site.examples import load_examples, render_examples
from fedlex_bench.site.render import render_page

ASSETS = Path(__file__).parent / "assets"


def write_site(root: Path, out: Path, manifest: Path | None = None) -> list[str]:
    """Write the results page, css, data, and examples.html if demo/examples.json exists."""
    report = build_report(root, manifest)
    out.mkdir(parents=True, exist_ok=True)
    # Explicit newlines so the committed output does not change with the platform.
    (out / "index.html").write_text(render_page(report), encoding="utf-8", newline="\n")
    css = (ASSETS / "style.css").read_text(encoding="utf-8")
    (out / "style.css").write_text(css, encoding="utf-8", newline="\n")
    (out / "results_data.json").write_text(dump_data(report), encoding="utf-8", newline="\n")
    examples = load_examples(root)
    if examples is not None:
        page = render_examples(examples)
        (out / "examples.html").write_text(page, encoding="utf-8", newline="\n")
    return report.warnings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fedlex_bench.site")
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write site/index.html from the committed harness outputs")
    b.add_argument("--root", type=Path, default=PROJECT_ROOT, help="project root (default: repo)")
    b.add_argument("--out", type=Path, default=None, help="output folder (default: <root>/site)")
    b.add_argument("--manifest", type=Path, default=None,
                   help=f"manifest (default: <root>/{DEFAULT_MANIFEST.as_posix()})")
    b.add_argument("--strict", action="store_true",
                   help="fail if a run id in the manifest has no run folder")
    args = parser.parse_args(argv)
    warnings = write_site(args.root, args.out or args.root / "site", args.manifest)
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    print(f"wrote {(args.out or args.root / 'site') / 'index.html'}")
    if warnings and args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
