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
    """Write the results page, css, data, examples.html if demo/examples.json exists, and
    post.html if docs/writeup/post.template.md exists (a failing template fails the build)."""
    report = build_report(root, manifest)
    examples = load_examples(root)
    post = build_post_html(root, report, has_examples=examples is not None)
    out.mkdir(parents=True, exist_ok=True)
    # Explicit newlines so the committed output does not change with the platform.
    (out / "index.html").write_text(
        render_page(report, has_post=post is not None), encoding="utf-8", newline="\n")
    css = (ASSETS / "style.css").read_text(encoding="utf-8")
    (out / "style.css").write_text(css, encoding="utf-8", newline="\n")
    (out / "results_data.json").write_text(dump_data(report), encoding="utf-8", newline="\n")
    if examples is not None:
        page = render_examples(examples, has_post=post is not None)
        (out / "examples.html").write_text(page, encoding="utf-8", newline="\n")
    if post is not None:
        (out / "post.html").write_text(post, encoding="utf-8", newline="\n")
    return report.warnings


def build_post_html(root: Path, report, has_examples: bool) -> str | None:
    """The write-up page, or None when the project has no post template."""
    from fedlex_bench.writeup.build import TEMPLATE, build_post
    from fedlex_bench.writeup.placeholders import BuildError

    template = root / TEMPLATE
    if not template.exists():
        return None
    try:
        return build_post(report, template.read_text(encoding="utf-8"), root,
                          has_examples=has_examples).html
    except BuildError as exc:
        raise SystemExit("write-up build failed:\n" + "\n".join(exc.errors)) from exc


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
