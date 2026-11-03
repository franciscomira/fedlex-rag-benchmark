"""Command line: ``uv run python -m fedlex_bench.writeup build`` (or ``figures``)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fedlex_bench.site.build import DEFAULT_MANIFEST, PROJECT_ROOT, build_report
from fedlex_bench.writeup import numbers
from fedlex_bench.writeup.build import (
    ALLOWLIST,
    FIGURES_DIR,
    POST_HTML,
    POST_MD,
    TEMPLATE,
    build_post,
)
from fedlex_bench.writeup.figures import Figure
from fedlex_bench.writeup.placeholders import BuildError, PlaceholderError, Resolver


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")  # same bytes on every platform


def write_figures(figures: dict[str, Figure], folder: Path) -> list[Path]:
    paths = []
    for name in sorted(figures):
        path = folder / f"{name}.svg"
        write_text(path, figures[name].standalone + "\n")
        paths.append(path)
    return paths


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fedlex_bench.writeup",
        description="Fill docs/writeup/post.template.md from the harness data.",
        epilog=numbers.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, text in (("build", "write post.md, site/post.html and the figures"),
                       ("figures", "write only the three SVG figures")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--root", type=Path, default=PROJECT_ROOT, help="project root")
        p.add_argument("--manifest", type=Path, default=None,
                       help=f"results manifest (default: <root>/{DEFAULT_MANIFEST.as_posix()})")
        p.add_argument("--out-figures", type=Path, default=None,
                       help=f"figure folder (default: <root>/{FIGURES_DIR.as_posix()})")
        if name == "build":
            p.add_argument("--template", type=Path, default=None,
                           help=f"template (default: <root>/{TEMPLATE.as_posix()})")
            p.add_argument("--out-md", type=Path, default=None,
                           help=f"Markdown output (default: <root>/{POST_MD.as_posix()})")
            p.add_argument("--out-html", type=Path, default=None,
                           help=f"HTML output (default: <root>/site/{POST_HTML})")
            p.add_argument("--allowlist", type=Path, default=None,
                           help=f"allowlist (default: <root>/{ALLOWLIST.as_posix()})")
            p.add_argument("--check-numbers", action="store_true",
                           help="after building, list numbers typed by hand that are not in "
                                "the allowlist, and exit 1 if there are any")
    return parser


def run_figures(args: argparse.Namespace) -> int:
    report = build_report(args.root, args.manifest)
    try:
        figures = Resolver(report).figures()
    except PlaceholderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for path in write_figures(figures, args.out_figures or args.root / FIGURES_DIR):
        print(f"wrote {path}")
    return 0


def run_build(args: argparse.Namespace) -> int:
    root: Path = args.root
    template = args.template or root / TEMPLATE
    if not template.exists():
        print(f"skipped: template {template} not found; nothing built", file=sys.stderr)
        return 0
    report = build_report(root, args.manifest)
    for w in report.warnings:
        print(f"warning: {w}", file=sys.stderr)
    allowlist = args.allowlist or root / ALLOWLIST
    try:
        post = build_post(report, template.read_text(encoding="utf-8"), root,
                          source=template.relative_to(root).as_posix()
                          if template.is_relative_to(root) else template.name,
                          allowlist=allowlist,
                          has_examples=(root / "demo" / "examples.json").exists())
    except BuildError as exc:
        for line in exc.errors:
            print(f"error: {line}", file=sys.stderr)
        print(f"build failed: {len(exc.errors)} problem(s), nothing written", file=sys.stderr)
        return 1
    write_text(args.out_md or root / POST_MD, post.markdown)
    write_text(args.out_html or root / "site" / POST_HTML, post.html)
    write_figures(post.figures, args.out_figures or root / FIGURES_DIR)
    print(f"wrote {args.out_md or root / POST_MD}")
    print(f"wrote {args.out_html or root / 'site' / POST_HTML}")
    if args.check_numbers and post.offences:
        name = template.name
        for o in post.offences:
            print(f"{name}:{o.line}: number '{o.token}' is typed by hand and not in "
                  f"{allowlist.name}: {o.text}", file=sys.stderr)
        print(f"{len(post.offences)} number(s) not allowed", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    code = run_build(args) if args.cmd == "build" else run_figures(args)
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
