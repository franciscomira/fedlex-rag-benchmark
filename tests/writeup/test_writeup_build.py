"""The build: outputs, determinism, notices, no external requests, site integration, real data."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import pytest
from writeup_fixtures import make_project

from fedlex_bench.serve.notices import NOT_LEGAL_ADVICE, NOT_OFFICIAL_DE, NOT_OFFICIAL_EN
from fedlex_bench.site.__main__ import write_site
from fedlex_bench.site.build import PROJECT_ROOT, build_report
from fedlex_bench.writeup.__main__ import main
from fedlex_bench.writeup.build import (
    ALLOWLIST,
    FIGURES_DIR,
    POST_HTML,
    POST_MD,
    TEMPLATE,
    build_post,
)
from fedlex_bench.writeup.figures import FIGURE_NAMES
from fedlex_bench.writeup.numbers import check_numbers, load_allowlist
from fedlex_bench.writeup.placeholders import Resolver, tokenize

ALLOWED_LINK_HOSTS = {"github.com", "huggingface.co", "www.fedlex.admin.ch"}


def run_build(root: Path, *extra: str) -> None:
    main(["build", "--root", str(root), "--manifest", str(root / "site/results_manifest.json"),
          *extra])


def outputs(root: Path) -> dict[str, bytes]:
    files = [root / POST_MD, root / "site" / POST_HTML,
             *sorted((root / FIGURES_DIR).glob("*.svg"))]
    return {f.relative_to(root).as_posix(): f.read_bytes() for f in files}


def test_build_writes_markdown_html_and_three_figures(tmp_path):
    root, _ = make_project(tmp_path)
    run_build(root)
    out = outputs(root)
    assert set(out) == {"docs/writeup/post.md", "site/post.html",
                        *(f"docs/writeup/figures/{n}.svg" for n in FIGURE_NAMES)}
    md = out["docs/writeup/post.md"].decode("utf-8")
    assert "![" in md and "](figures/fig_recall.svg)" in md  # relative link to the figure
    assert "| System |" in md  # the generated pipe table
    html = out["site/post.html"].decode("utf-8")
    assert "<svg" in html and "fig_recall.svg" not in html  # figures are inlined
    assert "<title>Fixture post</title>" in html and "<h1>Fixture post</h1>" in html


def test_markdown_numbers_equal_the_html_numbers(tmp_path):
    root, _ = make_project(tmp_path)
    run_build(root)
    md = (root / POST_MD).read_text("utf-8")
    html = (root / "site" / POST_HTML).read_text("utf-8")
    plain = re.sub(r"<[^>]+>", "", html)
    for value in re.findall(r"\d\.\d\d \[\d\.\d\d, \d\.\d\d\]", md):
        assert value in plain


def test_rebuild_is_byte_identical(tmp_path):
    root, _ = make_project(tmp_path)
    run_build(root)
    first = outputs(root)
    run_build(root)
    assert outputs(root) == first


def test_missing_template_is_skipped_with_a_message(tmp_path, capsys):
    root, _ = make_project(tmp_path)
    (root / TEMPLATE).unlink()
    run_build(root)
    err = capsys.readouterr().err
    assert "skipped" in err and "post.template.md" in err
    assert not (root / POST_MD).exists() and not (root / "site" / POST_HTML).exists()


def test_failed_build_writes_nothing_and_exits_nonzero(tmp_path, capsys):
    root, _ = make_project(tmp_path, template="# T\n\nbad {{v NOPE recall@10}}\n")
    with pytest.raises(SystemExit) as exc:
        run_build(root)
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "post.template.md:3: {{v NOPE recall@10}}: unknown run 'NOPE'" in err
    assert not (root / POST_MD).exists() and not (root / "site" / POST_HTML).exists()


def test_notices_verbatim_in_markdown_and_html(tmp_path):
    root, _ = make_project(tmp_path)
    run_build(root)
    md = (root / POST_MD).read_text("utf-8")
    html = (root / "site" / POST_HTML).read_text("utf-8")
    for text in (NOT_OFFICIAL_EN, NOT_OFFICIAL_DE, NOT_LEGAL_ADVICE):
        assert md.count(text) == 2, text  # after the title and at the end
        assert html.count(text) == 2, text  # header and footer of the shared shell
    assert html.index(NOT_OFFICIAL_EN) < html.index("<h1>")
    assert md.index(NOT_OFFICIAL_EN) < md.index("T1 reaches")


class Refs(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.refs: list[tuple[str, str, str]] = []
        self.tags: set[str] = set()

    def handle_starttag(self, tag, attrs):
        self.tags.add(tag)
        for name, value in attrs:
            if value and (name in ("src", "srcset", "action", "poster", "data", "href")
                          or name.endswith(":href")):
                self.refs.append((tag, name, value))


def test_no_external_resources_in_the_page(tmp_path):
    template = ("# T\n\nA [plain link](https://github.com/x/y) and [local](index.html).\n\n"
                "{{figure fig_cost_latency}}\n\n{{table by_language}}\n")
    root, _ = make_project(tmp_path, template=template)
    run_build(root)
    html = (root / "site" / POST_HTML).read_text("utf-8")
    p = Refs()
    p.feed(html)
    assert not p.tags & {"script", "iframe", "img", "video", "audio", "form", "object", "embed",
                         "use", "image"}
    for tag, name, value in p.refs:
        parsed = urlparse(value)
        if tag == "a" and name == "href":
            if parsed.scheme:
                assert parsed.scheme == "https" and parsed.netloc in ALLOWED_LINK_HOSTS, value
            else:
                assert value.startswith("#") or value in ("index.html", "examples.html"), value
        else:  # link (the stylesheet) and anything else must be a local file
            assert not parsed.scheme and not parsed.netloc and not value.startswith("//"), value
    # no request can start from CSS or SVG either
    assert "url(" not in html and "@import" not in html
    assert not re.findall(r'(?:src|href)="(?!#|index|examples|style|https://github)', html)
    # an http(s) address appears only as a link target, never elsewhere
    assert html.count("http") == html.count('href="https://github.com/x/y"')


def test_standalone_figures_have_no_external_references(tmp_path):
    root, _ = make_project(tmp_path)
    run_build(root)
    for path in sorted((root / FIGURES_DIR).glob("*.svg")):
        text = path.read_text("utf-8")
        assert text.count("http") == 1 and 'xmlns="http://www.w3.org/2000/svg"' in text
        assert "url(" not in text and "@import" not in text and "href" not in text


def test_javascript_links_are_refused(tmp_path):
    root, _ = make_project(tmp_path, template="# T\n\n[x](javascript:alert)\n")
    from fedlex_bench.writeup.placeholders import BuildError

    with pytest.raises(BuildError):
        build_post(build_report(root, root / "site/results_manifest.json"),
                   (root / TEMPLATE).read_text("utf-8"), root)


def test_site_build_also_builds_post_and_links_it(tmp_path):
    root, manifest = make_project(tmp_path)
    (root / "demo").mkdir()
    (root / "demo" / "examples.json").write_text(
        '{"recorded_at":"d","model":"m","total_cost_usd":0.1,"examples":[]}', encoding="utf-8")
    out = tmp_path / "out"
    write_site(root, out, manifest)
    assert (out / "post.html").exists()
    for name in ("index.html", "examples.html"):
        assert '<a href="post.html">Write-up</a>' in (out / name).read_text("utf-8"), name
    post = (out / "post.html").read_text("utf-8")
    assert '<a href="index.html">Results</a>' in post
    assert '<a href="examples.html">Recorded examples</a>' in post
    write_site(root, tmp_path / "out2", manifest)
    assert (tmp_path / "out2" / "post.html").read_bytes() == (out / "post.html").read_bytes()


def test_site_build_without_template_has_no_post_and_no_link(tmp_path):
    root, manifest = make_project(tmp_path)
    (root / TEMPLATE).unlink()
    out = tmp_path / "out"
    write_site(root, out, manifest)
    assert not (out / "post.html").exists()
    assert "post.html" not in (out / "index.html").read_text("utf-8")


def test_site_build_fails_on_a_bad_template(tmp_path):
    root, manifest = make_project(tmp_path, template="# T\n\n{{v NOPE x}}\n")
    with pytest.raises(SystemExit) as exc:
        write_site(root, tmp_path / "out", manifest)
    assert "unknown run 'NOPE'" in str(exc.value)


# ---------------------------------------------------------------- the real project

REAL_TEMPLATE = PROJECT_ROOT / TEMPLATE
real = pytest.mark.skipif(not REAL_TEMPLATE.exists(), reason="docs/writeup/post.template.md absent")


def real_post():
    report = build_report(PROJECT_ROOT)
    return build_post(report, REAL_TEMPLATE.read_text("utf-8"), PROJECT_ROOT,
                      allowlist=PROJECT_ROOT / ALLOWLIST,
                      has_examples=(PROJECT_ROOT / "demo" / "examples.json").exists())


@real
def test_real_post_builds_and_every_number_is_a_placeholder_or_allowlisted():
    post = real_post()
    assert not post.offences, "\n".join(
        f"post.template.md:{o.line}: '{o.token}' {o.text}" for o in post.offences)


@real
def test_committed_post_files_match_a_fresh_build():
    post = real_post()
    committed = {
        PROJECT_ROOT / POST_MD: post.markdown,
        PROJECT_ROOT / "site" / POST_HTML: post.html,
        **{PROJECT_ROOT / FIGURES_DIR / f"{n}.svg": f.standalone + "\n"
           for n, f in post.figures.items()},
    }
    for path, text in committed.items():
        assert path.read_bytes() == text.encode("utf-8"), (
            f"{path.relative_to(PROJECT_ROOT)} is stale: run "
            "`uv run python -m fedlex_bench.writeup build`")


def test_real_figures_match_the_committed_files_when_present():
    folder = PROJECT_ROOT / FIGURES_DIR
    if not folder.exists() or not (PROJECT_ROOT / "site" / "results_manifest.json").exists():
        pytest.skip("no committed figures")
    figures = Resolver(build_report(PROJECT_ROOT)).figures()
    for name, fig in figures.items():
        assert (folder / f"{name}.svg").read_bytes() == (fig.standalone + "\n").encode("utf-8")


def test_check_numbers_uses_the_template_lines_not_the_filled_text(tmp_path):
    root, _ = make_project(tmp_path, template="# T\n\nValue {{v T1 recall@10}} and 77 typed.\n")
    report = build_report(root, root / "site/results_manifest.json")
    tok = tokenize((root / TEMPLATE).read_text("utf-8"), Resolver(report, root / "x"), "t")
    offences = check_numbers(tok.lines, load_allowlist(root / ALLOWLIST))
    assert [(o.line, o.token) for o in offences] == [(3, "77")]
