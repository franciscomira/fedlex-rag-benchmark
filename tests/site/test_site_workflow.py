"""The Pages workflow only uploads the committed site folder."""

from __future__ import annotations

import re

from fedlex_bench.site.build import PROJECT_ROOT

WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "pages.yml"


def test_workflow_uploads_site_and_builds_nothing():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert re.search(r"upload-pages-artifact@v\d+", text)
    assert re.search(r"^\s+path: site\s*$", text, flags=re.M)
    assert "deploy-pages@" in text
    for forbidden in ("run:", "uv ", "pip ", "python"):
        body = "\n".join(x for x in text.splitlines() if not x.lstrip().startswith("#"))
        assert forbidden not in body, forbidden
    assert "secrets." not in text


def test_committed_site_matches_a_fresh_build(tmp_path):
    """site/ is committed output: it must be what the builder produces from the committed data."""
    from fedlex_bench.site.__main__ import write_site

    committed = PROJECT_ROOT / "site"
    if not (committed / "index.html").exists():
        return
    write_site(PROJECT_ROOT, tmp_path)
    names = ["index.html", "style.css", "results_data.json", "examples.html"]
    if (PROJECT_ROOT / "docs" / "writeup" / "post.template.md").exists():
        names.append("post.html")  # built from the write-up template when there is one
    for name in names:
        assert (tmp_path / name).read_bytes() == (committed / name).read_bytes(), (
            f"site/{name} is stale: run `uv run python -m fedlex_bench.site build`"
        )
