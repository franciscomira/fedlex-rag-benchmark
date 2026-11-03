"""A fake project with every run of the manifest, plus facts and a post template."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from site_fixtures import make_root, write_run

HERE = Path(__file__).parent
FACTS = {"facts": {"corpus_acts": {"value": "8", "source": "test"}}}


def make_project(tmp_path: Path, template: str | None = None) -> tuple[Path, Path]:
    """Root with runs for all manifest keys (P1 to P3 included), facts, an empty allowlist
    and a post template (the fixture one unless ``template`` text is given)."""
    root, manifest = make_root(tmp_path)
    data = json.loads(manifest.read_text("utf-8"))
    for key, rid, base, answers, offset in (
        ("P1", "40_test_p1pppp", 0.78, False, 0.7),
        ("P2", "40_test_p2pppp", 0.84, True, 0.8),
        ("P3", "40_test_p3pppp", 0.86, True, 0.9),
    ):
        write_run(root, rid, base, answers=answers, judge="judge-x" if answers else "none",
                  offset=offset, config_hash=rid[-6:])
        data["runs"][key]["id"] = rid
    manifest.write_text(json.dumps(data), encoding="utf-8")
    folder = root / "docs" / "writeup"
    folder.mkdir(parents=True)
    (folder / "facts.json").write_text(json.dumps(FACTS), encoding="utf-8")
    (folder / "numbers_allowlist.txt").write_text("", encoding="utf-8")
    if template is None:
        shutil.copy(HERE / "fixture_post.template.md", folder / "post.template.md")
    else:
        (folder / "post.template.md").write_text(template, encoding="utf-8")
    return root, manifest
