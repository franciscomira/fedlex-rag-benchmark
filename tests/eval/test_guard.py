import json
import subprocess

import pandas as pd
import pytest

from fedlex_bench.eval.guard import TestRunRefused, article_text_lookup, log_test_run, require_clean_tree


def test_article_text_lookup_returns_all_languages(tmp_path):
    df = pd.DataFrame(
        [
            {"article_id": "220:1", "lang": "de", "text": "Eins"},
            {"article_id": "220:1", "lang": "fr", "text": "Un"},
            {"article_id": "220:2", "lang": "de", "text": "Zwei"},
        ]
    )
    path = tmp_path / "a.parquet"
    df.to_parquet(path)
    lookup = article_text_lookup(path)
    got = lookup(["220:1"])
    assert "Eins" in got["220:1"] and "Un" in got["220:1"]
    assert lookup(["999:9"]) == {"999:9": ""}


def test_log_test_run_appends_and_refuses_duplicate_hash(tmp_path):
    log = tmp_path / "test_runs.log"
    log_test_run(log, run_id="r1", config_hash="h1", date="2026-10-17")
    log_test_run(log, run_id="r2", config_hash="h2", date="2026-10-17")
    assert len(log.read_text().splitlines()) == 2
    assert json.loads(log.read_text().splitlines()[0])["config_hash"] == "h1"
    with pytest.raises(TestRunRefused):
        log_test_run(log, run_id="r3", config_hash="h1", date="2026-10-18")


def test_require_clean_tree(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "f").write_text("x")
    with pytest.raises(TestRunRefused):
        require_clean_tree(tmp_path)
