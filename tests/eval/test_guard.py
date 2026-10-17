import json
import subprocess

import pandas as pd
import pytest

from fedlex_bench.eval.guard import (
    TestRunRefused,
    article_text_lookup,
    log_test_done,
    log_test_start,
    require_clean_tree,
    require_frozen,
)


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


def test_test_log_allows_rerun_after_a_crash_but_not_after_completion(tmp_path):
    log = tmp_path / "test_runs.log"
    log_test_start(log, run_id="r1", config_hash="h1", date="2026-10-17")
    log_test_start(log, run_id="r1b", config_hash="h1", date="2026-10-17")  # r1 never completed
    log_test_done(log, run_id="r1b", config_hash="h1", date="2026-10-17")
    with pytest.raises(TestRunRefused):
        log_test_start(log, run_id="r3", config_hash="h1", date="2026-10-18")
    log_test_start(log, run_id="r4", config_hash="h2", date="2026-10-18")


def test_require_frozen_accepts_only_listed_hashes(tmp_path):
    f = tmp_path / "frozen.json"
    f.write_text(json.dumps({"B": {"hash": "abc"}}))
    assert require_frozen(f, "abc") == "B"
    with pytest.raises(TestRunRefused):
        require_frozen(f, "zzz")
    with pytest.raises(TestRunRefused):
        require_frozen(tmp_path / "missing.json", "abc")


def test_require_clean_tree(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "f").write_text("x")
    with pytest.raises(TestRunRefused):
        require_clean_tree(tmp_path)
