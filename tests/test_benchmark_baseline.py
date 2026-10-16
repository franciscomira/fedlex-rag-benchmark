"""``baseline``: BM25 recall@10 on the dev split."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from benchmark_fixtures import COLUMNS, _row
from fedlex_bench.benchmark import baseline, cli
from fedlex_bench.benchmark.common import sha256_file, write_jsonl

N = 30


def build_corpus() -> pd.DataFrame:
    """30 articles in act 220. Every language has its own unique word per article
    (``dewort5``, ``frmot5``, ``itparola5``) and every text shares the filler ``gemeinsam``."""
    rows = []
    for n in range(1, N + 1):
        for lang, word in (("de", "dewort"), ("fr", "frmot"), ("it", "itparola")):
            rows.append(_row("220", str(n), lang, f"{word}{n} gemeinsam text"))
    # a repealed article is not in any index
    rows.append(_row("220", "99", "de", "repealwort99 gemeinsam", is_repealed=True))
    return pd.DataFrame(rows, columns=COLUMNS)


def item(i, question, gold, lang="de", qtype="single", **kw):
    it = {
        "id": f"v1-{i:04d}",
        "split": "dev",
        "qtype": qtype,
        "is_trap": qtype == "trap",
        "query_lang": lang,
        "question": question,
        "gold_article_ids": gold,
    }
    it.update(kw)
    return it


@pytest.fixture(scope="module")
def corpus():
    return build_corpus()


@pytest.fixture(scope="module")
def indexes(corpus):
    return baseline.build_all_indexes(corpus)


def test_single_language_index_finds_only_its_own_language(indexes):
    it = item(1, "dewort5", ["220:5"], lang="de")
    assert baseline.item_recall(it, indexes) == {"de": 1.0, "fr": 0.0, "it": 0.0, "all": 1.0}


def test_zero_score_gold_is_not_a_hit_even_though_it_sorts_into_the_top_10(indexes):
    # article 3 would be in the top 10 of the fr index by the tie-break alone
    it = item(1, "dewort3", ["220:3"], lang="de")
    assert baseline.item_recall(it, indexes)["fr"] == 0.0


def test_english_query_is_cross_lingual_by_construction(indexes):
    it = item(1, "what is the rule", ["220:5"], lang="en")
    assert baseline.item_recall(it, indexes) == {"de": 0.0, "fr": 0.0, "it": 0.0, "all": 0.0}


def test_all_index_hits_if_any_language_version_ranks(indexes):
    it = item(1, "itparola7", ["220:7"], lang="it")
    assert baseline.item_recall(it, indexes) == {"de": 0.0, "fr": 0.0, "it": 1.0, "all": 1.0}


def test_gold_below_rank_10_is_a_miss(indexes):
    # 'gemeinsam' hits every document equally; the 12th article by tie-break is rank 12 of 30
    it = item(1, "gemeinsam", ["220:12"], lang="de")
    assert baseline.item_recall(it, indexes)["de"] == 0.0
    it = item(1, "gemeinsam", ["220:2"], lang="de")
    assert baseline.item_recall(it, indexes)["de"] == 1.0


def test_cross_ref_counts_recall_over_the_gold_set(indexes):
    it = item(1, "dewort5", ["220:5", "220:6"], lang="de", qtype="cross_ref")
    assert baseline.item_recall(it, indexes)["de"] == 0.5
    it = item(1, "dewort5 dewort6", ["220:5", "220:6"], lang="de", qtype="cross_ref")
    assert baseline.item_recall(it, indexes)["de"] == 1.0


def test_all_index_counts_gold_ids_not_language_versions(indexes):
    # 'gemeinsam' ties every short document: the top 10 are the de/fr/it versions of articles 1
    # to 3 plus one version of article 4, so article 5 is out
    it = item(1, "gemeinsam", ["220:1", "220:5"], lang="de", qtype="cross_ref")
    assert baseline.item_recall(it, indexes)["all"] == 0.5
    it = item(1, "gemeinsam", ["220:1", "220:3"], lang="de", qtype="cross_ref")
    assert baseline.item_recall(it, indexes)["all"] == 1.0


def test_summary_per_query_language_and_overall_excludes_traps(indexes):
    items = [
        item(1, "dewort5", ["220:5"], lang="de"),
        item(2, "dewort6", ["220:6"], lang="de"),
        item(3, "nothing here", ["220:7"], lang="de"),
        item(4, "frmot8", ["220:8"], lang="fr"),
        item(5, "how", [], lang="en", qtype="trap"),
        item(6, "english question", ["220:9"], lang="en"),
    ]
    out = baseline.summarise(items, indexes)
    assert out["n_items"] == 5 and out["n_traps_excluded"] == 1
    r = out["recall_at_10"]
    assert r["de"]["de"] == {"n": 3, "recall": round(2 / 3, 4)}
    assert r["fr"]["fr"] == {"n": 1, "recall": 1.0}
    assert r["de"]["fr"] == {"n": 1, "recall": 0.0}
    assert r["de"]["en"] == {"n": 1, "recall": 0.0}
    assert r["de"]["overall"] == {"n": 5, "recall": round(2 / 5, 4)}
    assert r["all"]["overall"] == {"n": 5, "recall": round(3 / 5, 4)}


def test_an_empty_query_language_has_n_zero_and_null_recall(indexes):
    out = baseline.summarise([item(1, "dewort5", ["220:5"])], indexes)
    assert out["recall_at_10"]["de"]["it"] == {"n": 0, "recall": None}


# ------------------------------------------------------------------ command


@pytest.fixture
def selected(tmp_path):
    folder = tmp_path / "selected"
    items = [
        item(1, "dewort5", ["220:5"], lang="de"),
        item(2, "frmot6", ["220:6"], lang="fr"),
        item(3, "how", [], lang="en", qtype="trap"),
    ]
    write_jsonl(folder / "dev.jsonl", items)
    write_jsonl(folder / "test.jsonl", [dict(item(9, "dewort1", ["220:1"]), split="test")])
    return folder


@pytest.fixture(scope="module")
def parquet(tmp_path_factory, corpus):
    path = tmp_path_factory.mktemp("corpus") / "articles.parquet"
    corpus.to_parquet(path, index=False)
    return path


def test_command_prints_a_table_and_writes_json(selected, parquet, tmp_path, capsys):
    out = tmp_path / "baseline_dev.json"
    code = cli.main(
        ["baseline", "--dir", str(selected), "--parquet", str(parquet), "--out", str(out)]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "cross-lingual by construction" in printed
    assert "recall@10" in printed and "overall" in printed
    data = json.loads(out.read_bytes())
    assert data["split"] == "dev" and data["k"] == 10
    assert data["n_items"] == 2 and data["n_traps_excluded"] == 1
    assert data["dev_sha256"] == sha256_file(selected / "dev.jsonl")
    assert data["recall_at_10"]["de"]["de"] == {"n": 1, "recall": 1.0}
    assert data["recall_at_10"]["all"]["overall"] == {"n": 2, "recall": 1.0}
    assert "cross-lingual by construction" in data["note"]
    raw = out.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")


def test_command_is_idempotent(selected, parquet, tmp_path):
    out = tmp_path / "b.json"
    argv = ["baseline", "--dir", str(selected), "--parquet", str(parquet), "--out", str(out)]
    cli.main(argv)
    first = out.read_bytes()
    cli.main(argv)
    assert out.read_bytes() == first


def test_the_test_split_is_refused(selected, parquet, tmp_path, capsys):
    out = tmp_path / "b.json"
    argv = ["--split", "test", "--dir", str(selected), "--parquet", str(parquet)]
    code = cli.main(["baseline", *argv, "--out", str(out)])
    assert code != 0
    assert "test split" in capsys.readouterr().out
    assert not out.exists()


def test_default_output_is_benchmark_baseline_dev_json():
    assert baseline.DEFAULT_OUT.name == "baseline_dev.json"
    assert baseline.DEFAULT_OUT.parent.name == "benchmark"
