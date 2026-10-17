import pandas as pd

from fedlex_bench.agent.store import ParquetArticleStore


def test_store_returns_exact_candidate_or_none(tmp_path):
    p = tmp_path / "a.parquet"
    pd.DataFrame([{"article_id": "220:1", "lang": "de", "text": "Eins"}]).to_parquet(p)
    s = ParquetArticleStore(p)
    c = s.get("220:1", "de")
    assert c.text == "Eins" and c.article_id == "220:1" and c.lang == "de"
    assert s.get("220:1", "fr") is None and s.get("9:9", "de") is None
