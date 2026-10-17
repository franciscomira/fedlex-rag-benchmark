"""Exact article lookup backed by the corpus parquet (implements ArticleStore)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fedlex_bench.types import Candidate, Lang


class ParquetArticleStore:
    def __init__(self, parquet: Path) -> None:
        df = pd.read_parquet(parquet, columns=["article_id", "lang", "text"])
        self._text = {(a, lang): t for a, lang, t in df.itertuples(index=False)}

    def get(self, article_id: str, lang: Lang) -> Candidate | None:
        text = self._text.get((article_id, lang))
        if text is None:
            return None
        return Candidate(article_id, lang, text, 0.0, article_id)
