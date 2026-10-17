"""Indexes and retrievers: BM25, embeddings, hybrid (RRF, k = 60).

``RetrievalEnv`` owns the corpus, the (cached) embedder and the built indexes, so BM25 indexes and
chunk embeddings are built once and shared by every retriever. ``env.retriever(method, config,
chunking)`` returns an object implementing :class:`fedlex_bench.types.Retriever`.

- ``method``: ``bm25`` | ``embedding`` | ``hybrid``.
- ``config`` (which index a query hits): ``same_lang`` (the index of the query language, undefined
  for English: raises ``ValueError``), ``single_lang_de|fr|it`` (one fixed language for every query
  language), ``all_lang`` (one index with all three languages).
- ``chunking``: ``article`` | ``paragraph`` | ``window`` (see :mod:`.chunking`).
- ``name`` = ``"<method>/<config>/<chunking>"``.

``retrieve(query, query_lang, k)`` returns the ``k`` best **chunks** (candidates); one article can
occur several times (languages, chunks). Use :func:`retrieve_articles` (or
:func:`fedlex_bench.retrieval.fusion.merge_article_ids`) to get unique article ids.

BM25 is the project's own (:class:`fedlex_bench.benchmark.bm25.Bm25Index`): only documents sharing
a term with the query are returned, so a cross-lingual query with no lexical overlap yields fewer
than ``k`` candidates. The hybrid fuses the top ``depth`` (default 200) of BM25 and of the
embedding ranking with RRF; a chunk found by only one of them still enters with its single term.
Scores: BM25 raw score, cosine, RRF sum. Ties break by index order (act, article, language).
"""

from __future__ import annotations

import time
from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from fedlex_bench.benchmark.bm25 import Bm25Index
from fedlex_bench.chunk import ARTICLES_PARQUET
from fedlex_bench.retrieval.chunking import CHUNKINGS, CORPUS_LANGS, Chunk, build_chunks
from fedlex_bench.retrieval.embedding import INDEX_DIR, CachedEmbedder, Embedder, OnnxE5Embedder
from fedlex_bench.retrieval.fusion import RRF_K, merge_article_ids, rrf
from fedlex_bench.types import Candidate, QueryLang, RetrievalResult, Retriever, Usage

METHODS: tuple[str, ...] = ("bm25", "embedding", "hybrid")
INDEX_CONFIGS: tuple[str, ...] = (
    "same_lang",
    "single_lang_de",
    "single_lang_fr",
    "single_lang_it",
    "all_lang",
)
QUERY_LANGS: tuple[str, ...] = ("en", "de", "fr", "it")
DEFAULT_DEPTH = 200

Scope = Literal["de", "fr", "it", "all"]


class ChunkIndex:
    """All chunks of one scope (one language or all) for one chunking, with lazy BM25 and
    embedding matrix. ``chunks`` is in deterministic order; positions are the tie-break."""

    def __init__(self, env: RetrievalEnv, chunking: str, scope: Scope) -> None:
        self.env, self.chunking, self.scope = env, chunking, scope
        langs = CORPUS_LANGS if scope == "all" else (scope,)
        self.chunks: list[Chunk] = [c for c in env.chunks(chunking) if c.lang in langs]
        if not self.chunks:
            raise ValueError(f"no retrieval-eligible chunks for scope {scope!r}")

    @cached_property
    def _keys(self) -> list[str]:
        return [f"{c.chunk_id}@{c.lang}" for c in self.chunks]

    @cached_property
    def bm25(self) -> Bm25Index:
        return Bm25Index(dict(zip(self._keys, (c.text for c in self.chunks), strict=True)))

    @cached_property
    def _pos(self) -> dict[str, int]:
        return {k: i for i, k in enumerate(self._keys)}

    @cached_property
    def matrix(self) -> np.ndarray:
        return self.env.embedder.embed_passages([c.text for c in self.chunks])

    def rank_bm25(self, query: str, depth: int) -> list[tuple[int, float]]:
        scored = [(self._pos[k], s) for k, s in self.bm25.scores(query).items()]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:depth]

    def rank_embedding(self, query: str, depth: int) -> list[tuple[int, float]]:
        q = self.env.embedder.embed_queries([query])[0]
        sims = self.matrix @ q
        order = np.argsort(-sims, kind="stable")[:depth]
        return [(int(i), float(sims[i])) for i in order]


class IndexRetriever:
    def __init__(
        self,
        env: RetrievalEnv,
        method: str,
        index_config: str,
        chunking: str,
        depth: int = DEFAULT_DEPTH,
        rrf_k: int = RRF_K,
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
        if index_config not in INDEX_CONFIGS:
            raise ValueError(f"unknown index config {index_config!r}; expected {INDEX_CONFIGS}")
        if chunking not in CHUNKINGS:
            raise ValueError(f"unknown chunking {chunking!r}; expected one of {CHUNKINGS}")
        self.env, self.method, self.index_config, self.chunking = (
            env,
            method,
            index_config,
            chunking,
        )
        self.depth, self.rrf_k = depth, rrf_k
        self.name = f"{method}/{index_config}/{chunking}"

    def scope_for(self, query_lang: str) -> Scope:
        if query_lang not in QUERY_LANGS:
            raise ValueError(f"query_lang must be one of {QUERY_LANGS}, got {query_lang!r}")
        if self.index_config == "same_lang":
            if query_lang == "en":
                raise ValueError(
                    "same_lang is undefined for English queries (there is no English index); "
                    "use a single_lang_* or all_lang configuration"
                )
            return query_lang  # type: ignore[return-value]
        if self.index_config == "all_lang":
            return "all"
        return self.index_config.removeprefix("single_lang_")  # type: ignore[return-value]

    def retrieve(self, query: str, query_lang: QueryLang, k: int) -> RetrievalResult:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        scope = self.scope_for(query_lang)
        t0 = time.perf_counter()
        index = self.env.index(self.chunking, scope)
        depth = max(self.depth, k)
        if self.method == "bm25":
            ranked = index.rank_bm25(query, depth)
        elif self.method == "embedding":
            ranked = index.rank_embedding(query, depth)
        else:
            lists = [index.rank_bm25(query, depth), index.rank_embedding(query, depth)]
            fused = rrf([[pos for pos, _ in lst] for lst in lists], k=self.rrf_k)
            ranked = [(int(pos), score) for pos, score in fused]
        cands = [
            Candidate(
                article_id=(c := index.chunks[pos]).article_id,
                lang=c.lang,  # type: ignore[arg-type]
                text=c.text,
                score=float(score),
                chunk_id=c.chunk_id,
            )
            for pos, score in ranked[:k]
        ]
        return RetrievalResult(cands, Usage(latency_s=time.perf_counter() - t0))


class RetrievalEnv:
    """Corpus + embedder + index cache. ``articles`` defaults to ``articles.parquet``;
    ``embedder`` defaults to multilingual-e5-small (ONNX, CPU). Passage vectors are cached on disk
    by content hash in ``cache_dir``."""

    def __init__(
        self,
        articles: pd.DataFrame | None = None,
        embedder: Embedder | None = None,
        cache_dir: Path | str = INDEX_DIR,
    ) -> None:
        self.articles = pd.read_parquet(ARTICLES_PARQUET) if articles is None else articles
        self.embedder = CachedEmbedder(
            embedder if embedder is not None else OnnxE5Embedder(), cache_dir
        )
        self._chunks: dict[str, list[Chunk]] = {}
        self._indexes: dict[tuple[str, str], ChunkIndex] = {}

    def chunks(self, chunking: str) -> list[Chunk]:
        if chunking not in self._chunks:
            self._chunks[chunking] = build_chunks(self.articles, chunking)
        return self._chunks[chunking]

    def index(self, chunking: str, scope: Scope) -> ChunkIndex:
        key = (chunking, scope)
        if key not in self._indexes:
            self._indexes[key] = ChunkIndex(self, chunking, scope)
        return self._indexes[key]

    def retriever(
        self,
        method: str,
        index_config: str,
        chunking: str = "article",
        *,
        depth: int = DEFAULT_DEPTH,
    ) -> Retriever:
        return IndexRetriever(self, method, index_config, chunking, depth)

    def warm(self, chunking: str = "article", methods: tuple[str, ...] = METHODS) -> None:
        """Build (and cache to disk) everything the given methods need for all index scopes."""
        for scope in (*CORPUS_LANGS, "all"):
            index = self.index(chunking, scope)  # type: ignore[arg-type]
            if "bm25" in methods or "hybrid" in methods:
                _ = index.bm25
            if "embedding" in methods or "hybrid" in methods:
                _ = index.matrix


def retrieve_articles(
    retriever: Retriever, query: str, query_lang: QueryLang, n: int, max_candidates: int = 2000
) -> list[str]:
    """Up to ``n`` unique article ids, best first (language copies and chunks merged, first
    occurrence wins). Widens the candidate list until ``n`` unique articles are found or the
    retriever has nothing left."""
    k = max(3 * n, 10)
    while True:
        cands = retriever.retrieve(query, query_lang, k).candidates
        ids = merge_article_ids(cands, n)
        if len(ids) >= n or len(cands) < k or k >= max_candidates:
            return ids
        k = min(k * 2, max_candidates)
