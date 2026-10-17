"""Retrieval layer: BM25, multilingual embeddings, hybrid RRF; index configurations and chunking.

Public API::

    env = RetrievalEnv()                                   # corpus + cached e5-small embedder
    r = env.retriever("hybrid", "all_lang", "article")     # r.name == "hybrid/all_lang/article"
    res = r.retrieve(question, "de", k=60)                 # RetrievalResult (chunk candidates)
    ids = retrieve_articles(r, question, "de", 20)         # unique article ids, best first
"""

from fedlex_bench.retrieval.chunking import CHUNKINGS, Chunk, build_chunks, chunk_article
from fedlex_bench.retrieval.embedding import (
    CachedEmbedder,
    Embedder,
    OnnxE5Embedder,
    content_hash,
)
from fedlex_bench.retrieval.fusion import RRF_K, merge_article_ids, rrf
from fedlex_bench.retrieval.index import (
    INDEX_CONFIGS,
    METHODS,
    IndexRetriever,
    RetrievalEnv,
    retrieve_articles,
)

__all__ = [
    "CHUNKINGS",
    "INDEX_CONFIGS",
    "METHODS",
    "RRF_K",
    "CachedEmbedder",
    "Chunk",
    "Embedder",
    "IndexRetriever",
    "OnnxE5Embedder",
    "RetrievalEnv",
    "build_chunks",
    "chunk_article",
    "content_hash",
    "merge_article_ids",
    "retrieve_articles",
    "rrf",
]
