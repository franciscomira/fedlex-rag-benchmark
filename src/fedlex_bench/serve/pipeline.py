"""The serving pipeline: tuned int8 embedder, Haiku reranker and a code-validated cited answer.

This is the harness' own `PipelineSystem` (the frozen configuration P2), not a re-implementation:
the demo and the MCP server call exactly what `eval run` measured. Article ids, the article texts
and the citations come from the corpus and from code; the model only ranks and drafts.

    pipe = ServingPipeline()                    # model + corpus from the repo, or from the Hub
    hits = pipe.search("Wie lange ist die Kündigungsfrist?", k=10)      # no LLM, 0 USD
    res = pipe.ask("What is the notice period?", "en")                  # needs ANTHROPIC_API_KEY
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from fedlex_bench.chunk import ARTICLES_PARQUET
from fedlex_bench.eval.metrics import dedupe_ids
from fedlex_bench.eval.systems import PipelineSystem
from fedlex_bench.ingest import PROJECT_ROOT
from fedlex_bench.retrieval.embedding import INDEX_DIR, Embedder, OnnxE5Embedder
from fedlex_bench.retrieval.index import RetrievalEnv
from fedlex_bench.types import LLMClient, QueryLang

HUB_REPO = "franciscomira/fedlex-e5-small"
HUB_REVISION = "786ca378bae0b6e6b8e32be6eaae2df517693118"  # pinned: onnx/model_int8.onnx
INT8_FILE = "model_int8.onnx"
LOCAL_MODEL_DIR = PROJECT_ROOT / "models" / "r1_inbatch" / "onnx"
LANGS = ("de", "fr", "it")
QUERY_LANGS = ("en", "de", "fr", "it")
MAX_QUESTION_CHARS = 1000

# The frozen serving configuration (eval/frozen_configs.json, entry P2).
RETRIEVE_K = 150
RERANK_K = 60
ANSWER_ARTICLES = 5
LLM_MODEL = "claude-haiku-5-5"


def resolve_model_dir() -> Path:
    """Folder with `model_int8.onnx` and `tokenizer.json`: $FEDLEX_MODEL_DIR, the local export, or
    the pinned revision on the Hugging Face Hub (downloaded once into the Hub cache)."""
    env = os.environ.get("FEDLEX_MODEL_DIR")
    for cand in (Path(env) if env else None, LOCAL_MODEL_DIR):
        if cand and (cand / INT8_FILE).exists() and (cand / "tokenizer.json").exists():
            return cand
    from huggingface_hub import hf_hub_download

    model = hf_hub_download(HUB_REPO, f"onnx/{INT8_FILE}", revision=HUB_REVISION)
    hf_hub_download(HUB_REPO, "onnx/tokenizer.json", revision=HUB_REVISION)
    return Path(model).parent


@dataclass(frozen=True)
class ArticleHit:
    """One article with its text in every language the corpus has, straight from the corpus."""

    rank: int
    article_id: str
    sr: str
    score: float | None
    titles: dict[str, str]
    texts: dict[str, str]
    source_urls: dict[str, str]
    is_repealed: bool
    consolidation_date: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "article_id": self.article_id,
            "sr": self.sr,
            "score": self.score,
            "titles": self.titles,
            "texts": self.texts,
            "source_urls": self.source_urls,
            "is_repealed": self.is_repealed,
            "consolidation_date": self.consolidation_date,
        }


@dataclass(frozen=True)
class AskResult:
    status: str  # "answered" | "refused" (refused = "not in the sources")
    answer: str
    citations: list[ArticleHit]  # only ids that were among the retrieved articles
    pointers: list[ArticleHit]
    n_invalid_ids: int  # ids the model produced that were not retrieved (dropped by code)
    hits: list[ArticleHit]  # ranked articles shown next to the answer
    parse_failed: bool = False
    cost_usd: float = 0.0
    latency_s: float = 0.0
    query_lang: str = "en"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "answer": self.answer,
            "citations": [h.to_dict() for h in self.citations],
            "pointers": [h.to_dict() for h in self.pointers],
            "n_invalid_ids": self.n_invalid_ids,
            "hits": [h.to_dict() for h in self.hits],
            "parse_failed": self.parse_failed,
            "cost_usd": self.cost_usd,
            "latency_s": self.latency_s,
            "query_lang": self.query_lang,
        }


class Corpus:
    """Exact article lookup (all languages) over the corpus parquet."""

    def __init__(self, articles: pd.DataFrame) -> None:
        self._rows: dict[str, dict[str, dict[str, Any]]] = {}
        for rec in articles.sort_values(["article_id", "lang"]).to_dict("records"):
            self._rows.setdefault(rec["article_id"], {})[rec["lang"]] = rec

    def __contains__(self, article_id: str) -> bool:
        return article_id in self._rows

    def __len__(self) -> int:
        return len(self._rows)

    def hit(self, article_id: str, rank: int, score: float | None = None) -> ArticleHit:
        by_lang = self._rows[article_id]
        first = next(iter(by_lang.values()))
        return ArticleHit(
            rank=rank,
            article_id=article_id,
            sr=str(first["sr"]),
            score=score,
            titles={lang: str(r.get("title") or "") for lang, r in by_lang.items()},
            texts={lang: str(r.get("text") or "") for lang, r in by_lang.items()},
            source_urls={lang: str(r.get("source_url") or "") for lang, r in by_lang.items()},
            is_repealed=bool(first.get("is_repealed")),
            consolidation_date=str(first.get("consolidation_date") or ""),
        )


class ServingPipeline:
    """Lazy: the model, the index and the LLM client are built on first use."""

    def __init__(
        self,
        *,
        model_dir: Path | str | None = None,
        articles: pd.DataFrame | None = None,
        index_dir: Path | str = INDEX_DIR,
        llm: LLMClient | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self._embedder = embedder  # tests inject a fake; serving builds the int8 model
        self._model_dir = Path(model_dir) if model_dir else None
        self._articles = articles
        self._index_dir = index_dir
        self._llm = llm
        self._env: RetrievalEnv | None = None
        self._corpus: Corpus | None = None
        self._system: PipelineSystem | None = None

    @property
    def env(self) -> RetrievalEnv:
        if self._env is None:
            embedder = self._embedder
            if embedder is None:
                model_dir = self._model_dir or resolve_model_dir()
                embedder = OnnxE5Embedder(local_dir=model_dir, onnx_file=INT8_FILE)
            articles = (
                pd.read_parquet(ARTICLES_PARQUET) if self._articles is None else self._articles
            )
            self._env = RetrievalEnv(
                articles=articles, embedder=embedder, cache_dir=self._index_dir
            )
            self._articles = articles
        return self._env

    @property
    def corpus(self) -> Corpus:
        if self._corpus is None:
            _ = self.env
            self._corpus = Corpus(self._articles)
        return self._corpus

    @property
    def retriever(self):
        return self.env.retriever("embedding", "all_lang", "article")

    def _llm_client(self) -> LLMClient:
        if self._llm is None:
            from fedlex_bench.llm import build_client

            self._llm = build_client(LLM_MODEL)
        return self._llm

    @property
    def system(self) -> PipelineSystem:
        """The frozen P2 system: tuned int8 retrieval, Haiku rerank, cited answer."""
        if self._system is None:
            from fedlex_bench.answer import LLMAnswerer
            from fedlex_bench.llm import LLMReranker

            client = self._llm_client()
            self._system = PipelineSystem(
                self.retriever,
                reranker=LLMReranker(client, max_candidates=RERANK_K),
                answerer=LLMAnswerer(client),
                retrieve_k=RETRIEVE_K,
                rerank_k=RERANK_K,
                answer_articles=ANSWER_ARTICLES,
            )
        return self._system

    def search(self, query: str, k: int = 10) -> list[ArticleHit]:
        """Tuned embedder only: no LLM, no cost. Articles ranked by their best passage."""
        query = _check_question(query)
        k = max(1, min(int(k), 50))
        cands = self.retriever.retrieve(query, "en", max(60, 3 * k)).candidates
        best: dict[str, float] = {}
        for c in cands:  # candidates arrive best first
            best.setdefault(c.article_id, c.score)
        return [
            self.corpus.hit(article_id, rank, score)
            for rank, (article_id, score) in enumerate(list(best.items())[:k], 1)
        ]

    def get_article(self, article_id: str) -> ArticleHit | None:
        article_id = article_id.strip()
        return self.corpus.hit(article_id, 0) if article_id in self.corpus else None

    def ask(self, query: str, query_lang: QueryLang = "en", n_hits: int = 10) -> AskResult:
        """The full pipeline. Needs an LLM key. Citations are validated by code against the
        retrieved articles; `status == "refused"` is the "not in the sources" path."""
        query = _check_question(query)
        if query_lang not in QUERY_LANGS:
            raise ValueError(f"query_lang must be one of {QUERY_LANGS}, got {query_lang!r}")
        out = self.system.run(query, query_lang)
        ans = out.answer
        ranked = dedupe_ids(out.ranked_article_ids)[:n_hits]
        hits = [self.corpus.hit(a, r) for r, a in enumerate(ranked, 1)]
        by_id = {h.article_id: h for h in hits}

        def lookup(ids: list[str]) -> list[ArticleHit]:
            return [
                by_id[i] if i in by_id else self.corpus.hit(i, 0) for i in ids if i in self.corpus
            ]

        return AskResult(
            status=ans.status if ans else "refused",
            answer=ans.answer if ans else "",
            citations=lookup(ans.citations) if ans else [],
            pointers=lookup(ans.pointers) if ans else [],
            n_invalid_ids=len(ans.invalid_citations) if ans else 0,
            hits=hits,
            parse_failed=bool(ans and ans.parse_failed),
            cost_usd=out.usage.cost_usd,
            latency_s=out.usage.latency_s,
            query_lang=query_lang,
        )


def _check_question(query: str) -> str:
    query = (query or "").strip()
    if not query:
        raise ValueError("empty question")
    if len(query) > MAX_QUESTION_CHARS:
        raise ValueError(f"question longer than {MAX_QUESTION_CHARS} characters")
    return query
