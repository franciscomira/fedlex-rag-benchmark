"""Shared interfaces for Phase 2. Owned by the lead; teammates build against these.

Changing a name or field here needs the lead's approval (message the lead).
All article ids are "<SR>:<article>" and identical across languages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

Lang = Literal["de", "fr", "it"]
QueryLang = Literal["en", "de", "fr", "it"]
Status = Literal["answered", "refused"]


@dataclass(frozen=True)
class Usage:
    """Token, cost and latency totals of one or more model calls."""

    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    calls: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.cached_input_tokens + other.cached_input_tokens,
            self.output_tokens + other.output_tokens,
            self.cost_usd + other.cost_usd,
            self.latency_s + other.latency_s,
            self.calls + other.calls,
        )


@dataclass(frozen=True)
class Candidate:
    """One retrieved passage. `article_id` is the scoring key; `lang` is the text language."""

    article_id: str
    lang: Lang
    text: str
    score: float = 0.0
    chunk_id: str = ""  # article id plus chunk suffix for paragraph/window chunking


@dataclass(frozen=True)
class RetrievalResult:
    candidates: list[Candidate]  # ranked best first, may hold several languages per article
    usage: Usage = field(default_factory=Usage)  # embedding/rerank cost and latency


@dataclass(frozen=True)
class LLMRequest:
    system: str
    user: str
    max_output_tokens: int = 1024
    temperature: float = 0.0
    json_output: bool = False


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model_id: str  # dated, pinned id
    usage: Usage
    cached: bool  # served from the disk cache; usage then replays the stored original


@dataclass(frozen=True)
class AnswerResult:
    """Parsed, code-validated answer object (docs/eval_spec.md section 3)."""

    status: Status
    answer: str
    citations: list[str]  # valid ids only (present in the candidate list)
    pointers: list[str]
    invalid_citations: list[str]  # ids the model produced that were not in the candidates
    parse_failed: bool = False
    usage: Usage = field(default_factory=Usage)


@dataclass(frozen=True)
class SystemOutput:
    """What the harness gets back from any system under test for one item."""

    ranked_article_ids: list[str]  # unique ids, best first (language copies already merged)
    answer: AnswerResult | None  # None for retrieval-only systems
    usage: Usage = field(default_factory=Usage)  # whole pipeline, judge excluded
    trace: list[dict[str, Any]] = field(default_factory=list)  # agent steps; [] otherwise


@runtime_checkable
class Retriever(Protocol):
    name: str  # stable config name, e.g. "hybrid/all_lang/article"

    def retrieve(self, query: str, query_lang: QueryLang, k: int) -> RetrievalResult: ...


@runtime_checkable
class Reranker(Protocol):
    name: str

    def rerank(
        self, query: str, query_lang: QueryLang, candidates: list[Candidate], k: int
    ) -> RetrievalResult: ...


@runtime_checkable
class LLMClient(Protocol):
    model_id: str

    def complete(self, request: LLMRequest) -> LLMResponse: ...


@runtime_checkable
class Answerer(Protocol):
    name: str

    def answer(
        self, query: str, query_lang: QueryLang, candidates: list[Candidate]
    ) -> AnswerResult: ...


@runtime_checkable
class System(Protocol):
    """A complete system under test: retriever (+ reranker) (+ answerer) or the agent."""

    name: str

    def run(self, query: str, query_lang: QueryLang) -> SystemOutput: ...

    def config(self) -> dict[str, Any]:  # goes into config.json of the run
        ...
