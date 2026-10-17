"""LLM access: provider clients, disk cache, budget, LLM reranker.

Typical use (live calls need keys in .env; tests use `llm.fake`):

    from fedlex_bench.llm import build_client, LLMReranker
    client = build_client("anthropic/small")     # or a pinned id, e.g. "gpt-6-luna"
"""

from fedlex_bench.llm.budget import (
    Budget,
    BudgetExceeded,
    Ledger,
    PlannedCalls,
    Projection,
    project_cost,
)
from fedlex_bench.llm.cache import DiskCache
from fedlex_bench.llm.client import ManagedClient
from fedlex_bench.llm.config import ModelSpec, load_models, resolve_model
from fedlex_bench.llm.factory import build_client
from fedlex_bench.llm.prices import PriceTable
from fedlex_bench.llm.rerank import LLMReranker

__all__ = [
    "Budget",
    "BudgetExceeded",
    "DiskCache",
    "LLMReranker",
    "Ledger",
    "ManagedClient",
    "ModelSpec",
    "PlannedCalls",
    "PriceTable",
    "Projection",
    "build_client",
    "load_models",
    "project_cost",
    "resolve_model",
]
