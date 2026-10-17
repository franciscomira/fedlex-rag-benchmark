"""Build a ready-to-use client from a model reference ("claude-haiku-5-5" or "openai/small")."""

from __future__ import annotations

from fedlex_bench.llm.budget import Budget
from fedlex_bench.llm.cache import DiskCache
from fedlex_bench.llm.client import ManagedClient
from fedlex_bench.llm.config import load_models, load_provider_env_vars, resolve_model
from fedlex_bench.llm.prices import PriceTable
from fedlex_bench.llm.transports import TRANSPORTS, Transport


def build_client(
    ref: str,
    *,
    transport: Transport | None = None,
    cache: DiskCache | None = None,
    budget: Budget | None = None,
    prices: PriceTable | None = None,
) -> ManagedClient:
    """Live clients need the provider key in .env; pass `transport` to stay offline."""
    spec = resolve_model(ref, load_models())
    if transport is None:
        env_var = load_provider_env_vars()[spec.provider]
        transport = TRANSPORTS[spec.provider](env_var=env_var)
    return ManagedClient(
        spec,
        transport,
        cache or DiskCache(),
        budget or Budget.from_config(),
        prices or PriceTable.load(),
    )
