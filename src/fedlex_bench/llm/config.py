"""Model registry and project config (config.toml) loading."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

MODELS_PATH = Path(__file__).with_name("models.toml")


def repo_root() -> Path:
    """Repository root (holds config.toml, eval/, .env)."""
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ModelSpec:
    model_id: str  # dated/pinned id sent to the provider
    provider: str  # anthropic | openai | google
    tier: str  # small | mid
    effort: str | None  # reasoning/thinking level sent with every call
    send_temperature: bool  # False where the provider rejects or ignores it
    source_url: str
    retrieved: str


@dataclass(frozen=True)
class BudgetConfig:
    monthly_usd: float
    confirm_usd: float


def load_models(path: Path | None = None) -> dict[str, ModelSpec]:
    data = tomllib.loads((path or MODELS_PATH).read_text(encoding="utf-8"))
    out: dict[str, ModelSpec] = {}
    for model_id, m in sorted(data["models"].items()):
        out[model_id] = ModelSpec(
            model_id=model_id,
            provider=m["provider"],
            tier=m["tier"],
            effort=m.get("effort"),
            send_temperature=bool(m.get("send_temperature", False)),
            source_url=m["source_url"],
            retrieved=m["retrieved"],
        )
    return out


def resolve_model(ref: str, models: dict[str, ModelSpec] | None = None) -> ModelSpec:
    """`ref` is a pinned model id ("claude-haiku-5-5") or a role ("anthropic/small")."""
    models = models if models is not None else load_models()
    if ref in models:
        return models[ref]
    if "/" in ref:
        provider, tier = ref.split("/", 1)
        hits = [m for m in models.values() if m.provider == provider and m.tier == tier]
        if len(hits) == 1:
            return hits[0]
    raise KeyError(f"unknown model reference {ref!r}; known: {sorted(models)}")


def _load_config(path: Path | None) -> dict:
    return tomllib.loads((path or repo_root() / "config.toml").read_text(encoding="utf-8"))


def load_budget_config(path: Path | None = None) -> BudgetConfig:
    b = _load_config(path)["budget"]
    return BudgetConfig(float(b["monthly_usd"]), float(b["confirm_usd"]))


def load_provider_env_vars(path: Path | None = None) -> dict[str, str]:
    """provider -> name of the environment variable holding its key (from config.toml)."""
    return dict(_load_config(path)["providers"])
