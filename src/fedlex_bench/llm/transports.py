"""Thin provider adapters: one raw call in, text plus token counts out.

Token convention (matches `types.Usage` and the price table):
  input_tokens         prompt tokens billed at the full input rate (cache hits excluded)
  cached_input_tokens  prompt tokens served from the provider's own prompt cache
  output_tokens        generated tokens, reasoning/thinking tokens included
Secrets are read from the environment (loaded from .env); they never appear in code or config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

from dotenv import load_dotenv

from fedlex_bench.llm.config import ModelSpec, repo_root
from fedlex_bench.types import LLMRequest


class ProviderError(RuntimeError):
    """Configuration or provider-side failure (never cached, never billed to the ledger)."""


@dataclass(frozen=True)
class RawResult:
    text: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int


class Transport(Protocol):
    def call(self, spec: ModelSpec, request: LLMRequest) -> RawResult: ...


def _api_key(env_var: str, load_env: bool) -> str:
    if load_env:
        load_dotenv(repo_root() / ".env")
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise ProviderError(f"{env_var} is not set; put it in .env (see .env.example)")
    return key


def _n(value: Any) -> int:
    return int(value) if value else 0


class AnthropicTransport:
    def __init__(
        self, *, sdk: Any = None, env_var: str = "ANTHROPIC_API_KEY", load_env: bool = True
    ):
        if sdk is None:
            import anthropic

            sdk = anthropic.Anthropic(api_key=_api_key(env_var, load_env))
        self._sdk = sdk

    def call(self, spec: ModelSpec, request: LLMRequest) -> RawResult:
        kwargs: dict[str, Any] = {
            "model": spec.model_id,
            "max_tokens": request.max_output_tokens,
            "system": request.system,
            "messages": [{"role": "user", "content": request.user}],
        }
        if spec.effort:
            kwargs["output_config"] = {"effort": spec.effort}
        if spec.send_temperature:
            kwargs["temperature"] = request.temperature
        resp = self._sdk.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        u = resp.usage
        return RawResult(
            text=text,
            # cache-creation tokens are billed above the base rate; we never set cache_control,
            # so they should be 0. Counting them as plain input keeps the estimate conservative.
            input_tokens=_n(u.input_tokens) + _n(getattr(u, "cache_creation_input_tokens", 0)),
            cached_input_tokens=_n(getattr(u, "cache_read_input_tokens", 0)),
            output_tokens=_n(u.output_tokens),
        )


class OpenAITransport:
    def __init__(self, *, sdk: Any = None, env_var: str = "OPENAI_API_KEY", load_env: bool = True):
        if sdk is None:
            import openai

            sdk = openai.OpenAI(api_key=_api_key(env_var, load_env))
        self._sdk = sdk

    def call(self, spec: ModelSpec, request: LLMRequest) -> RawResult:
        kwargs: dict[str, Any] = {
            "model": spec.model_id,
            "instructions": request.system,
            "input": request.user,
            "max_output_tokens": request.max_output_tokens,
        }
        if spec.effort:
            kwargs["reasoning"] = {"effort": spec.effort}
        if request.json_output:
            kwargs["text"] = {"format": {"type": "json_object"}}
        if spec.send_temperature:
            kwargs["temperature"] = request.temperature
        resp = self._sdk.responses.create(**kwargs)
        u = resp.usage
        details = getattr(u, "input_tokens_details", None)
        cached = _n(getattr(details, "cached_tokens", 0)) if details else 0
        return RawResult(
            text=resp.output_text or "",
            input_tokens=max(_n(u.input_tokens) - cached, 0),
            cached_input_tokens=cached,
            output_tokens=_n(u.output_tokens),  # includes reasoning tokens
        )


class GoogleTransport:
    def __init__(self, *, sdk: Any = None, env_var: str = "GEMINI_API_KEY", load_env: bool = True):
        if sdk is None:
            from google import genai

            sdk = genai.Client(api_key=_api_key(env_var, load_env))
        self._sdk = sdk

    def call(self, spec: ModelSpec, request: LLMRequest) -> RawResult:
        from google.genai import types

        cfg: dict[str, Any] = {
            "system_instruction": request.system,
            "max_output_tokens": request.max_output_tokens,
        }
        if spec.effort:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_level=spec.effort.upper())
        if request.json_output:
            cfg["response_mime_type"] = "application/json"
        if spec.send_temperature:
            cfg["temperature"] = request.temperature
        resp = self._sdk.models.generate_content(
            model=spec.model_id,
            contents=request.user,
            config=types.GenerateContentConfig(**cfg),
        )
        u = resp.usage_metadata
        cached = _n(getattr(u, "cached_content_token_count", 0))
        out = getattr(u, "candidates_token_count", None)
        if out is None:
            out = getattr(u, "response_token_count", 0)
        return RawResult(
            text=resp.text or "",
            input_tokens=max(_n(getattr(u, "prompt_token_count", 0)) - cached, 0),
            cached_input_tokens=cached,
            # thoughts are billed as output and are not part of the candidates count
            output_tokens=_n(out) + _n(getattr(u, "thoughts_token_count", 0)),
        )


TRANSPORTS = {
    "anthropic": AnthropicTransport,
    "openai": OpenAITransport,
    "google": GoogleTransport,
}
