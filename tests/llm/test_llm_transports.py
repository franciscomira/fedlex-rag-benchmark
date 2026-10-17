from types import SimpleNamespace as NS

import pytest

from fedlex_bench.llm.config import ModelSpec
from fedlex_bench.llm.transports import (
    AnthropicTransport,
    GoogleTransport,
    OpenAITransport,
    ProviderError,
)
from fedlex_bench.types import LLMRequest


def spec(provider, model_id="m", effort="low", send_temperature=False):
    return ModelSpec(model_id, provider, "small", effort, send_temperature, "url", "2026-10-17")


REQ = LLMRequest(system="SYS", user="USER", max_output_tokens=300, temperature=0.0)


class Recorder:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def test_anthropic_maps_request_and_usage():
    resp = NS(
        content=[
            NS(type="thinking", thinking=""),
            NS(type="text", text='{"a":'),
            NS(type="text", text="1}"),
        ],
        usage=NS(
            input_tokens=100,
            output_tokens=40,
            cache_read_input_tokens=30,
            cache_creation_input_tokens=0,
        ),
        stop_reason="end_turn",
    )
    rec = Recorder(resp)
    t = AnthropicTransport(sdk=NS(messages=NS(create=rec)))
    raw = t.call(spec("anthropic", "claude-x"), REQ)
    assert raw.text == '{"a":1}'
    assert (raw.input_tokens, raw.cached_input_tokens, raw.output_tokens) == (100, 30, 40)
    kw = rec.kwargs
    assert kw["model"] == "claude-x" and kw["max_tokens"] == 300 and kw["system"] == "SYS"
    assert kw["messages"] == [{"role": "user", "content": "USER"}]
    assert kw["output_config"] == {"effort": "low"}
    assert "temperature" not in kw


def test_anthropic_cache_creation_tokens_count_as_input():
    resp = NS(
        content=[NS(type="text", text="x")],
        usage=NS(
            input_tokens=10,
            output_tokens=1,
            cache_read_input_tokens=None,
            cache_creation_input_tokens=5,
        ),
    )
    raw = AnthropicTransport(sdk=NS(messages=NS(create=Recorder(resp)))).call(
        spec("anthropic"), REQ
    )
    assert raw.input_tokens == 15 and raw.cached_input_tokens == 0


def test_anthropic_sends_temperature_only_when_enabled():
    resp = NS(content=[NS(type="text", text="x")], usage=NS(input_tokens=1, output_tokens=1))
    rec = Recorder(resp)
    AnthropicTransport(sdk=NS(messages=NS(create=rec))).call(
        spec("anthropic", send_temperature=True), REQ
    )
    assert rec.kwargs["temperature"] == 0.0


def test_openai_maps_request_and_usage():
    resp = NS(
        output_text="hello",
        usage=NS(input_tokens=200, input_tokens_details=NS(cached_tokens=50), output_tokens=70),
    )
    rec = Recorder(resp)
    t = OpenAITransport(sdk=NS(responses=NS(create=rec)))
    raw = t.call(spec("openai", "gpt-x"), LLMRequest("SYS", "USER", 300, 0.0, json_output=True))
    assert raw.text == "hello"
    assert (raw.input_tokens, raw.cached_input_tokens, raw.output_tokens) == (150, 50, 70)
    kw = rec.kwargs
    assert kw["model"] == "gpt-x" and kw["instructions"] == "SYS"
    assert kw["input"].startswith("USER") and "json" in kw["input"].lower()
    assert kw["max_output_tokens"] == 300
    assert kw["reasoning"] == {"effort": "low"}
    assert kw["text"] == {"format": {"type": "json_object"}}
    assert "temperature" not in kw


def test_openai_plain_text_has_no_format():
    resp = NS(output_text="x", usage=NS(input_tokens=1, input_tokens_details=None, output_tokens=1))
    rec = Recorder(resp)
    OpenAITransport(sdk=NS(responses=NS(create=rec))).call(spec("openai"), REQ)
    assert "text" not in rec.kwargs


def test_google_maps_request_and_usage():
    resp = NS(
        text="hey",
        usage_metadata=NS(
            prompt_token_count=120,
            cached_content_token_count=20,
            candidates_token_count=30,
            thoughts_token_count=25,
        ),
    )
    rec = Recorder(resp)
    t = GoogleTransport(sdk=NS(models=NS(generate_content=rec)))
    raw = t.call(spec("google", "gemini-x"), LLMRequest("SYS", "USER", 300, 0.0, json_output=True))
    assert raw.text == "hey"
    # thinking tokens are billed as output
    assert (raw.input_tokens, raw.cached_input_tokens, raw.output_tokens) == (100, 20, 55)
    kw = rec.kwargs
    assert kw["model"] == "gemini-x" and kw["contents"] == "USER"
    cfg = kw["config"]
    assert cfg.system_instruction == "SYS"
    assert cfg.max_output_tokens == 300
    assert cfg.response_mime_type == "application/json"
    assert str(cfg.thinking_config.thinking_level.value) == "LOW"
    assert cfg.temperature is None


def test_google_missing_text_is_empty_string():
    resp = NS(text=None, usage_metadata=NS(prompt_token_count=1, candidates_token_count=None))
    raw = GoogleTransport(sdk=NS(models=NS(generate_content=Recorder(resp)))).call(
        spec("google"), REQ
    )
    assert raw.text == "" and raw.output_tokens == 0


def test_missing_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY"):
        AnthropicTransport(env_var="ANTHROPIC_API_KEY", load_env=False)


def test_openai_json_output_puts_the_word_json_in_the_input():
    from types import SimpleNamespace as NS

    from fedlex_bench.llm.config import resolve_model
    from fedlex_bench.llm.transports import OpenAITransport
    from fedlex_bench.types import LLMRequest

    seen = {}

    class Sdk:
        class responses:  # noqa: N801
            @staticmethod
            def create(**kw):
                seen.update(kw)
                u = NS(input_tokens=1, output_tokens=1, input_tokens_details=None)
                return NS(output_text="{}", usage=u)

    OpenAITransport(sdk=Sdk).call(
        resolve_model("gpt-6-luna"), LLMRequest(system="s", user="question", json_output=True)
    )
    assert "json" in seen["input"].lower()
