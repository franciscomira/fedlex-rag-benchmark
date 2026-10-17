import pytest

from fedlex_bench.llm.budget import Budget, Ledger
from fedlex_bench.llm.cache import DiskCache
from fedlex_bench.llm.factory import build_client
from fedlex_bench.llm.fake import FakeTransport
from fedlex_bench.llm.transports import ProviderError, RawResult
from fedlex_bench.types import LLMRequest


def test_build_client_by_role_with_fake_transport(tmp_path):
    t = FakeTransport([RawResult("ok", 10, 0, 5)])
    c = build_client(
        "google/mid",
        transport=t,
        cache=DiskCache(tmp_path / "c"),
        budget=Budget(25.0, 5.0, Ledger(tmp_path / "l.jsonl")),
    )
    assert c.model_id == "gemini-3.6-flash"
    r = c.complete(LLMRequest("s", "u"))
    assert r.text == "ok" and r.usage.cost_usd > 0


@pytest.mark.parametrize("env", ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"])
def test_live_client_without_key_fails_cleanly(monkeypatch, tmp_path, env):
    monkeypatch.delenv(env, raising=False)
    # make sure a developer's real .env cannot leak into the test
    monkeypatch.setattr("fedlex_bench.llm.transports.load_dotenv", lambda *a, **k: None)
    ref = {"ANTHROPIC_API_KEY": "anthropic/small", "OPENAI_API_KEY": "openai/small"}.get(
        env, "google/small"
    )
    with pytest.raises(ProviderError, match=env):
        build_client(ref, cache=DiskCache(tmp_path))
