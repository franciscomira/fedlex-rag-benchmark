"""MCP server: tool list and schemas, happy and error paths, notices everywhere (offline, fakes)."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "retrieval"))
from mcp import StdioServerParameters  # noqa: E402
from mcp.client import Client  # noqa: E402
from retrieval_fixtures import FakeEmbedder, make_articles  # noqa: E402

from fedlex_bench.llm.budget import BudgetExceeded  # noqa: E402
from fedlex_bench.llm.fake import FakeClient  # noqa: E402
from fedlex_bench.serve import (
    mcp_server,  # noqa: E402
    notices,  # noqa: E402
)
from fedlex_bench.serve.pipeline import MAX_QUESTION_CHARS, ServingPipeline  # noqa: E402
from fedlex_bench.types import LLMRequest  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def llm_reply(request: LLMRequest) -> str:
    if "ranking" in request.user or "ranking" in request.system:
        return json.dumps({"ranking": [1, 2, 3]})
    return json.dumps(
        {
            "status": "answered",
            "answer": "Notice must be in writing.",
            "citations": ["220:1", "999:9"],  # the second id is invented
            "pointers": ["220:2"],
        }
    )


@pytest.fixture
def pipe(tmp_path):
    return ServingPipeline(
        articles=make_articles(),
        embedder=FakeEmbedder(),
        index_dir=tmp_path,
        llm=FakeClient(llm_reply),
    )


def call(server, name, **args):
    """(is_error, payload) of one tool call through the real MCP protocol (in-process)."""

    async def go():
        async with Client(server) as client:
            res = await client.call_tool(name, args)
        text = "".join(c.text for c in res.content if getattr(c, "type", "") == "text")
        return res.is_error, text, res.structured_content

    is_error, text, structured = asyncio.run(go())
    return is_error, (structured if structured is not None and not is_error else text)


def list_tools(server):
    async def go():
        async with Client(server) as client:
            return (await client.list_tools()).tools

    return asyncio.run(go())


def has_notices(result: dict) -> bool:
    n = result["notices"]
    blob = json.dumps(n, ensure_ascii=False)
    return (
        n["not_official_en"] == notices.NOT_OFFICIAL_EN
        and n["not_official_de"] == notices.NOT_OFFICIAL_DE
        and n["not_legal_advice"] == notices.NOT_LEGAL_ADVICE
        and "verbatim Fedlex" in n["source_note"]
        and "generated" in n["source_note"]
        and notices.NOT_LEGAL_ADVICE in blob
    )


@pytest.fixture
def server(pipe):
    return mcp_server.create_server(pipe, api_key_available=lambda: True)


# --- tool list and schemas ---------------------------------------------------------------------


def test_tool_list_and_schemas(server):
    tools = {t.name: t for t in list_tools(server)}
    assert set(tools) == {"list_acts", "search_articles", "get_article", "answer_with_citations"}
    s = tools["search_articles"].input_schema
    assert s["required"] == ["query"]
    assert s["properties"]["k"]["minimum"] == 1 and s["properties"]["k"]["maximum"] == 50
    assert s["properties"]["k"]["default"] == 10
    assert tools["get_article"].input_schema["required"] == ["article_id"]
    a = tools["answer_with_citations"].input_schema
    assert a["required"] == ["question"]
    assert a["properties"]["language"]["enum"] == ["en", "de", "fr", "it"]
    assert a["properties"]["language"]["default"] == "en"
    assert all(t.description for t in tools.values())


def test_question_limit_matches_pipeline():
    assert mcp_server.MAX_QUESTION_CHARS == MAX_QUESTION_CHARS


# --- happy paths ---------------------------------------------------------------------------------


def test_list_acts_reads_the_corpus(server):
    err, res = call(server, "list_acts")
    assert not err and has_notices(res)
    assert [a["sr"] for a in res["acts"]] == ["210", "220"]
    act = {a["sr"]: a for a in res["acts"]}["220"]
    assert act["name"] == "Code of Obligations"
    assert act["n_articles"] >= 2 and res["n_acts"] == 2


def test_search_articles_returns_ranked_summaries_not_full_text(server, pipe):
    err, res = call(server, "search_articles", query="Der Mieter zahlt den Mietzins", k=2)
    assert not err and has_notices(res)
    assert res["cost_usd"] == 0.0 and res["n_results"] == 2
    first = res["results"][0]
    assert [r["rank"] for r in res["results"]] == [1, 2]
    assert first["article_id"] == pipe.search("Der Mieter zahlt den Mietzins", k=1)[0].article_id
    assert first["sr"] == first["article_id"].split(":")[0]
    assert set(first["titles"]) == {"de", "fr", "it"} and isinstance(first["score"], float)
    assert set(first["snippets"]) == {"de", "fr", "it"}
    assert "texts" not in first and first["source_urls"] is not None


def test_search_snippets_are_short():
    long = "wort " * 100
    snip = mcp_server._snippet(long)
    assert len(snip) <= mcp_server.SNIPPET_CHARS + 4 and snip.endswith(" ...")
    assert mcp_server._snippet("kurz  und\nbuendig") == "kurz und buendig"


def test_get_article_full_text_in_three_languages(server):
    err, res = call(server, "get_article", article_id=" 220:1 ")
    assert not err and has_notices(res)
    assert res["article_id"] == "220:1" and res["sr"] == "220"
    assert (
        res["texts"]["de"] == "1 Der Mieter zahlt den Mietzins.\n2 Die Kuendigung ist schriftlich."
    )
    assert set(res["texts"]) == {"de", "fr", "it"} and set(res["titles"]) == {"de", "fr", "it"}
    assert res["is_repealed"] is False and "consolidation_date" in res and "source_urls" in res


def test_answer_with_citations_validates_ids_and_reports_cost(server):
    err, res = call(
        server, "answer_with_citations", question="Wie ist die Kuendigung?", language="de"
    )
    assert not err and has_notices(res)
    assert res["status"] == "answered" and res["answer"] == "Notice must be in writing."
    assert [c["article_id"] for c in res["citations"]] == ["220:1"]  # 999:9 dropped by code
    assert res["n_invalid_ids"] == 1
    assert [p["article_id"] for p in res["pointers"]] == ["220:2"]
    assert res["citations"][0]["titles"]["de"] == "Mietzins"
    assert "texts" not in res["citations"][0]
    assert isinstance(res["cost_usd"], float) and res["language"] == "de"
    assert res["ranked_articles"] and res["ranked_articles"][0]["rank"] == 1


def test_refusal_is_returned_as_a_status_not_an_error(tmp_path):
    refusal = json.dumps(
        {"status": "refused", "answer": "Not in the sources.", "citations": [], "pointers": []}
    )
    llm = FakeClient(
        lambda r: json.dumps({"ranking": [1]}) if "ranking" in r.user + r.system else refusal
    )
    pipe = ServingPipeline(
        articles=make_articles(), embedder=FakeEmbedder(), index_dir=tmp_path, llm=llm
    )
    server = mcp_server.create_server(pipe, api_key_available=lambda: True)
    err, res = call(server, "answer_with_citations", question="What is the moon made of?")
    assert not err and has_notices(res)
    assert res["status"] == "refused" and res["citations"] == []


# --- error paths ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad", ["", "220", "220:", ":1", "abc:1", "220:1 DROP", "220:" + "9" * 40, "../etc:1"]
)
def test_get_article_bad_id_is_a_tool_error(server, bad):
    err, text = call(server, "get_article", article_id=bad)
    assert err and "Invalid article id" in text


def test_get_article_unknown_id_is_a_clear_error_not_a_guess(server):
    err, text = call(server, "get_article", article_id="220:9999")
    assert err and "Unknown article id" in text and "220:9999" in text
    assert "texts" not in text


@pytest.mark.parametrize("k", [0, -1, 51, 1000])
def test_search_bad_k_is_a_tool_error(server, k):
    err, text = call(server, "search_articles", query="Mietzins", k=k)
    assert err


@pytest.mark.parametrize("bad", ["", "   ", "x" * (MAX_QUESTION_CHARS + 1)])
def test_bad_questions_are_tool_errors(server, bad):
    assert call(server, "search_articles", query=bad)[0]
    assert call(server, "answer_with_citations", question=bad)[0]


def test_bad_language_is_a_tool_error(server):
    assert call(server, "answer_with_citations", question="Frage", language="es")[0]


def test_no_api_key_is_a_clear_error_and_nothing_is_called(tmp_path):
    asked = []

    def reply(request):
        asked.append(request)
        return "{}"

    pipe = ServingPipeline(
        articles=make_articles(), embedder=FakeEmbedder(), index_dir=tmp_path, llm=FakeClient(reply)
    )
    server = mcp_server.create_server(pipe, api_key_available=lambda: False)
    err, text = call(server, "answer_with_citations", question="Wie ist die Kuendigung?")
    assert err and "ANTHROPIC_API_KEY" in text and ".env" in text
    assert asked == []
    # the keyless tools still work
    assert not call(server, "search_articles", query="Mietzins")[0]
    assert not call(server, "get_article", article_id="220:1")[0]


def test_default_key_check_reads_the_environment(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert mcp_server._default_key_available() is False
    monkeypatch.setenv("ANTHROPIC_API_KEY", "  ")
    assert mcp_server._default_key_available() is False
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert mcp_server._default_key_available() is True


def test_budget_error_is_surfaced_as_a_tool_error():
    class Broke:
        def ask(self, *a, **k):
            raise BudgetExceeded("call would cost up to 0.1 USD; month spend 25 of 25 USD cap")

    server = mcp_server.create_server(Broke(), api_key_available=lambda: True)
    err, text = call(server, "answer_with_citations", question="Frage")
    assert err and "Budget" in text and "cap" in text


def test_unexpected_failure_hides_details_and_never_echoes_the_question():
    class Broken:
        def ask(self, *a, **k):
            raise RuntimeError("boom with secret-question-text")

    server = mcp_server.create_server(Broken(), api_key_available=lambda: True)
    err, text = call(server, "answer_with_citations", question="secret-question-text")
    assert err and "RuntimeError" in text and "secret-question-text" not in text


def test_questions_are_not_logged(server, caplog):
    with caplog.at_level("DEBUG"):
        call(server, "search_articles", query="geheime-frage-xyz")
        call(server, "answer_with_citations", question="geheime-frage-xyz")
    assert "geheime-frage-xyz" not in caplog.text


# --- laziness and the real stdio process ---------------------------------------------------------


def test_pipeline_is_not_built_at_creation(monkeypatch):
    import fedlex_bench.serve.pipeline as pl

    def boom(*a, **k):
        raise AssertionError("pipeline built at server creation")

    monkeypatch.setattr(pl, "ServingPipeline", boom)
    server = mcp_server.create_server()
    assert len(list_tools(server)) == 4  # listing tools does not load anything


def test_importing_the_server_module_does_not_load_the_model_stack():
    # pandas is pulled in by serve/__init__ (about 1 s); the model runtimes must wait for first use
    import subprocess

    code = (
        "import sys; import fedlex_bench.serve.mcp_server; "
        "heavy = ('onnxruntime', 'torch', 'sentence_transformers'); "
        "print(sorted(m for m in heavy if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=PROJECT_ROOT, check=True
    )
    assert out.stdout.strip() == "[]", out.stdout + out.stderr


def test_stdio_process_speaks_only_protocol_on_stdout():
    """Start the real server as a subprocess (no model is loaded by listing tools). The SDK client
    fails on a non-protocol line on stdout, so a clean tools/list shows stdout is not polluted."""

    async def go():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "fedlex_bench.serve.mcp_server"],
            cwd=str(PROJECT_ROOT),
            env={"FEDLEX_MCP_LOG": "DEBUG", "PATH": "", "SYSTEMROOT": "C:\\Windows"},
        )
        async with Client(params) as client:
            return [t.name for t in (await client.list_tools()).tools]

    assert sorted(asyncio.run(go())) == [
        "answer_with_citations",
        "get_article",
        "list_acts",
        "search_articles",
    ]


def test_tool_calls_do_not_write_to_stdout(server, capfd):
    call(server, "search_articles", query="Mietzins")
    call(server, "get_article", article_id="220:1")
    call(server, "answer_with_citations", question="Frage")
    assert capfd.readouterr().out == ""
