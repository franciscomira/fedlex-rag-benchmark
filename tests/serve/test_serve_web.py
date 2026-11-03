"""Demo web app and example recorder: no network, no model, no LLM, no API key."""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "retrieval"))
from retrieval_fixtures import FakeEmbedder, make_articles  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from fedlex_bench.llm.fake import FakeClient  # noqa: E402
from fedlex_bench.serve import ServingPipeline, notices, record_examples  # noqa: E402
from fedlex_bench.serve import web as web_mod  # noqa: E402
from fedlex_bench.serve.web import (  # noqa: E402
    STATIC_DIR,
    DemoConfig,
    SlidingWindowLimiter,
    create_app,
)

EN = notices.NOT_OFFICIAL_EN
DE = notices.NOT_OFFICIAL_DE
LEGAL = notices.NOT_LEGAL_ADVICE


class NoLLMPipeline(ServingPipeline):
    """The real pipeline with fake models; any attempt to reach an LLM fails the test."""

    def _llm_client(self):
        raise AssertionError("the public demo must never create an LLM client")


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def pipe(tmp_path):
    return NoLLMPipeline(articles=make_articles(), embedder=FakeEmbedder(), index_dir=tmp_path)


def make_client(pipe, tmp_path, *, client=("203.0.113.5", 50000), clock=None, **cfg):
    app = create_app(
        pipe,
        DemoConfig(**cfg),
        examples_path=tmp_path / "examples.json",
        **({"clock": clock} if clock else {}),
    )
    return TestClient(app, client=client)


@pytest.fixture
def client(pipe, tmp_path):
    return make_client(pipe, tmp_path)


def post(c, q="Mietzins", **extra):
    return c.post("/api/search", json={"q": q, **extra})


# ----- the page -----


def test_page_shows_required_notices_and_coverage(client):
    r = client.get("/")
    assert r.status_code == 200
    page = r.text
    for text in (EN, DE, LEGAL, notices.SOURCE_NOTE):
        assert text in page
    assert "SR 210" in page and "SR 220" in page  # read from the corpus
    assert "SR 221.214.1" not in page  # an act that is not in the test corpus is not claimed
    assert "wrong sometimes" in page
    assert "{{" not in page  # every placeholder was filled
    assert 'rel="stylesheet" href="/static/app.css"' in page


def test_page_and_assets_load_nothing_from_outside():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    css = (STATIC_DIR / "app.css").read_text(encoding="utf-8")
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert "http" not in html and "http" not in css
    # the only URL-like text in the script is the check that source links start with https://
    assert re.findall(r"https?://[^\s\"')]*", js) == ["https://"]
    assert "innerHTML" not in js  # server text is only ever inserted as text


def test_static_files_are_served(client):
    css = client.get("/static/app.css")
    js = client.get("/static/app.js")
    assert css.status_code == 200 and css.headers["content-type"].startswith("text/css")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]
    assert client.get("/static/../pipeline.py").status_code == 404


def test_examples_placeholder_renders(pipe, tmp_path):
    (tmp_path / "examples.json").write_text('{"examples": []}', encoding="utf-8")
    r = make_client(pipe, tmp_path).get("/")
    assert r.status_code == 200
    assert "Recorded full pipeline" in r.text
    assert "No recorded examples have been added yet." in r.text


def test_missing_or_broken_examples_file_is_a_placeholder(pipe, tmp_path):
    missing = make_client(pipe, tmp_path)  # tmp_path/examples.json does not exist
    assert "No recorded examples" in missing.get("/").text
    (tmp_path / "examples.json").write_text("{not json", encoding="utf-8")
    assert "No recorded examples" in make_client(pipe, tmp_path).get("/").text
    (tmp_path / "examples.json").write_text('{"examples": "oops"}', encoding="utf-8")
    assert "No recorded examples" in make_client(pipe, tmp_path).get("/").text


def test_shipped_examples_file_is_valid():
    data = json.loads((Path(web_mod.DEFAULT_EXAMPLES)).read_text(encoding="utf-8"))
    assert isinstance(data["examples"], list)


def test_recorded_examples_render_escaped_with_citations(pipe, tmp_path):
    data = {
        "recorded_at": "2026-11-03",
        "model": "claude-haiku-5-5",
        "examples": [
            {
                "lang": "de",
                "question": "Wie ist die <b>Kuendigung</b>?",
                "status": "answered",
                "answer": "Schriftlich. <script>alert(1)</script>",
                "citations": ["220:1", "999:9"],
                "expected_status": "answered",
            },
            {"lang": "en", "question": "Moon?", "status": "refused", "answer": "",
             "citations": [], "expected_status": "refused"},
        ],
    }
    (tmp_path / "examples.json").write_text(json.dumps(data), encoding="utf-8")
    page = make_client(pipe, tmp_path).get("/").text
    assert "&lt;b&gt;Kuendigung&lt;/b&gt;" in page and "<b>Kuendigung</b>" not in page
    assert "<script>alert(1)" not in page
    assert 'data-article="220:1"' in page and "Mietzins" in page  # title read from the corpus
    assert 'data-article="999:9"' in page  # an id the corpus lacks is shown as is
    assert "Refused: not in the sources" in page and "out-of-scope test" in page
    assert "recorded on 2026-11-03" in page


# ----- the API -----


def test_search_returns_hits_and_notice(client):
    r = post(client, "Der Mieter zahlt den Mietzins", k=2)
    assert r.status_code == 200
    body = r.json()
    assert [h["rank"] for h in body["hits"]] == [1, 2]
    assert body["hits"][0]["article_id"] == "220:1"
    assert set(body["hits"][0]["texts"]) == {"de", "fr", "it"}
    assert EN in body["notice"] and DE in body["notice"] and LEGAL in body["notice"]
    assert body["source_note"] == notices.SOURCE_NOTE


def test_k_is_capped_at_twenty_and_floored_at_one(pipe, tmp_path):
    seen = []
    real = pipe.search
    pipe.search = lambda q, k=10: (seen.append(k), real(q, k))[1]
    c = make_client(pipe, tmp_path)
    assert post(c, k=500).status_code == 200
    assert post(c, k=0).status_code == 200
    assert post(c).status_code == 200
    assert seen == [20, 1, 10]


@pytest.mark.parametrize(
    "payload",
    [{}, {"q": 5}, {"q": ""}, {"q": "   "}, {"q": "x" * 1001}, {"q": "ok", "k": "3"},
     {"q": "ok", "k": True}, {"q": "ok", "k": 2.5}, ["q"]],
)
def test_bad_requests_get_400_with_notice(client, payload):
    r = client.post("/api/search", json=payload)
    assert r.status_code == 400
    assert "error" in r.json() and LEGAL in r.json()["notice"]


def test_non_json_body_gets_400(client):
    r = client.post("/api/search", content=b"not json")
    assert r.status_code == 400 and LEGAL in r.json()["notice"]


def test_longest_allowed_question_is_accepted(client):
    assert post(client, "ü" * 1000).status_code == 200


def test_there_is_no_ask_route_and_no_llm_is_used(client):
    for method in ("get", "post"):
        r = getattr(client, method)("/api/ask")
        assert r.status_code in (404, 405)
        assert LEGAL in r.json()["notice"]
    assert post(client).status_code == 200  # NoLLMPipeline would raise if an LLM were touched
    assert not any("/api/ask" in str(r.path) for r in client.app.routes)


def test_article_endpoint(client):
    r = client.get("/api/article/220:1")
    assert r.status_code == 200
    assert r.json()["article"]["article_id"] == "220:1" and LEGAL in r.json()["notice"]
    slash = client.get("/api/article/220:u1%2F1")  # ids can contain a slash
    assert slash.status_code == 200 and slash.json()["article"]["article_id"] == "220:u1/1"


def test_unknown_article_is_404_with_notice(client):
    r = client.get("/api/article/999:9")
    assert r.status_code == 404
    assert "error" in r.json() and EN in r.json()["notice"]


def test_unknown_api_path_and_wrong_method(client):
    r = client.get("/api/nothing")
    assert r.status_code == 404 and LEGAL in r.json()["notice"]
    r = client.get("/api/search")
    assert r.status_code == 405 and LEGAL in r.json()["notice"]
    assert client.get("/nothing").status_code == 404


def test_healthz_has_notice(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert LEGAL in r.json()["notice"]


def test_search_failure_does_not_leak_the_question(pipe, tmp_path, caplog):
    secret = "geheime-frage-4711"

    def boom(q, k=10):
        raise RuntimeError(f"model broke on {q}")

    pipe.search = boom
    c = make_client(pipe, tmp_path)
    with caplog.at_level(logging.DEBUG):
        r = post(c, secret)
    assert r.status_code == 500 and LEGAL in r.json()["notice"]
    assert secret not in r.text and secret not in caplog.text


# ----- kill switch -----


def test_kill_switch_returns_503_on_api_only(pipe, tmp_path):
    c = make_client(pipe, tmp_path, disabled=True)
    for r in (post(c), c.get("/api/article/220:1"), c.get("/api/anything")):
        assert r.status_code == 503
        assert LEGAL in r.json()["notice"]
    assert c.get("/").status_code == 200
    assert c.get("/healthz").json()["disabled"] is True


# ----- rate limits -----


def test_per_ip_limit_and_window_slide(pipe, tmp_path):
    clock = Clock()
    c = make_client(pipe, tmp_path, clock=clock, ip_limit=3, global_limit=100)
    assert [post(c).status_code for _ in range(3)] == [200, 200, 200]
    blocked = post(c)
    assert blocked.status_code == 429
    assert int(blocked.headers["retry-after"]) >= 1
    assert LEGAL in blocked.json()["notice"]
    clock.now += 61
    assert post(c).status_code == 200


def test_blocked_requests_do_not_extend_the_block(pipe, tmp_path):
    clock = Clock()
    c = make_client(pipe, tmp_path, clock=clock, ip_limit=2, global_limit=100)
    post(c), post(c)
    for _ in range(20):
        assert post(c).status_code == 429
    clock.now += 61
    assert post(c).status_code == 200


def test_other_addresses_are_not_limited_by_one(pipe, tmp_path):
    clock = Clock()
    c = make_client(pipe, tmp_path, clock=clock, ip_limit=1, global_limit=100, trust_proxy=True)
    first = c.post("/api/search", json={"q": "a"}, headers={"X-Forwarded-For": "198.51.100.1"})
    again = c.post("/api/search", json={"q": "a"}, headers={"X-Forwarded-For": "198.51.100.1"})
    other = c.post("/api/search", json={"q": "a"}, headers={"X-Forwarded-For": "198.51.100.2"})
    assert (first.status_code, again.status_code, other.status_code) == (200, 429, 200)


def test_global_limit(pipe, tmp_path):
    clock = Clock()
    c = make_client(pipe, tmp_path, clock=clock, ip_limit=100, global_limit=4, trust_proxy=True)
    codes = [
        c.post("/api/search", json={"q": "a"}, headers={"X-Forwarded-For": f"198.51.100.{i}"})
        .status_code
        for i in range(6)
    ]
    assert codes == [200, 200, 200, 200, 429, 429]
    clock.now += 61
    assert c.post("/api/search", json={"q": "a"}, headers={"X-Forwarded-For": "198.51.100.99"}
                  ).status_code == 200


def test_forwarded_for_is_ignored_unless_trusted(pipe, tmp_path):
    c = make_client(pipe, tmp_path, ip_limit=2, global_limit=100, trust_proxy=False)
    codes = [
        c.post("/api/search", json={"q": "a"}, headers={"X-Forwarded-For": f"198.51.100.{i}"})
        .status_code
        for i in range(3)
    ]
    assert codes == [200, 200, 429]  # spoofed headers did not buy extra requests


def test_trusted_proxy_uses_the_last_entry(pipe, tmp_path):
    c = make_client(pipe, tmp_path, ip_limit=1, global_limit=100, trust_proxy=True)

    def go(xff):
        return c.post("/api/search", json={"q": "a"}, headers={"X-Forwarded-For": xff}).status_code

    # the client writes the first entry; the proxy appended the last one
    assert go("1.1.1.1, 10.0.0.9") == 200
    assert go("2.2.2.2, 10.0.0.9") == 429


def test_healthz_and_page_are_not_rate_limited(pipe, tmp_path):
    c = make_client(pipe, tmp_path, ip_limit=1, global_limit=1)
    for _ in range(5):
        assert c.get("/healthz").status_code == 200
        assert c.get("/").status_code == 200


def test_sliding_window_limiter_unit():
    clock = Clock()
    lim = SlidingWindowLimiter(2, 10.0, clock)
    assert lim.retry_after("a") == 0
    lim.record("a")
    clock.now += 4
    lim.record("a")
    assert lim.retry_after("a") == pytest.approx(6.0)
    assert lim.retry_after("b") == 0
    clock.now += 6.5
    assert lim.retry_after("a") == 0  # the first event left the window


def test_limiter_forgets_quiet_keys():
    clock = Clock()
    lim = SlidingWindowLimiter(5, 10.0, clock)
    lim._SWEEP_AT = 50
    for i in range(60):
        lim.record(f"ip{i}")
    clock.now += 100
    lim.record("late")
    assert len(lim._events) < 10


# ----- size caps -----


def test_declared_body_too_large(pipe, tmp_path):
    c = make_client(pipe, tmp_path, max_body_bytes=200)
    r = c.post("/api/search", json={"q": "x" * 500})
    assert r.status_code == 413 and LEGAL in r.json()["notice"]


def test_streamed_body_too_large_without_content_length(pipe, tmp_path):
    c = make_client(pipe, tmp_path, max_body_bytes=200)

    def chunks():
        for _ in range(10):
            yield b'{"q": "' + b"x" * 60 + b'"}'

    r = c.post("/api/search", content=chunks())
    assert r.status_code == 413


def test_bad_content_length_header(client):
    r = client.post("/api/search", content=b"{}", headers={"Content-Length": "abc"})
    assert r.status_code == 400


# ----- headers -----


@pytest.mark.parametrize(
    "path", ["/", "/healthz", "/api/article/999:9", "/nothing", "/static/app.js"]
)
def test_security_headers_everywhere(client, path):
    h = client.get(path).headers
    assert "default-src 'self'" in h["content-security-policy"]
    assert "frame-ancestors 'none'" in h["content-security-policy"]
    assert h["x-content-type-options"] == "nosniff"
    assert h["referrer-policy"] == "no-referrer"
    assert h["x-frame-options"] == "DENY"


def test_security_headers_on_errors(pipe, tmp_path):
    c = make_client(pipe, tmp_path, disabled=True)
    assert c.post("/api/search", json={"q": "a"}).headers["x-frame-options"] == "DENY"
    c = make_client(pipe, tmp_path, ip_limit=1)
    post(c)
    assert post(c).headers["x-content-type-options"] == "nosniff"


# ----- privacy -----


def test_question_text_is_never_logged_or_stored(pipe, tmp_path, caplog):
    secret = "Streng-vertrauliche-Frage-zum-Mietrecht-93817"
    c = make_client(pipe, tmp_path)
    with caplog.at_level(logging.DEBUG):
        logging.getLogger().setLevel(logging.DEBUG)
        assert post(c, secret).status_code == 200
        assert post(c, secret + "!" * 2000).status_code == 400  # too long: error path
        assert c.post("/api/search", json={"q": secret, "k": "bad"}).status_code == 400
    assert secret not in caplog.text
    assert all(secret not in rec.getMessage() for rec in caplog.records)
    assert all(secret not in str(rec.args) for rec in caplog.records)
    needle = secret.encode()
    for f in tmp_path.rglob("*"):
        if f.is_file():
            assert needle not in f.read_bytes(), f"question found in {f}"


def test_uvicorn_is_started_without_access_log(monkeypatch, pipe):
    calls = {}

    class FakeUvicorn:
        @staticmethod
        def run(app, **kw):
            calls.update(kw)

    monkeypatch.setitem(sys.modules, "uvicorn", FakeUvicorn)
    monkeypatch.setattr(web_mod, "ServingPipeline", lambda: pipe)
    web_mod.main(["--port", "7861"])
    assert calls["access_log"] is False and calls["port"] == 7861
    assert calls["proxy_headers"] is False


def test_warm_mode_exits_without_serving(monkeypatch, pipe):
    monkeypatch.setattr(web_mod, "ServingPipeline", lambda: pipe)
    monkeypatch.setitem(sys.modules, "uvicorn", None)  # importing it would fail
    web_mod.main(["--warm"])


# ----- settings -----


def test_config_from_env():
    cfg = DemoConfig.from_env(
        {"DEMO_RATE_PER_IP": "5", "DEMO_DISABLED": "1", "TRUST_PROXY": "true"}
    )
    assert (cfg.ip_limit, cfg.global_limit, cfg.disabled, cfg.trust_proxy) == (5, 600, True, True)
    default = DemoConfig.from_env({})
    assert (default.ip_limit, default.global_limit, default.disabled) == (30, 600, False)
    assert default.trust_proxy is False
    with pytest.raises(ValueError):
        DemoConfig.from_env({"DEMO_RATE_PER_IP": "0"})


# ----- the example recorder -----


def llm_reply(request):
    if "ranking" in request.user or "ranking" in request.system:
        return json.dumps({"ranking": [1, 2, 3]})
    return json.dumps(
        {"status": "answered", "answer": "Notice must be in writing.",
         "citations": ["220:1", "999:9"], "pointers": ["220:2"]}
    )


@pytest.fixture
def full_pipe(tmp_path):
    return ServingPipeline(
        articles=make_articles(), embedder=FakeEmbedder(), index_dir=tmp_path,
        llm=FakeClient(llm_reply),
    )


def test_example_questions_cover_the_languages_and_include_refusals():
    qs = record_examples.EXAMPLE_QUESTIONS
    assert len(qs) == 10
    per_lang = {lang: sum(q["lang"] == lang for q in qs) for lang in ("en", "de", "fr", "it")}
    assert all(2 <= n <= 3 for n in per_lang.values()), per_lang
    assert 1 <= sum(q["expected_status"] == "refused" for q in qs) <= 2
    assert len({q["question"] for q in qs}) == 10
    assert all(0 < len(q["question"]) <= 1000 for q in qs)


def test_record_runs_ask_once_per_question(full_pipe):
    from datetime import UTC, datetime

    data = record_examples.record(
        full_pipe, record_examples.EXAMPLE_QUESTIONS[:3], now=datetime(2026, 11, 3, tzinfo=UTC)
    )
    assert data["recorded_at"] == "2026-11-03" and len(data["examples"]) == 3
    first = data["examples"][0]
    assert first["status"] == "answered" and first["citations"] == ["220:1"]  # invented id dropped
    assert first["n_invalid_ids"] == 1 and first["pointers"] == ["220:2"]
    assert first["lang"] == "en" and "texts" not in first  # ids only, never statute text
    assert "Mieter zahlt" not in json.dumps(data)


def test_recorded_file_round_trips_into_the_page(full_pipe, tmp_path):
    out = tmp_path / "out" / "examples.json"
    data = record_examples.record(full_pipe, record_examples.EXAMPLE_QUESTIONS[:2])
    record_examples.write_examples(data, out)
    assert json.loads(out.read_text(encoding="utf-8")) == data
    c = TestClient(create_app(full_pipe, DemoConfig(), examples_path=out))
    page = c.get("/").text
    assert 'data-article="220:1"' in page
    assert record_examples.EXAMPLE_QUESTIONS[0]["question"] in page


def test_recorder_dry_run_calls_nothing(monkeypatch, capsys, tmp_path):
    def forbidden(*a, **k):
        raise AssertionError("must not build the pipeline in a dry run")

    monkeypatch.setattr(record_examples, "ServingPipeline", forbidden)
    out = tmp_path / "examples.json"
    assert record_examples.main(["--out", str(out)]) == 0
    assert "Dry run" in capsys.readouterr().out
    assert not out.exists()


def test_recorder_needs_a_key(monkeypatch, tmp_path):
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def forbidden(*a, **k):
        raise AssertionError("must not build the pipeline without a key")

    monkeypatch.setattr(record_examples, "ServingPipeline", forbidden)
    assert record_examples.main(["--yes", "--out", str(tmp_path / "e.json")]) == 1


def test_recorder_writes_nothing_if_a_question_fails(full_pipe, tmp_path):
    class Boom:
        def ask(self, q, lang):
            raise RuntimeError("API down")

    out = tmp_path / "examples.json"
    with pytest.raises(RuntimeError):
        record_examples.write_examples(record_examples.record(Boom()), out)
    assert not out.exists()
