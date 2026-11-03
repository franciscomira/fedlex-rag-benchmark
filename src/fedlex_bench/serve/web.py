"""The public live demo: a small Starlette app that does retrieval only.

    uv run python -m fedlex_bench.serve.web            # serves on $PORT (default 7860)
    uv run python -m fedlex_bench.serve.web --warm     # load the model and index, then exit

What it does
- GET /                      one static page: a question box and ranked articles with the DE, FR
                             and IT texts side by side, plus a "Recorded full pipeline" section
                             filled from demo/examples.json.
- POST /api/search           {"q": "...", "k": 10} -> {"hits": [...], "notice": "..."}
- GET  /api/article/{id}     one article in all languages.
- GET  /healthz              liveness.

What it deliberately does not do
- It never calls an LLM and never needs an API key. There is no /api/ask route. The full pipeline
  (LLM rerank and cited answer) is shown only as answers recorded offline by `record_examples.py`.
- It never logs or stores questions. Access logging is off, error handling never prints the
  question, and the question travels in a POST body, not in a URL.

Abuse protection (all in memory, so it resets on restart and is per process)
- A sliding-window rate limit per client address and one for the whole service.
- Question length cap, JSON body size cap, and a cap on parallel searches.
- A kill switch: set DEMO_DISABLED=1 and every /api/* route answers 503.
- Security headers on every response.

Settings come from the environment (see `DemoConfig.from_env`):
DEMO_RATE_PER_IP (30/min), DEMO_RATE_GLOBAL (120/min), DEMO_MAX_BODY_BYTES (8192),
DEMO_MAX_CONCURRENT (2), DEMO_MAX_K (10), DEMO_MAX_TEXT_CHARS (6000), DEMO_DISABLED,
TRUST_PROXY, DEMO_EXAMPLES, PORT.
"""

from __future__ import annotations

import argparse
import asyncio
import html
import ipaddress
import json
import logging
import os
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

from fedlex_bench.ingest import PROJECT_ROOT
from fedlex_bench.serve import notices
from fedlex_bench.serve.pipeline import MAX_QUESTION_CHARS, ServingPipeline

log = logging.getLogger("fedlex_bench.serve.web")

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_EXAMPLES = PROJECT_ROOT / "demo" / "examples.json"
DEFAULT_PORT = 7860
DEFAULT_K = 10

# Display names for the acts, used only as labels next to SR numbers the corpus itself reports.
# An act that is not listed here is shown by its SR number alone.
ACT_NAMES = {
    "210": "Civil Code",
    "220": "Code of Obligations",
    "221.214.1": "Consumer Credit Act",
    "221.229.1": "Insurance Contract Act",
    "235.1": "Data Protection Act",
    "241": "Unfair Competition Act",
    "272": "Civil Procedure Code",
    "281.1": "Debt Enforcement and Bankruptcy Act",
}

NOTICE = {
    "notice": " ".join(
        [notices.NOT_OFFICIAL_EN, notices.NOT_OFFICIAL_DE, notices.NOT_LEGAL_ADVICE]
    ),
    "source_note": notices.SOURCE_NOTE,
}


# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = int(raw)  # a bad value should stop the start-up, not be guessed
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")
    return value


@dataclass(frozen=True)
class DemoConfig:
    ip_limit: int = 30  # requests per window per client address
    global_limit: int = 120  # requests per window for the whole service (not measured; tune it)
    window_s: float = 60.0
    max_body_bytes: int = 8192  # 1000 characters escaped as \uXXXX is about 6000 bytes
    max_concurrent: int = 2  # searches running at the same time (CPU protection)
    max_k: int = 10  # most articles one search may return
    max_text_chars: int = 6000  # longest article text sent per language (longer ones are cut)
    disabled: bool = False  # kill switch
    trust_proxy: bool = False  # believe X-Forwarded-For (only behind a proxy you control)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> DemoConfig:
        env = os.environ if env is None else env
        return cls(
            ip_limit=_env_int(env, "DEMO_RATE_PER_IP", cls.ip_limit),
            global_limit=_env_int(env, "DEMO_RATE_GLOBAL", cls.global_limit),
            max_body_bytes=_env_int(env, "DEMO_MAX_BODY_BYTES", cls.max_body_bytes),
            max_concurrent=_env_int(env, "DEMO_MAX_CONCURRENT", cls.max_concurrent),
            max_k=_env_int(env, "DEMO_MAX_K", cls.max_k),
            max_text_chars=_env_int(env, "DEMO_MAX_TEXT_CHARS", cls.max_text_chars),
            disabled=_truthy(env.get("DEMO_DISABLED")),
            trust_proxy=_truthy(env.get("TRUST_PROXY")),
        )


# --------------------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------------------


class SlidingWindowLimiter:
    """At most `limit` admitted requests per key in any `window_s` seconds (in memory)."""

    MAX_KEYS = 10_000  # hard cap on tracked keys; the least recently used one is dropped first

    def __init__(
        self,
        limit: int,
        window_s: float,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int | None = None,
    ) -> None:
        self.limit = limit
        self.window_s = window_s
        self._clock = clock
        self.max_keys = max_keys or self.MAX_KEYS
        # Least recently admitted key first. Only admitted requests create or touch a key, so a
        # flood of rejected requests from many addresses cannot grow this.
        self._events: OrderedDict[str, deque[float]] = OrderedDict()
        self._last_sweep = clock()

    def _trim(self, events: deque[float], now: float) -> None:
        while events and events[0] <= now - self.window_s:
            events.popleft()

    def retry_after(self, key: str) -> float:
        """0 if one more request is allowed now, else the seconds until a slot frees up.
        Never adds a key."""
        events = self._events.get(key)
        if not events:
            return 0.0
        now = self._clock()
        self._trim(events, now)
        if len(events) < self.limit:
            return 0.0
        return max(events[0] + self.window_s - now, 0.001)

    def record(self, key: str) -> None:
        now = self._clock()
        events = self._events.get(key)
        if events is None:
            events = self._events[key] = deque()
        self._events.move_to_end(key)
        self._trim(events, now)
        events.append(now)
        if now - self._last_sweep >= self.window_s:
            self._sweep(now)
        while len(self._events) > self.max_keys:
            self._events.popitem(last=False)

    def _sweep(self, now: float) -> None:
        """Drop keys with no request in the last window."""
        self._last_sweep = now
        for k in list(self._events):
            self._trim(self._events[k], now)
            if not self._events[k]:
                del self._events[k]


# --------------------------------------------------------------------------------------
# What the corpus covers (read from the pipeline, never hard-coded)
# --------------------------------------------------------------------------------------


def _sr_sort_key(sr: str) -> tuple:
    parts = sr.split(".")
    return tuple((0, int(p), "") if p.isdigit() else (1, 0, p) for p in parts)


def corpus_coverage(pipeline: Any) -> list[dict[str, Any]]:
    """[{"sr": "220", "name": "Code of Obligations" | None, "articles": 1234}, ...] in SR order."""
    counts: dict[str, int] = {}
    for article_id in pipeline.corpus._rows:  # the Corpus keeps every article id here
        sr = article_id.partition(":")[0]
        counts[sr] = counts.get(sr, 0) + 1
    return [
        {"sr": sr, "name": ACT_NAMES.get(sr), "articles": counts[sr]}
        for sr in sorted(counts, key=_sr_sort_key)
    ]


# --------------------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------------------


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def render_notices_html() -> str:
    en, legal = _esc(notices.NOT_OFFICIAL_EN), _esc(notices.NOT_LEGAL_ADVICE)
    de, source = _esc(notices.NOT_OFFICIAL_DE), _esc(notices.SOURCE_NOTE)
    return (
        f'<p class="notice-main"><strong>{en}</strong> {legal}</p>'
        f'<p lang="de">{de}</p>'
        f"<p>{source}</p>"
    )


def render_coverage_html(coverage: list[dict[str, Any]] | None) -> str:
    if not coverage:
        return "<p>The list of covered acts could not be read from the corpus.</p>"
    items = []
    for act in coverage:
        name = f" {_esc(act['name'])}" if act["name"] else ""
        items.append(
            f"<li><strong>SR {_esc(act['sr'])}</strong>{name} "
            f'<span class="muted">({act["articles"]} article entries)</span></li>'
        )
    return (
        "<p>This demo searches only these federal acts, in German, French and Italian:</p>"
        f'<ul class="acts">{"".join(items)}</ul>'
        "<p>Everything else in Swiss law (other federal acts, cantonal law, case law) is not "
        "covered. A question about it will still return the closest articles it can find, "
        "which are then simply not relevant.</p>"
    )


def _load_examples(path: Path | None) -> dict[str, Any]:
    """The recorded examples, or an empty placeholder. A broken file is treated as empty."""
    if path is None or not path.exists():
        return {"examples": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.warning("could not read the recorded examples file")
        return {"examples": []}
    if not isinstance(data, dict) or not isinstance(data.get("examples"), list):
        return {"examples": []}
    return data


def render_examples_html(data: dict[str, Any], titles: Callable[[str], str]) -> str:
    """The "Recorded full pipeline" section body.

    `titles(article_id)` returns a short title, or "" if there is none."""
    examples = [e for e in data["examples"] if isinstance(e, dict)]
    if not examples:
        return '<p class="muted">No recorded examples have been added yet.</p>'
    intro_bits = []
    if data.get("recorded_at"):
        intro_bits.append(f"recorded on {_esc(data['recorded_at'])}")
    if data.get("model"):
        intro_bits.append(f"reranker and answer model {_esc(data['model'])}")
    intro = (
        '<p class="muted">Each answer below was produced once, offline, by the full pipeline '
        "(retrieval, LLM reranking, then a drafted answer whose citations are checked by code "
        f"against the retrieved articles){'; ' + ', '.join(intro_bits) if intro_bits else ''}. "
        "The live search above does not call any language model. "
        "Click a cited article to read it.</p>"
    )
    out = [intro]
    for ex in examples:
        status = ex.get("status") if ex.get("status") in ("answered", "refused") else "refused"
        status_text = "Answered" if status == "answered" else "Refused: not in the sources"
        out_of_scope = ex.get("expected_status") == "refused"
        cites = ex.get("citations") or []
        cite_html = "".join(
            f'<button type="button" class="cite" data-article="{_esc(c)}">'
            f"{_esc(c)}{' - ' + _esc(titles(c)) if titles(c) else ''}</button>"
            for c in cites
            if isinstance(c, str)
        )
        answer = ex.get("answer") or (
            "The system found no sufficient basis in the covered acts."
            if status == "refused"
            else ""
        )
        oos_badge = '<span class="badge oos">out-of-scope test</span>' if out_of_scope else ""
        out.append(
            '<details class="example">'
            f'<summary><span class="badge lang">{_esc(str(ex.get("lang", "")).upper())}</span> '
            f'<span class="q">{_esc(ex.get("question", ""))}</span> '
            f'<span class="badge {status}">{status_text}</span>'
            f"{oos_badge}"
            "</summary>"
            '<div class="example-body">'
            f'<p class="answer">{_esc(answer)}</p>'
            + (
                f'<p class="cites-label">Cited articles:</p><div class="cites">{cite_html}</div>'
                if cite_html
                else '<p class="muted">No articles cited.</p>'
            )
            + '<div class="cite-panel" aria-live="polite"></div>'
            "</div></details>"
        )
    return "\n".join(out)


class _Page:
    """Builds the HTML once the corpus is available, then caches it."""

    def __init__(self, pipeline: Any, examples_path: Path | None) -> None:
        self._pipeline = pipeline
        self._examples_path = examples_path
        self._template = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        self._html: str | None = None

    def titles(self, article_id: str) -> str:
        try:
            hit = self._pipeline.get_article(article_id)
        except Exception:  # a label is optional; never fail the page for it
            return ""
        if not hit:
            return ""
        return next((hit.titles[lang] for lang in ("de", "fr", "it") if hit.titles.get(lang)), "")

    def render(self) -> str:
        if self._html is not None:
            return self._html
        coverage: list[dict[str, Any]] | None
        try:
            coverage = corpus_coverage(self._pipeline)
        except Exception as exc:
            log.error("corpus coverage unavailable (%s)", type(exc).__name__)
            coverage = None
        examples = _load_examples(self._examples_path)
        page = (
            self._template.replace("{{NOTICES}}", render_notices_html())
            .replace("{{COVERAGE}}", render_coverage_html(coverage))
            .replace("{{EXAMPLES}}", render_examples_html(examples, self.titles))
            .replace("{{MAX_CHARS}}", str(MAX_QUESTION_CHARS))
        )
        if coverage:  # do not cache a page built without the corpus
            self._html = page
        return page


# --------------------------------------------------------------------------------------
# The app
# --------------------------------------------------------------------------------------

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; "
        "form-action 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


def _json(payload: dict[str, Any], status: int = 200, headers: dict[str, str] | None = None):
    """Every JSON response carries the notices."""
    body = {**payload, **NOTICE}
    out = JSONResponse(body, status_code=status, headers=headers)
    out.headers["Cache-Control"] = "no-store"
    return out


class BodyTooLarge(Exception):
    pass


async def _read_body(request: Request, cap: int) -> bytes:
    """The request body, refusing to read more than `cap` bytes (also when chunked)."""
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > cap:
        raise BodyTooLarge
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > cap:
            raise BodyTooLarge
        chunks.append(chunk)
    return b"".join(chunks)


def client_address(request: Request, trust_proxy: bool) -> str:
    """The address rate limits are counted by.

    With TRUST_PROXY the last X-Forwarded-For entry is used (one trusted proxy in front): it is
    the one that proxy added, whereas earlier entries can be written by the client. The entry
    must be a valid IP address; otherwise (and without TRUST_PROXY) the connection's own address
    is used. So the header can never put arbitrary text into the limiter."""
    peer = request.client.host if request.client else "unknown"
    if trust_proxy:
        parts = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",")]
        parts = [p for p in parts if p]
        if parts:
            try:
                return str(ipaddress.ip_address(parts[-1]))
            except ValueError:
                pass
    return peer


def limit_texts(payload: dict[str, Any], max_chars: int) -> dict[str, Any]:
    """Cut article texts longer than `max_chars` and say so in the payload.

    Nothing else is changed: a text within the limit is returned exactly as it is, and a cut text
    is a plain prefix. `truncated_langs` lists the languages that were cut; the page shows a note
    with a link to the full text at Fedlex."""
    texts = payload.get("texts") or {}
    cut = sorted(lang for lang, text in texts.items() if len(text) > max_chars)
    out = {**payload, "texts": {lang: text[:max_chars] if lang in cut else text
                                for lang, text in texts.items()}}
    out["truncated_langs"] = cut
    out["truncated"] = bool(cut)
    return out


class GuardMiddleware(BaseHTTPMiddleware):
    """Kill switch, rate limits and body-size pre-check for /api/*, security headers for all."""

    def __init__(self, app, *, config: DemoConfig, per_ip: SlidingWindowLimiter,
                 overall: SlidingWindowLimiter) -> None:
        super().__init__(app)
        self.config = config
        self.per_ip = per_ip
        self.overall = overall

    async def dispatch(self, request: Request, call_next):
        response = await self._guard(request, call_next)
        for name, value in SECURITY_HEADERS.items():
            response.headers[name] = value
        return response

    async def _guard(self, request: Request, call_next) -> Response:
        path = request.url.path
        if not (path == "/api" or path.startswith("/api/")):
            return await call_next(request)
        if self.config.disabled:
            return _json({"error": "The demo is switched off at the moment."}, 503)
        ip = client_address(request, self.config.trust_proxy)
        wait = max(self.per_ip.retry_after(ip), self.overall.retry_after("*"))
        if wait > 0:
            return _json(
                {"error": "Too many requests. Please wait a moment and try again."},
                429,
                {"Retry-After": str(max(1, int(wait + 0.999)))},
            )
        self.per_ip.record(ip)
        self.overall.record("*")
        declared = request.headers.get("content-length")
        if declared is not None and not declared.isdigit():
            return _json({"error": "Bad Content-Length header."}, 400)
        if declared is not None and int(declared) > self.config.max_body_bytes:
            return _json({"error": "Request body too large."}, 413)
        return await call_next(request)


def create_app(
    pipeline: Any,
    config: DemoConfig | None = None,
    examples_path: Path | str | None = None,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> Starlette:
    """Build the app around a pipeline (anything with search(), get_article() and corpus).

    Tests pass a pipeline wired to fake models; the server passes the real ServingPipeline."""
    cfg = config or DemoConfig.from_env()
    if examples_path is None:
        examples_path = os.environ.get("DEMO_EXAMPLES") or DEFAULT_EXAMPLES
    page = _Page(pipeline, Path(examples_path))
    slots = asyncio.Semaphore(cfg.max_concurrent)
    assets = {
        "/static/app.css": (STATIC_DIR / "app.css", "text/css; charset=utf-8"),
        "/static/app.js": (STATIC_DIR / "app.js", "text/javascript; charset=utf-8"),
    }

    async def index(request: Request) -> Response:
        return HTMLResponse(await run_in_threadpool(page.render))

    async def static_file(request: Request) -> Response:
        path, media = assets[request.url.path]
        return Response(path.read_bytes(), media_type=media)

    async def healthz(request: Request) -> Response:
        return _json({"status": "ok", "disabled": cfg.disabled})

    async def search(request: Request) -> Response:
        try:
            raw = await _read_body(request, cfg.max_body_bytes)
        except BodyTooLarge:
            return _json({"error": "Request body too large."}, 413)
        try:
            body = json.loads(raw)
        except (ValueError, RecursionError):  # bad JSON, or nested so deeply that parsing gives up
            return _json({"error": "The body must be JSON like {\"q\": \"...\"}."}, 400)
        if not isinstance(body, dict) or not isinstance(body.get("q"), str):
            return _json({"error": "Field q (the question, a string) is required."}, 400)
        q = body["q"].strip()
        if not q:
            return _json({"error": "The question is empty."}, 400)
        if len(q) > MAX_QUESTION_CHARS:
            too_long = f"The question is longer than {MAX_QUESTION_CHARS} characters."
            return _json({"error": too_long}, 400)
        k = body.get("k", min(DEFAULT_K, cfg.max_k))
        if isinstance(k, bool) or not isinstance(k, int):
            return _json({"error": "Field k must be a whole number."}, 400)
        k = max(1, min(k, cfg.max_k))
        try:
            async with slots:
                hits = await run_in_threadpool(pipeline.search, q, k)
        except ValueError:
            return _json({"error": "The question could not be used."}, 400)
        except Exception as exc:  # never print the question or the exception text
            log.error("search failed (%s)", type(exc).__name__)
            return _json({"error": "Search failed. Please try again later."}, 500)
        return _json({"hits": [limit_texts(h.to_dict(), cfg.max_text_chars) for h in hits]})

    async def article(request: Request) -> Response:
        article_id = request.path_params["article_id"]
        try:
            hit = await run_in_threadpool(pipeline.get_article, article_id)
        except Exception as exc:
            log.error("article lookup failed (%s)", type(exc).__name__)
            return _json({"error": "Lookup failed. Please try again later."}, 500)
        if hit is None:
            return _json({"error": "Unknown article id."}, 404)
        return _json({"article": limit_texts(hit.to_dict(), cfg.max_text_chars)})

    async def not_found(request: Request, exc: Exception) -> Response:
        if request.url.path.startswith("/api/"):
            return _json({"error": "Not found."}, 404)
        return PlainTextResponse("Not found", status_code=404)

    async def method_not_allowed(request: Request, exc: Exception) -> Response:
        if request.url.path.startswith("/api/"):
            return _json({"error": "Method not allowed."}, 405)
        return PlainTextResponse("Method not allowed", status_code=405)

    routes = [
        Route("/", index, methods=["GET"]),
        Route("/healthz", healthz, methods=["GET"]),
        Route("/api/search", search, methods=["POST"]),
        Route("/api/article/{article_id:path}", article, methods=["GET"]),
        *[Route(p, static_file, methods=["GET"]) for p in assets],
    ]
    per_ip = SlidingWindowLimiter(cfg.ip_limit, cfg.window_s, clock)
    overall = SlidingWindowLimiter(cfg.global_limit, cfg.window_s, clock)
    return Starlette(
        routes=routes,
        middleware=[
            Middleware(GuardMiddleware, config=cfg, per_ip=per_ip, overall=overall),
        ],
        exception_handlers={404: not_found, 405: method_not_allowed},
    )


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------


def warm(pipeline: ServingPipeline) -> int:
    """Load the model, build or load the passage index and run one search. Returns the number of
    articles in the corpus. At image build time this fills the index cache on disk."""
    pipeline.search("warm-up", 1)
    return len(pipeline.corpus)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--warm", action="store_true", help="load model and index, then exit")
    ap.add_argument("--host", default="0.0.0.0")  # noqa: S104 (a container must listen on all)
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", DEFAULT_PORT)))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    pipeline = ServingPipeline()
    n = warm(pipeline)  # slow start-up is better than a slow first visitor
    logging.warning("ready: %d articles", n)
    if args.warm:
        return
    import uvicorn

    uvicorn.run(
        create_app(pipeline),
        host=args.host,
        port=args.port,
        access_log=False,  # access lines would carry paths and addresses; keep no request log
        log_level="warning",
        proxy_headers=False,  # we handle X-Forwarded-For ourselves, and only with TRUST_PROXY=1
    )


if __name__ == "__main__":
    main()
