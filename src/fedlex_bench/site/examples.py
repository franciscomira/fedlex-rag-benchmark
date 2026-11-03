"""Second page: the recorded full-pipeline answers of ``demo/examples.json``.

There is no live demo. The page shows the recorded answers exactly as stored, the cited articles
as act and article numbers parsed from the ids (no article text, no links into the Fedlex
filestore), and notes computed from expected versus recorded status.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

from fedlex_bench.site.data import read_json
from fedlex_bench.site.render import page_shell

e = escape

EXAMPLES_FILE = Path("demo") / "examples.json"
FEDLEX_URL = "https://www.fedlex.admin.ch"
TITLE = "Recorded examples: the full pipeline"
STATUS_TEXT = {"answered": "Answered", "refused": "Refused: not in the sources"}
LANG_NAMES = {"en": "English", "de": "German", "fr": "French", "it": "Italian"}
WEB_COMMAND = "uv run --group serve python -m fedlex_bench.serve.web"
MCP_COMMAND = "uv run --group serve python -m fedlex_bench.serve.mcp_server"
CORPUS_COMMAND = "uv run python -m fedlex_bench.chunk"
COMMANDS = (WEB_COMMAND, MCP_COMMAND, CORPUS_COMMAND)


def article_label(article_id: str) -> str:
    """``220:335c`` -> ``SR 220, article 335c (id 220:335c)``; parsed by code, never generated."""
    sr, _, number = article_id.partition(":")
    if not sr or not number:
        return f"id {article_id}"
    return f"SR {sr}, article {number} (id {article_id})"


def status_text(status: str) -> str:
    return STATUS_TEXT.get(status, status)


def mismatch_note(expected: str, actual: str) -> str:
    """Plain-words note from expected versus recorded status; empty when nothing is notable."""
    if expected == actual == "refused":
        return (f"Expected: {expected}. Recorded: {actual}. The question is outside the "
                "sources, and the system refused, as intended.")
    if expected == actual:
        return ""
    if actual == "refused":
        return (f"Expected: {expected}. Recorded: {actual}. "
                "The system refused a question that the sources answer at least in part: "
                "a miss.")
    return (f"Expected: {expected}. Recorded: {actual}. "
            "The system answered a question it should have refused: a miss.")


def _articles(ids: list[str]) -> str:
    return "".join(f"<li>{e(article_label(i))}</li>" for i in ids)


def example_html(n: int, ex: dict) -> str:
    lang = str(ex["lang"])
    note = mismatch_note(ex["expected_status"], ex["status"])
    differs = ex["expected_status"] != ex["status"]
    cls = "example differs" if differs else "example"
    parts = [
        f'<article class="{cls}" id="example-{n}" data-status="{e(ex["status"])}" '
        f'data-expected="{e(ex["expected_status"])}">',
        f'<h3>Example {n} <span class="langtag" lang="{e(lang)}" '
        f'title="{e(LANG_NAMES.get(lang, lang))}">{e(lang.upper())}</span></h3>',
        f'<p class="question">{e(ex["question"])}</p>',
        f'<p><strong class="status status-{e(ex["status"])}">{e(status_text(ex["status"]))}'
        "</strong></p>",
    ]
    if note:
        parts.append(f'<p class="note{" warn" if differs else ""}">{e(note)}</p>')
    parts.append(f'<p class="answer" lang="{e(lang)}">{e(ex["answer"])}</p>')
    if ex["citations"]:
        parts.append(f'<p class="fine">Cited articles:</p><ul class="cites">'
                     f"{_articles(ex['citations'])}</ul>")
    if ex["pointers"]:
        parts.append(f'<p class="fine">Related articles the answer points to:</p><ul class="cites">'
                     f"{_articles(ex['pointers'])}</ul>")
    meta = [f"{len(ex['top_ranked'])} top-ranked articles shown to the answerer"]
    if ex.get("n_invalid_ids"):
        meta.append(f"{ex['n_invalid_ids']} invalid ids rejected by code")
    meta.append(f"cost {ex['cost_usd']} USD")
    meta.append(f"{ex['latency_s']} s")
    parts.append(f'<p class="fine">{e("; ".join(meta))}</p></article>')
    return "".join(parts)


def render_examples(data: dict) -> str:
    items = data["examples"]
    n = len(items)
    answered = sum(1 for x in items if x["status"] == "answered")
    refused = sum(1 for x in items if x["status"] == "refused")
    differing = sum(1 for x in items if x["status"] != x["expected_status"])
    intro = (
        '<section id="top"><h2>No live demo</h2>'
        "<p><strong>There is no live demo.</strong> These are "
        f"{n} answers recorded once on {e(str(data['recorded_at']))} with the full pipeline "
        f"({e(str(data['model']))} writes the answers). All {n} recorded answers are shown, in "
        "the order recorded; none was picked for how it looks.</p>"
        f"<p>{answered} were answered and {refused} refused as not in the sources; "
        f"{differing} had a recorded status different from the expected one and "
        f"{'is' if differing == 1 else 'are'} marked. Recording cost "
        f"{e(str(data['total_cost_usd']))} USD in total. Answers are shown exactly as recorded "
        "and may be wrong; no article text is shown. To read an article, look it up by its "
        f'number on <a href="{FEDLEX_URL}">Fedlex</a>, the official source.</p></section>'
    )
    examples = (
        '<section id="examples"><h2>The answers</h2>'
        + "".join(example_html(i, x) for i, x in enumerate(items, 1))
        + "</section>"
    )
    cmds = "".join(f'<code class="cmd">{e(c)}</code>' for c in (WEB_COMMAND, MCP_COMMAND))
    tryit = (
        '<section id="try"><h2>Try the search yourself</h2>'
        "<p>Run it on your own machine. The first command starts the retrieval-only search (no "
        "LLM, no key); the second starts the MCP server.</p>"
        f"{cmds}"
        "<p>The search demo needs the corpus built locally first: "
        f'<code class="cmd">{e(CORPUS_COMMAND)}</code></p>'
        '<p><a href="index.html">Back to the results page</a></p></section>'
    )
    nav = ('<a href="#top">About</a><a href="#examples">Answers</a>'
           '<a href="#try">Try it yourself</a><a href="index.html">Results</a>')
    return page_shell(
        TITLE,
        "Recorded answers of the full pipeline over Swiss federal law. There is no live demo.",
        nav,
        intro + examples + tryit,
        "Statute text is not shown on this page. Built by "
        "<code>uv run python -m fedlex_bench.site build</code> from "
        "<code>demo/examples.json</code>; no script, font or tracker is loaded from elsewhere.",
    )


def load_examples(root: Path) -> dict | None:
    path = root / EXAMPLES_FILE
    return read_json(path) if path.exists() else None
