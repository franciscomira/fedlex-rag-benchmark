# MCP server

A stdio [Model Context Protocol](https://modelcontextprotocol.io) server over the project's serving
pipeline: retrieval with the fine-tuned embedder, exact article lookup, and cited answers. Code:
`src/fedlex_bench/serve/mcp_server.py`. Tests: `tests/serve/test_serve_mcp.py`.

```text
uv run --group serve python -m fedlex_bench.serve.mcp_server
```

The server starts without loading anything heavy. The model and the index load on the first tool
call (measured on the author's machine: about 5 s for the first `search_articles`, about 20 ms for
the next ones; see "Latency" below).

## Requirements

- A checkout of this repository with the `serve` dependency group (`uv sync --group serve`).
- The corpus at `data/processed/articles.parquet` (built by the ingest step; it is not committed).
- The tuned int8 model. It is read from `$FEDLEX_MODEL_DIR`, else `models/r1_inbatch/onnx`, else the
  pinned revision on the Hugging Face Hub (downloaded once into the Hub cache).
- Only for `answer_with_citations`: `ANTHROPIC_API_KEY`, in the repository's `.env` or in the
  server's environment. The monthly budget cap in `config.toml` applies and cannot be bypassed from
  the server. Each call costs about 0.003 USD of your API budget. In addition, a server process
  allows at most 20 `answer_with_citations` calls (set `FEDLEX_MCP_MAX_PAID_CALLS` to change it, `0`
  disables the paid tool), and the calls run one at a time. Restart the server to reset the counter.

## Install

Use an absolute path to the repository. Replace `C:\Dev\Fedlex_RAG_Benchmark` with your checkout.

### Claude Code

```text
claude mcp add fedlex-rag -- uv --directory C:\Dev\Fedlex_RAG_Benchmark run --group serve python -m fedlex_bench.serve.mcp_server
```

Add `--scope user` to make it available in every project. Check with `claude mcp list`.

### Claude Desktop

Edit `claude_desktop_config.json` (Settings, Developer, Edit Config) and restart the app:

```json
{
  "mcpServers": {
    "fedlex-rag": {
      "command": "uv",
      "args": [
        "--directory", "C:\\Dev\\Fedlex_RAG_Benchmark",
        "run", "--group", "serve",
        "python", "-m", "fedlex_bench.serve.mcp_server"
      ]
    }
  }
}
```

If `uv` is not on the PATH of the desktop app, use its absolute path as `command`. To enable
`answer_with_citations`, put `ANTHROPIC_API_KEY=...` in the repository's `.env` (git-ignored); do not
paste keys into the config file.

## Tools

Every result is a JSON object whose first field, `notices`, carries the three project notices
(`not_official_en`, `not_official_de`, `not_legal_advice`) and a `source_note`: statute text is
verbatim Fedlex data, while rankings, scores, answers and the choice of citations are generated.
Tool errors (bad input, unknown id, no key, budget, paid-call limit) are plain error messages without statute text.

| Tool | Arguments | Cost | Returns |
| --- | --- | --- | --- |
| `list_acts` | none | free | The acts (SR numbers) in the corpus, read from the corpus: SR, English working title, number of articles, number repealed, consolidation date. |
| `search_articles` | `query` (at most 1000 chars), `k` (1 to 50, default 10) | free, no key | Ranked articles: `rank`, `article_id`, `sr`, `score` (a similarity, not a probability), `titles` and short `snippets` in DE/FR/IT, `source_urls`, `is_repealed`, `consolidation_date`. No full text. Uses the tuned int8 embedder only, no LLM. |
| `get_article` | `article_id` as `<SR>:<article>`, e.g. `220:257d` | free | Exact lookup: full German, French and Italian text, titles, repealed flag, consolidation date, source URLs. An invalid or unknown id is a tool error; nothing is guessed. |
| `answer_with_citations` | `question` (at most 1000 chars), `language` (`en`, `de`, `fr`, `it`; default `en`) | paid, needs `ANTHROPIC_API_KEY` | The frozen measured pipeline (tuned retrieval, Haiku rerank, cited answer). `status` is `answered` or `refused` (the sources do not contain the answer); `answer`; `citations` and `pointers` as article ids with titles; `n_invalid_ids`; `ranked_articles`; `cost_usd`. Without a key it returns an error and calls nothing. |

How citations are handled: article ids, titles and texts come from the corpus. The model only
proposes ids; code keeps those that were among the retrieved articles and drops the rest.
`n_invalid_ids` reports how many were dropped. Fetch the text of a cited article with
`get_article`. Questions and keys are never logged (the HTTP and SDK loggers stay at WARNING even with
`FEDLEX_MCP_LOG=DEBUG`, because at DEBUG they print request options); logs go to stderr, and stdout carries only the
protocol.

Example prompts: "Which acts does the Fedlex server cover?", "Search for the notice period for
residential leases, then show me the German text of the first hit", "Answer with citations: wann
kann der Vermieter bei Zahlungsrückstand kündigen?".

## Latency

On the author's machine (Windows 11, CPU, int8 ONNX model): server start and handshake 2.9 s
(including `uv run`), first `search_articles` 5.2 s (model and index load), second 0.02 s,
`get_article` a few milliseconds. Your numbers will differ.

## What this adds and what it does not

What it adds:

- **Evaluated retrieval.** The embedder behind `search_articles` was fine-tuned and measured on the
  project's frozen benchmark; results with 95% intervals, and the differences that are not
  claimable, are in [phase3_report.md](phase3_report.md) (and the Phase 2 baseline, with its
  erratum, in [phase2_report.md](phase2_report.md)). This page does not repeat the numbers.
- **Code-validated citations.** Article ids in answers are checked against the retrieved articles
  by code, and invented ids are dropped and counted.
- **A three-language view.** Every article comes with its German, French and Italian text under one
  id; questions may be asked in any of the three or in English.
- **The benchmark.** The same pipeline is what the benchmark measured, so the numbers describe this
  server's behaviour, within the limits of that benchmark.

What it does not do:

- **It does not cover all of Fedlex.** Only the corpus slice of eight acts (`list_acts` shows
  them: SR 210, 220, 221.214.1, 221.229.1, 235.1, 241, 272, 281.1). No ordinances, no other acts, no
  Official Compilation, no case law, no historical versions. A question outside the slice should be
  refused or answered with pointers; check `status`.
- **It is not an official source.** Only the publication by the Federal Chancellery is
  authoritative. The corpus is a snapshot (each act has its consolidation date) and can be out of
  date.
- **It is not legal advice.** Generated rankings and answers can be wrong; check the cited
  article text yourself.
- **It is not the only Fedlex MCP server.** Others exist. What I could verify, from the README on
  GitHub on the day of writing: [malkreide/fedlex-mcp](https://github.com/malkreide/fedlex-mcp) (MIT,
  community project) queries the Fedlex SPARQL endpoint with 12 tools: keyword search of laws by
  title, lookup by SR number, version history, upcoming changes, Federal Gazette, treaties,
  consultations and TERMDAT terminology. It uses SPARQL and keyword filters, not semantic retrieval,
  and its README reports no retrieval benchmark. It covers far more of the Fedlex catalogue than
  this server does. Two more appear on PyPI, [ch-eli-mcp](https://pypi.org/project/ch-eli-mcp/)
  (described in search results as Fedlex metadata via SPARQL) and
  [swiss-courts-mcp](https://pypi.org/project/swiss-courts-mcp/) (court decisions); I could not
  open their pages and have not checked their features. I have not tested any of them.
  Choose by need: for breadth of the catalogue, use one of those; for measured retrieval and
  validated citations on a small slice, use this one.
