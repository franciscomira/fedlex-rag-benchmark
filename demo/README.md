# Live demo

A small web page where you type a question (English, German, French or Italian) and get the
most relevant articles of eight Swiss federal acts, with the German, French and Italian texts side
by side. It is retrieval only: it calls the tuned embedding model on the server's CPU and never
calls a language model, so it needs no API key and costs nothing per visit.

Below the search box the page shows "Recorded full pipeline": ten example questions with the
answers and citations the full pipeline (including the LLM reranker and answer step) produced
once, offline. These come from `demo/examples.json`.

## Status

Written and unit-tested (`uv run pytest tests/serve`). **The Dockerfile has not been built or run**:
the machine it was written on has no Docker. Treat the first build as a test.

## Files

| File | What it is |
| --- | --- |
| `src/fedlex_bench/serve/web.py` | The Starlette app and its command line. |
| `src/fedlex_bench/serve/static/` | The page (`index.html`, `app.css`, `app.js`). No external fonts, scripts or trackers. |
| `src/fedlex_bench/serve/record_examples.py` | Records `examples.json` with the full pipeline. |
| `demo/Dockerfile` | Image for a Hugging Face Docker Space. |
| `demo/examples.json` | The recorded examples. Ships as the placeholder `{"examples": []}`. |

## Run it locally (no Docker)

Needs the corpus (`uv run python -m fedlex_bench.chunk`) and the model; the model is downloaded
from the Hub on first start if `models/r1_inbatch/onnx` is missing.

```
uv sync --group serve
uv run python -m fedlex_bench.serve.web          # http://localhost:7860
```

The first start embeds the corpus and can take a while; later starts reuse
`data/processed/index/`. `uv run python -m fedlex_bench.serve.web --warm` only does that
preparation and exits.

## Record the examples (costs money, run it yourself)

```
uv run python -m fedlex_bench.serve.record_examples          # dry run, shows the questions
uv run python -m fedlex_bench.serve.record_examples --yes    # needs ANTHROPIC_API_KEY in .env
```

About 0.002 USD per question, about 0.02 USD in total. It overwrites `demo/examples.json` only if all
ten questions succeed. Read the file before you publish it: the page shows what the pipeline said,
including refusals that were meant to be answers and the other way round. Two questions (value
added tax, naturalisation) are out of scope on purpose and should be refused.

## Deploy as a Hugging Face Docker Space (manual steps)

Nothing here has been done yet, and nothing in the repository creates an account, a Space or a remote.

1. Create a Space on huggingface.co: SDK **Docker**, hardware **CPU basic** (free).
2. Clone the Space repository. Copy into it the files tracked by this project (not `data/raw`,
   `data/processed` or `models/`, which git already ignores): for example
   `git archive HEAD | tar -x -C <space-dir>` from the project root.
3. In the Space repository, copy `demo/Dockerfile` to `Dockerfile` at the root (Spaces build the
   root Dockerfile), and make the root `README.md` start with this header, keeping the
   project text below it:

   ```
   ---
   title: Fedlex cross-lingual search
   sdk: docker
   app_port: 7860
   pinned: false
   ---
   ```
4. Commit and push to the Space. The Space builds the image; follow the build log.
5. Optional Space variables: `DEMO_RATE_PER_IP`, `DEMO_RATE_GLOBAL`, `TRUST_PROXY=1` (see below).
   To switch the demo off at once, set `DEMO_DISABLED=1`: every `/api/*` route then answers 503.
6. Open the Space URL, run one search in each language, and check that the notices are visible.

To test the image locally first: `docker build -f demo/Dockerfile -t fedlex-demo .` and
`docker run --rm -p 7860:7860 fedlex-demo`.

## Settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `PORT` | 7860 | Port to listen on. |
| `DEMO_RATE_PER_IP` | 30 | Requests per minute per client address. |
| `DEMO_RATE_GLOBAL` | 600 | Requests per minute for the whole service. |
| `DEMO_MAX_BODY_BYTES` | 8192 | Largest accepted JSON body. |
| `DEMO_MAX_CONCURRENT` | 2 | Searches running at once. |
| `DEMO_DISABLED` | off | `1` returns 503 on `/api/*`. |
| `TRUST_PROXY` | off | `1` counts rate limits by the last `X-Forwarded-For` entry (the one the proxy added). Turn on only behind a proxy that sets it; otherwise everyone looks like one address. |
| `DEMO_EXAMPLES` | `demo/examples.json` | Another examples file. |

Rate limits are kept in memory per process: they reset on restart and are not shared between
replicas. Questions are never logged or stored; uvicorn access logging is off.

## Risks and things to know

- **Fedlex must be reachable at build time.** The build downloads the statute XML (the exact files
  pinned in `data/manifest.json`, checked against their hashes). If Fedlex is down or changes a URL,
  the build fails. Retry later. A build that succeeds is frozen: the Space does not refresh the
  law by itself, and the page shows the consolidation date of each article.
- **Build time and size.** Embedding the corpus on a free CPU build machine takes a long time
  (the Phase 3 note for the full-precision model is about 15 minutes; the int8 model should be
  faster, but this is unmeasured). If the build hits the platform time limit, build the index
  elsewhere and add `data/processed/index/` to the image instead.
- **The image is bigger than it needs to be.** The project's main dependency list also holds
  the LLM SDKs and the agent framework, which the demo never imports; `uv sync --group serve` installs
  them anyway. Torch is not installed. Moving those SDKs into their own group would shrink the
  image; that needs a change to `pyproject.toml`, which this work did not touch.
- **Free-tier sleep and cold start.** A free Space goes to sleep after some idle time and
  restarts on the next visit. The restart loads the model and the cached vectors from disk, which takes
  tens of seconds. The first visitor after a sleep waits for that. Free CPU Spaces also have little
  memory; if the app is killed at start-up, check the memory in the logs.
- **Offline at run time.** The image sets `HF_HUB_OFFLINE=1` after the build downloaded the pinned model
  revision. If the model lookup fails at run time, remove that line and rebuild.
- **Abuse.** Limits are modest and in memory. Behind the Space proxy all visitors may appear as one
  address unless `TRUST_PROXY=1` is set; with it off, the global limit still protects the service
  but one heavy user can use it up for everyone.
- **Not legal advice, not official.** The page shows these notices from `serve/notices.py`.
  Keep them if you change the page.
