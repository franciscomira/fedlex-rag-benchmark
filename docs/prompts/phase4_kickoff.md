Phase 3 is done and published. We are starting Phase 4: ship. Read CLAUDE.md, docs/PROJECT_PLAN.md (the Phase 3 result section), docs/phase3_report.md, docs/phase3_progress.md (the review and publication entries at the end), docs/phase2_report.md (including its errata) and docs/eval_spec.md first.

## Where we are
- Repo: github.com/franciscomira/fedlex-rag-benchmark (public, main pushed). Model: huggingface.co/franciscomira/fedlex-e5-small (public; fp32 safetensors and `onnx/model_int8.onnx`). Local checkpoint: models/r1_inbatch (git-ignored).
- Phase 3 result on test (retrieval only): tuned embedder alone recall@10 0.81, 0.1 s, 0 USD; tuned embedder plus Haiku reranker (T2) 0.89, recall@1 0.79, 1.6 s, 0.0016 USD per question; LLM translation pipeline (C') 0.87, 8 s, 0.0024 USD. T2 and C' are not distinguishable on recall. No answer-level measurement was made with the corrected first stage.
- Known open items: the int8 model was checked on dev only (recall@10 0.78 against 0.79 for fp32); Phase 2's fact recall, faithfulness and trap numbers were measured with the buggy embedder and not repeated; system D (ADK agent) was not rerun; the training questions have not been reviewed by anyone who reads German, French or Italian; no Swiss-law expert has reviewed anything.
- Budget: 25 USD per month in config.toml, about 13 USD spent in October (the ledger resets by calendar month); confirm anything above 5 USD with me before running it.

## How we work
- I read English only (no German, French or Italian), and I want less hand-holding than in Phase 3: work mostly autonomously, show me decisions in plain English with numbers, and ask only for decisions that are mine (scope, anything that costs money or publishes something). Learning mode is off; the Learning output style no longer applies. Keep explanations short.
- Use a plan first (a short one: scope, what runs where, cost, risks) and wait for my OK before building. Use subagents for independent work or reviews, one owner per file; reviewer agents never edit code.
- Every number that goes into the results page must come from the eval harness with a 95% interval, reproducible with one command. Anything frozen for test is run once; write hypotheses before running; never tune on test.
- Ask before pushing, before creating any account, project or paid resource, and before anything that sends data to an external service. Secrets live in .env only. Never put a key in a notebook, a log or a commit.

## Scope (from the plan, to be confirmed in your plan)
1. **Serving pipeline.** The tuned int8 ONNX embedder (CPU), the Haiku reranker over the top candidates, and a cited answer where article ids and citations are validated by code, with an explicit "not in the sources" path. Decide the exact configuration with me; measure it with the harness, including the int8 model as its own frozen configuration on test, and answer-level metrics (fact recall, faithfulness, citation precision, refusals) once with the corrected first stage and the judge gpt-6.1-sol. Show the cost of that run before starting it.
2. **Live demo.** A small web app: question in English, German, French or Italian; ranked articles with their text in the three languages; the cited answer. Hosting: prefer a free or near-free option; do not sign me up for paid compute, and tell me before anything needs a billing account (for example Google Cloud Run). A public demo that calls a paid LLM needs hard protections: per-IP and global daily caps, a monthly spend cap enforced in code, no storing of user questions, and a kill switch. If that is too risky for a public page, propose an alternative (for example a retrieval-only public demo plus a recorded example of the full pipeline).
3. **MCP server.** Tools such as search articles, get article (all three languages), and answer with citations. Several Fedlex MCP servers already exist, so say in the README what this one adds (evaluated retrieval, code-validated citations, the benchmark) and do not overclaim.
4. **Results page and one-command reproduction.** A static page (GitHub Pages or similar) with the Phase 2 and Phase 3 results, the errata, the limitations in plain words, and the commands to reproduce each number. `uv run` commands only; say which steps need keys or a GPU.
5. **Required notices everywhere the texts are shown:** "This is not an official publication. Only the publication by the Federal Chancellery is authoritative." and "Not legal advice."

## Stop and ask me if
- a step would cost more than 5 USD, or the month's total would pass 20 USD;
- the int8 model or the full pipeline is clearly worse than its dev numbers on the frozen test run;
- hosting needs a credit card, a new account, or a public endpoint that spends money;
- you find another bug in an earlier phase, as you would for the embedder: report it with its scope before fixing quietly;
- anything I would have to publish or accept (terms, licences) needs a decision.

Start by reading the files above, then write the short plan.
