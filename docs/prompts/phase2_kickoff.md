Phase 1 is done: benchmark v1 is frozen (docs/phase1_report.md). We are starting Phase 2: the baseline harness, the provider comparison and the agent. Read CLAUDE.md, docs/PROJECT_PLAN.md (including the "Target role" section), docs/benchmark_spec.md and docs/phase1_report.md first.

Ground rules for the whole phase:
- benchmark/v1 is read-only. Iterate on dev only. test.jsonl is used once per frozen configuration, at the end, and every test run is logged.
- Every reported number is reproducible with one command from a cached run.
- Ask me before spending money beyond the budget in Step 0.

## Step 0: housekeeping (you, before anything else)
1. Copy docs/agents/retrieval-engineer.md, llm-integrator.md and agent-engineer.md into .claude/agents/.
2. Publish the repo: check `gh auth status`; if logged in, propose creating a public GitHub repo "fedlex-rag-benchmark", show me exactly what will be pushed (no raw data, no articles.parquet, no .env, no English article renderings), and push only after I say yes. Push the benchmark-v1 tag too.
3. Fix the date in docs/phase1_report.md (it says 2026-10-17; the freeze was 2026-10-16).
4. Ask me: monthly API budget, and which keys exist in .env (Anthropic, OpenAI, Google AI Studio). Do not call a paid API before I answer. Record the budget cap in config.
5. Update the CLAUDE.md status line to "Phase 2: harness, providers, agent".

## Step 1: docs/eval_spec.md (show it to me before writing harness code)
Decide and write down:
- Retrieval metrics: recall@1/5/10 and MRR on article ids (a hit if any language version is retrieved); reported per query language and per index configuration.
- Answer metrics: fact recall (pinned LLM judge against gold_facts), citation precision and recall (by code), faithfulness (judge: every claim supported by a cited article), trap accuracy, over-refusal rate on answerable questions.
- The trap citation rule (open from Phase 1): a trap answer is correct if it refuses and does not present any article as answering the question; citing a pointer article as "where to look" is allowed. Write the exact rule and test cases.
- Judge: which pinned model, the prompt, and a calibration set: I will hand-label about 30 dev answers so we can measure judge agreement before trusting it. Use a judge from a different model family than the answerer where possible, and report self-preference checks.
- Cost and latency: tokens, cost per query (pinned price table with date), p50/p95 latency.
- Statistics: bootstrap 95% confidence intervals and paired comparisons; say which differences are too small to claim with 150 questions.
- Run layout: eval/results/<run_id>/ with config.json (dated model ids, git sha, benchmark sha), per_item.jsonl, metrics.json.

## Step 2: build in parallel with an agent team
Agent teams are enabled (CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1). Spawn three teammates with the role files, one owner per directory:
- retrieval-engineer: BM25, a small multilingual embedding model, hybrid fusion (RRF), index configurations (same-language, single-language cross-lingual, all languages), and the chunking ablation (article vs. paragraph vs. fixed windows) scored on recall, and later faithfulness, latency and cost.
- llm-integrator: provider clients for Claude, GPT and Gemini with caching, cost and budget enforcement; the LLM reranker; cited answer generation with code-validated citations and an explicit refusal path.
- agent-engineer: the Google ADK system (planner, per-language retriever sub-agents, citation checker that can trigger another search), trace logging and the failure-classification script.
You (the lead) own src/fedlex_bench/eval/: the harness, metrics, judge, bootstrap CIs and the results table. Define the shared interfaces first and have teammates build against them. Use test-driven development. Wait for teammates to finish before integrating.

## Step 3: cross-family check of the benchmark itself
Run a judge from a different family (GPT or Gemini) over all 242 v1 items with the same rubric as the Phase 1 judge. List disagreements with reasons. I will review them in the review app. If any item must change, create benchmark/v2 with a changelog; never edit v1. Do this before any test-split run.

## Step 4: experiments on dev
Run and compare, with CIs:
1. Retrieval only: BM25 vs. embeddings vs. hybrid, each with no reranker and with the LLM reranker, across index configurations.
2. Single-shot answering (best retrieval config) with Claude, GPT and Gemini (the same retrieved context for all three).
3. The ADK agent vs. single-shot, on quality, steps, cost and latency; tool-selection accuracy; failure taxonomy from traces.
4. Chunking ablation.
Keep a short experiment log in docs/phase2_progress.md: what was tried, the result, what was kept and why.

## Step 5: freeze and test once
Freeze up to 4 configurations (written down before seeing test numbers), run them once on test, and write docs/phase2_report.md: results tables with CIs, cost and latency, the agent findings and failure taxonomy, judge calibration, what did not work, and limitations. Run the eval-reviewer on the harness and the report before calling the phase done.

## Stop and ask me if
- projected spend for a run exceeds 20% of the monthly budget;
- the judge agrees with my labels on fewer than 80% of the calibration set;
- the cross-family check disagrees on more than 10% of benchmark items;
- any result looks too good to be true (e.g. recall@10 above 95% on cross-lingual questions).
