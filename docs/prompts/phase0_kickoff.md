We're starting Phase 0 of this project. Read CLAUDE.md and docs/PROJECT_PLAN.md first.

Goal of this session: prove the data foundation is solid before we build anything else.

0. Copy docs/agents/data-engineer.md and docs/agents/eval-reviewer.md into .claude/agents/ so they become project subagents.
1. Set up the project: uv, pyproject.toml, pytest, ruff. Initialise git with a first commit. Do not create a remote or push.
2. Find the best source for Swiss federal law in DE/FR/IT at article level. First candidate: the Swiss Case Law Open Dataset on Hugging Face (federal laws mirrored per article in DE/FR/IT, CC0). Fallback: the Fedlex SPARQL endpoint (https://fedlex.data.admin.ch/sparqlendpoint) plus the filestore (Akoma Ntoso XML or HTML). Confirm the licence of whatever you use and note it in the README.
3. Alignment check on three acts: Code of Obligations (SR 220), Civil Code (SR 210), Data Protection Act (SR 235.1). For each act and language, count articles, and report how many article ids exist in all three languages, which are missing, and why (repealed, numbering differences, parsing issues). Spot-check 5 random aligned articles by printing DE/FR/IT side by side.
4. Write src/fedlex_bench/ingest.py and src/fedlex_bench/chunk.py: one chunk per article with metadata {sr, article_id, lang, title, text, source_url, consolidation_date}. Output to data/processed/articles.parquet. Add tests.
5. Write docs/phase0_report.md: the data source and licence, the counts table, alignment rate, known issues, and a go / adjust recommendation.

How to work:
- Use the data-engineer subagent for steps 2 to 4. When the code is done, run the eval-reviewer subagent on it and fix what it finds.
- Plan first and show me the plan before writing code.
- Stop and ask me if the alignment rate is below 95% or a source turns out to be unusable.
