Phase 0 is done (see docs/phase0_report.md: GO, 8 acts, 3,844 in-force articles per language, 100% id alignment). We are starting Phase 1: the benchmark. Read CLAUDE.md, docs/PROJECT_PLAN.md, docs/phase0_report.md and docs/related_work.md first.

## Step 0: housekeeping
1. Copy docs/agents/question-writer.md and docs/agents/question-critic.md into .claude/agents/.
2. Check git status; commit any uncommitted Phase 0 work. Tell me whether a GitHub remote exists. Do not create or push to one without asking me.
3. Update the status line in CLAUDE.md to "Phase 1: benchmark".

## Step 1: close the Phase 0 loose ends (use the data-engineer subagent, then the eval-reviewer)
- Corpus decision: keep all 8 acts, with the Code of Obligations (SR 220) as the core. Record this in docs/phase0_report.md.
- Implement range-id expansion (e.g. "210:135-149" resolves for "210:140") with tests. Citation checking in Phase 2 needs it.
- Check the two unverified structural mismatches (221.229.1:95c, 281.1:219) against the raw XML and record the verdict.
- Check that 5 sample source_url deep links resolve on fedlex.admin.ch; record the result.
- Add a helper that returns the retrieval-eligible articles (in force, not repealed, not historical, main text).
- Run the eval-reviewer on the fixed Phase 0 code (it never had a second independent review). Fix what it finds.

## Step 2: write docs/benchmark_spec.md and show it to me before generating anything
Propose concrete answers for:
- Size: about 150 questions, plus about 20 "not in corpus" traps where the right answer is a refusal.
- Mix: about 60% from SR 220, the rest spread across the other 7 acts; a share of questions whose answer needs one cross-reference.
- Query languages: EN, DE, FR, IT, balanced. Corpus configurations to evaluate later: same-language index, cross-lingual (e.g. EN question against DE-only index), all languages.
- Schema per question: id, query_lang, question, gold_article_ids, gold_facts, is_trap, difficulty, lexical_overlap, source_act, author, review_status.
- How lexical_overlap is computed (e.g. BM25 rank of the gold article for the question).
- Splits: dev (about 40) and test (about 110). The test split is locked once frozen. Also list the gold articles of the test split, so Phase 3 fine-tuning data can exclude them (no leakage).
- Versioning: benchmark/v1/*.jsonl plus a checksum file.

## Step 3: generate and stress-test (after I approve the spec)
- Spawn question-writer subagents in parallel, one per act group, each writing its own file under benchmark/drafts/. Over-generate by about 50% so weak questions can be dropped.
- Run question-critic subagents on every draft. Apply its keep / fix / drop verdicts, then re-run the critic on the fixed questions.
- Translate questions only where the spec asks for it, and have the critic check that translations keep the meaning.

## Step 4: human review tool
Build a small local review app (Streamlit) that shows each question with its gold article in DE/FR/IT side by side and lets me accept, edit or reject it, writing review_status back to the JSONL. I will hand-check every question; nothing is frozen until I have.

## Step 5: freeze and report
- After my review, write benchmark/v1/ (dev.jsonl, test.jsonl, test_gold_articles.txt, SHA256SUMS) and docs/phase1_report.md: counts by act, language, difficulty and split, overlap distribution, drop reasons from the critic, and known limitations.
- Add a baseline sanity check: BM25 recall@10 per query language on the dev split, so we know the benchmark is neither trivial nor impossible. Report it; do not tune anything on the test split.

## How to work
- Plan first and show me the plan. Stop and ask before anything irreversible.
- Use the superpowers skills for planning, test-driven development and code review.
- One owner per file; subagents never write to the same file.
- Stop and ask me if the critic drops more than 40% of the drafts, or if BM25 recall@10 on dev is above 90% (too easy) or below 20% (too hard).
