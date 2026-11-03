# Fedlex Cross-Lingual RAG Benchmark — Project Plan

> Live version: https://claude.ai/code/artifact/23542ae5-4153-4af0-b9ae-b5758fcb0a13

Oct 15, 2026 · @Francisco Mira

## Goal and positioning

Build one finished, public, evaluated project that gets Francisco interviews for forward-deployed / applied AI engineer roles at frontier labs (Anthropic, OpenAI, Google DeepMind), in Zürich or remote.

- **Target role:** forward-deployed / applied AI engineer. Building with models alongside customers, not pre-training research.
- **Location:** Zürich or remote. All three labs have Zürich offices.
- **The gap being closed:** strong private work (the Confluence RAG Assistant) that hiring managers cannot see or verify. This project is its public, reproducible counterpart.
- **Learning goal:** first hands-on fine-tuning experience, done inside the project rather than as a separate course.

## Core question

**Can a small, fine-tuned open-source reranker or embedding model match a frontier LLM reranker on cross-lingual legal retrieval, at a fraction of the cost?**

Swiss federal law (Fedlex) is published officially in German, French and Italian as parallel texts. The same content in several languages makes a controlled experiment: only the language changes.

What the project measures:

- **Cross-lingual retrieval:** the question in one language, the law in another (e.g. an English question over German text).
- **Model comparison:** Claude vs. GPT vs. Gemini on grounded, cited answers.
- **Faithfulness:** whether citations support the claims, and whether the model says "not in the sources" when it should.
- **Cost and latency:** a fine-tuned small model vs. a frontier API call, per query.

What it demonstrates to a hiring manager: eval discipline, model-behaviour understanding, cost trade-off thinking, fine-tuning basics, and honest reporting of results.

## Decisions

| Decision | Why |
| --- | --- |
| Fedlex benchmark is the single flagship | Model-agnostic and eval-centric; a public replay of the private Confluence RAG work |
| Broad AI focus, not insurance-specific | Targeting frontier labs, not only the insurance market |
| Fine-tuning learned inside this project | The reranker was the biggest lever at work (+11 pts fact recall); learning with a real purpose |
| LLM Cost Autopilot parked | Stays on GitHub as shipping evidence; its routing idea may return as a section of the write-up |
| Time budget: about 8–10 h/week | Assumed; adjust the plan if this changes |

**Done bar for every phase:** public repo, results reproducible with one command, numbers reported even when unflattering.

## Plan

Six phases over about 14 weeks at 8–10 h/week; each phase ends with something public.

| Phase | Weeks | Output | State |
| --- | --- | --- | --- |
| 0 · Scope and data | 1–2 | Corpus slice in DE/FR/IT, articles aligned and chunked, public repo created | done |
| 1 · Benchmark | 3–4 | ~150 hand-checked questions with gold answer + source article, not-in-corpus traps | done (frozen v1, review coverage lower than planned: see phase 1 report) |
| 2 · Baseline harness | 5–6 | Hybrid retrieval + LLM rerank; Claude, GPT, Gemini compared on recall, faithfulness, cost | done (plus a translation control and an ADK agent; see phase 2 report) |
| 3 · Fine-tuning | 7–10 | Fine-tuned multilingual embedding model (primary) and, if time allows, a small reranker; base vs. tuned vs. LLM query translation; quality, latency and cost per query | done (retrieval only; see `docs/phase3_report.md`) |
| 4 · Ship | 11–12 | Live demo + MCP server, results page, one-command reproduction | done: measured, results page live, MCP server in the repo, no live demo (see `docs/phase4_progress.md`) |
| 5 · Write-up and apply | 13–14 | Long-form post + LinkedIn, CV updated, applications start | planned |

Phase 3 is the core experiment and the fine-tuning learning block, so it gets the most time (4 weeks).

## Progress and what it changes (updated 2026-10-18, after phase 2)

Phases 0 to 2 are done; each has a report in `docs/`. Findings that shape the rest of the plan:

- **The benchmark has a ceiling for strong rerankers.** Questions were drafted from the article text, so once the gold article is among the candidates, small and large LLM rerankers all put it first (dev: Haiku, Gemini Flash-Lite, Sonnet identical). Phase 3 compares a tuned small reranker with an LLM reranker on this ceiling; expect the two to be hard to tell apart on answerable questions, and report that honestly. Consider adding harder evaluation material (more candidates per query, or hard negatives mined from the neighbouring articles) rather than only more of the same questions.
- **First-stage recall is the bottleneck**, mostly for English questions (14% recall@10 for bare hybrid retrieval, 71% after LLM query translation). A fine-tuned embedding model for the first stage may matter more than the reranker; decide in phase 3 which of the two to tune, or both, using the dev numbers.
- **Train/eval separation:** exclude `benchmark/v1/dev_gold_articles.txt`, `test_gold_articles.txt` and their neighbouring articles from all fine-tuning pairs.
- **Costs observed:** about 0.002 to 0.003 USD per question for a Haiku-based pipeline; judging costs about twice the system itself. Phase 2 spent about 11 USD of the 25 USD monthly cap.
- **Open from phase 2:** one non-expert, English-only labeller calibrated the judge (81% agreement); no Swiss-law expert has reviewed any benchmark item. An expert or student spot check remains the largest unaddressed validity risk.

## Phase 3 result (2026-10-28)

The fine-tuned embedding model (multilingual-e5-small, 11.5k synthetic questions, 3 epochs on a free GPU, 2.16 USD) reaches test recall@10 **0.81 [0.71, 0.90]** with no LLM call (0.1 s, 0 USD), against 0.55 for the untuned hybrid (+0.26 [+0.17, +0.35]); English questions 0.24 to 0.80. The LLM-translation pipeline reaches 0.87 [0.79, 0.94] at 8 s and 0.0024 USD (difference to the tuned model not claimable on recall@10, clearly better at ordering: MRR 0.84 against 0.63). Tuned embedder plus the Haiku reranker: 0.89 [0.80, 0.97], not distinguishable from the translation pipeline, five times faster. Retrieval only: answer quality was not re-measured. Hard negatives did not help; a quarter of the data gives most of the gain. A padding bug in the Phase 2 embedder was found and corrected (errata in the Phase 2 report: Phase 2 retrieval numbers were too low). Open for later: answer-level reruns with the corrected first stage, the int8 model on test, a Swiss-law expert review, and a visual follow-up project.

## Phase 4 result (2026-11-03)

The served pipeline (tuned int8 embedder on CPU, Haiku reranker over the top 60 of 150, code-validated cited answer) reaches test recall@10 **0.86 [0.78, 0.94]**, fact recall **0.80 [0.71, 0.89]** and faithfulness 0.84 at 0.0024 USD and 5.5 s per question; the LLM-translation pipeline with the same answer step reaches 0.87 and 0.82 at 0.0030 USD and 11.9 s: not distinguishable on quality, the served one cheaper and about twice as fast. The int8 file loses 0.055 MRR [0.018, 0.097] against fp32 on test (claimable) and 0.02 recall@10 (not claimable); decided with Francisco to keep it. Fact recall was above the range written beforehand for both. Spend for Phase 4 so far: about 4 USD (month 4.16 of 25). The results page and ten recorded answers are live at https://franciscomira.github.io/fedlex-rag-benchmark/; the MCP server and a retrieval-only demo run locally. There is no live demo: Hugging Face Docker Spaces need a paid plan (402), and Francisco chose not to pay or look for another host. Details: `docs/phase4_progress.md`.

## Phase 3 decision: what we fine-tune (2026-10-28)

Phase 2 showed that LLM rerankers are indistinguishable on this benchmark and that first-stage cross-lingual retrieval is the bottleneck (bare hybrid recall@10 for English questions 14%; 71% after LLM query translation at about 18 s per question). Decision, agreed with Francisco:

- **Primary: fine-tune the bi-encoder** (multilingual-e5-small, the embedding model of system A) on synthetic EN/DE/FR/IT questions paired with their articles. **Updated core question for this phase:** can a fine-tuned small embedding model close the gap that LLM query translation closes, with no LLM call at query time (about 0.1 s and 0 USD per query instead of about 18 s and 0.003 USD)? Compared against system C on recall@10 with paired bootstrap intervals, and also against C's retrieval before its reranker, so the comparison does not mix two effects.
- **Secondary, only if time allows:** fine-tune a small multilingual cross-encoder reranker over a wider candidate window and compare it with the Haiku reranker on quality, cost and latency (the original question; Phase 2 suggests a tie on quality).
- **Compute:** no NVIDIA GPU on the dev machine; training on a free Kaggle or Colab GPU (notebooks in `notebooks/`), everything else on local CPU. No paid compute.
- **Stop rules** unchanged from the kickoff: any overlap with dev/test gold, neighbours or twins; tuned model worse than base on dev; validation loss rising while training loss falls; a result that looks too good (English recall@10 above 90% without an LLM).

## Phase 0: access check and tasks

**Data access is solved; the risk is positioning.** Fedlex is open and machine-readable, ready-made aligned datasets exist, and serious prior work already covers parts of this space.

### What the check found

- **Official access:** Fedlex has a public SPARQL endpoint at fedlex.data.admin.ch for metadata ([Fedlex data guide](https://agriculture.ld.admin.ch/data-usage/fedlex)). Acts are available as PDF, HTML, DOCX and Akoma Ntoso XML (since 2022), and no API key is needed ([fedlex-mcp](https://mcpservers.org/pt-BR/servers/loicvuilliomenet-boop/fedlex-mcp)).
- **Shortcut dataset:** the Swiss Case Law Open Dataset mirrors 5,519 federal laws, about 133k articles in each of DE/FR/IT, as a CC0 Parquet download on Hugging Face ([repo](https://glama.ai/mcp/servers/jonashertner/caselaw-repo-1)). This may replace most of our own scraping.
- **Aligned translations:** SwiLTra-Bench offers 180k+ aligned Swiss legal translation pairs, including English ([paper](https://arxiv.org/abs/2503.01372v1)). This is a candidate source for fine-tuning pairs.
- **Prior work to cite, not repeat:**
  - The SCALE benchmark ([One Law, Many Languages](https://arxiv.org/abs/2306.09237)) already includes cross-lingual retrieval of laws from court-case facts.
  - ZHAW's JuRAG tested 12 multilingual embedding models on court decisions with 26 questions in four languages. The embedding choice was the biggest lever ([ZHAW blog](https://blog.zhaw.ch/artificial-intelligence/2026/06/02/can-ai-bridge-switzerlands-legal-language-gap/)).
- **Fedlex MCP servers already exist** (several). An MCP server is table stakes here, not a differentiator.

### Our angle, sharpened

No prior work found covers our exact combination: **article-level, cited answers over statutes, scored for citation faithfulness and refusal; frontier LLMs compared head to head; a fine-tuned small reranker vs. an LLM reranker on quality and cost.** The write-up positions against SCALE and JuRAG explicitly.

### Tasks

- [x] Download the Hugging Face law dataset; check that articles align one-to-one across DE/FR/IT for 3 sample acts (checked: that dataset holds court decisions only, so the corpus comes from our own Akoma Ntoso ingest)
- [x] Pick the corpus slice: Code of Obligations plus 3–5 related acts (target a few thousand articles)
- [x] Confirm licensing for each source (Fedlex terms, CC0, SwiLTra licence) and note it in the README
- [x] Read the SCALE IR section and the JuRAG post; list what they measured and what they did not
- [x] Create the public GitHub repo with README skeleton, licence and project structure
- [x] Write the chunking script: one chunk per article, with SR number and article id as metadata

## Open questions and risks

| Risk or question | Mitigation |
| --- | --- |
| Seen as a repeat of SCALE or JuRAG | Cite both; lead with what is new (statutes, citation faithfulness, LLM vs. fine-tuned reranker, cost) |
| No legal expert to vet questions | Questions answerable from a single article; check gold answers against the article text; ask a Swiss law student to spot-check a sample |
| Fine-tuning compute on modest hardware | Small cross-encoder or embedding model; free GPU tiers (Colab, Kaggle) or a short rented GPU run |
| API costs across three providers | Run the full benchmark only at phase ends; cache every response; set a monthly cap |
| Scope creep into a product | The benchmark and write-up are the deliverable; the demo stays small |

- [ ] Confirm weekly hours (8–10 assumed)
- [x] Set a monthly API and hosting budget (25 USD per month, enforced in `config.toml`)
- [ ] Decide on a target date to start applying
