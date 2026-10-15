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

| Phase | Weeks | Output |
| --- | --- | --- |
| 0 · Scope and data | 1–2 | Corpus slice in DE/FR/IT, articles aligned and chunked, public repo created |
| 1 · Benchmark | 3–4 | ~150 hand-checked questions with gold answer + source article, not-in-corpus traps |
| 2 · Baseline harness | 5–6 | Hybrid retrieval + LLM rerank; Claude, GPT, Gemini compared on recall, faithfulness, cost |
| 3 · Fine-tuning | 7–10 | Small multilingual reranker; base vs. tuned vs. LLM; quality vs. cost per query |
| 4 · Ship | 11–12 | Live demo + MCP server, results page, one-command reproduction |
| 5 · Write-up and apply | 13–14 | Long-form post + LinkedIn, CV updated, applications start |

Phase 3 is the core experiment and the fine-tuning learning block, so it gets the most time (4 weeks).

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

- [ ] Download the Hugging Face law dataset; check that articles align one-to-one across DE/FR/IT for 3 sample acts
- [ ] Pick the corpus slice: Code of Obligations plus 3–5 related acts (target a few thousand articles)
- [ ] Confirm licensing for each source (Fedlex terms, CC0, SwiLTra licence) and note it in the README
- [ ] Read the SCALE IR section and the JuRAG post; list what they measured and what they did not
- [ ] Create the public GitHub repo with README skeleton, licence and project structure
- [ ] Write the chunking script: one chunk per article, with SR number and article id as metadata

## Open questions and risks

| Risk or question | Mitigation |
| --- | --- |
| Seen as a repeat of SCALE or JuRAG | Cite both; lead with what is new (statutes, citation faithfulness, LLM vs. fine-tuned reranker, cost) |
| No legal expert to vet questions | Questions answerable from a single article; check gold answers against the article text; ask a Swiss law student to spot-check a sample |
| Fine-tuning compute on modest hardware | Small cross-encoder or embedding model; free GPU tiers (Colab, Kaggle) or a short rented GPU run |
| API costs across three providers | Run the full benchmark only at phase ends; cache every response; set a monthly cap |
| Scope creep into a product | The benchmark and write-up are the deliverable; the demo stays small |

- [ ] Confirm weekly hours (8–10 assumed)
- [ ] Set a monthly API and hosting budget
- [ ] Decide on a target date to start applying
