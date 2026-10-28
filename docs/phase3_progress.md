# Phase 3 progress log (fine-tuning, learning mode)

## 2026-10-28, Step 0: housekeeping

- CLAUDE.md status line updated to "Phase 3: fine-tuning (learning mode)".
- Hardware (this machine): AMD Ryzen 5 5500U (6 cores / 12 threads), 15.3 GB RAM, 260 GB free disk, AMD Radeon integrated graphics (512 MB). **No NVIDIA GPU, no CUDA** (`nvidia-smi` not found). Python 3.13.3, uv 0.13.1. torch and sentence-transformers are not installed; the repo runs embeddings through ONNX Runtime only.
- Proposal where to train:
  - Local CPU: data building, hard-negative mining, all evaluation, ONNX export and latency tests (inference of e5-small is fast on this CPU).
  - Training: free **Kaggle** notebook GPU (T4 x2 or P100, about 30 GPU hours per week, no payment details needed) as the main option, **Colab** free T4 as backup, in `notebooks/`. A rough estimate (not measured) puts CPU training of a 118M-parameter model on about 10k+ query/article pairs at several hours, above the 2-hour local limit set in the kickoff, so local CPU training is only for a smoke test of a few steps.
  - No paid compute. A new `finetune` dependency group (torch CPU, sentence-transformers) will be added to `pyproject.toml` when needed, pinned.
- Budget: 10.9 of 25 USD spent. Synthetic-query estimate (to be confirmed before any paid call, Step 2): about 3,000 training articles, one call per article producing about 4 questions, roughly 1,000 input and 400 output tokens per call with a Claude Haiku 5.5 class model ($0.10 / $0.50 per million): about 0.3 + 0.6 = **under 1 USD**, far below the 5 USD confirmation line. Exact figure comes from the real article count after exclusions.

## 2026-10-28, Step 2: training pool and synthetic queries

- Exclusions (`src/fedlex_bench/finetune/exclusions.py`, test `tests/test_finetune_leakage.py`): 3,663 retrieval-eligible article ids; removed 172 dev/test gold, 288 neighbours (previous and next eligible article of the same act), 3 twins of gold at character 5-gram Jaccard >= 0.5 (`220:257c`, `220:808`, `281.1:108`; the benchmark itself used 0.7 and found none). Training pool: **3,201 articles**. The test fails on any overlap; a poisoned pool made the gold and twin checks fail.
- Synthetic queries (`src/fedlex_bench/finetune/synth.py`, model `claude-haiku-5-5`, prompt hash in the run, seed 20261028): 4 questions per article, 2 English plus two of DE/FR/IT; the article is shown in the language left out. Six styles, four distinct per article, a fixed role for `third_party`.
- Pilots: 2 x 20 articles (0.005 USD each). The first showed a repeated "my business partner" persona and invented details; fixed with per-slot roles and a no-new-facts rule before the full run.
- Full run: 3,201 articles, **12,800 questions** from 3,200 articles (EN 6,400, DE 2,130, FR 2,162, IT 2,108); 1 article failed validation twice (`272:163`, id leak) and is left out; 22 articles needed the retry. Styles 2,082 to 2,163 each. Mean length in words: keywords 7.5, short_question 18, confused 28, third_party 31, lay_situation 32, legal_wording 32. Live cost **0.87 USD** (estimate was 1.45); month spend about 11.8 of 25 USD.
- Known noise (not measured yet): a few questions garble a legal term, and some questions on generic provisions fit a neighbouring article too. Planned handling: the hard-negative false-negative filter, and a manual read of 20 mined examples before training.
