# Phase 3 progress log (fine-tuning, learning mode)

## 2026-10-28, Step 0: housekeeping

- CLAUDE.md status line updated to "Phase 3: fine-tuning (learning mode)".
- Hardware (this machine): AMD Ryzen 5 5500U (6 cores / 12 threads), 15.3 GB RAM, 260 GB free disk, AMD Radeon integrated graphics (512 MB). **No NVIDIA GPU, no CUDA** (`nvidia-smi` not found). Python 3.13.3, uv 0.13.1. torch and sentence-transformers are not installed; the repo runs embeddings through ONNX Runtime only.
- Proposal where to train:
  - Local CPU: data building, hard-negative mining, all evaluation, ONNX export and latency tests (inference of e5-small is fast on this CPU).
  - Training: free **Kaggle** notebook GPU (T4 x2 or P100, about 30 GPU hours per week, no payment details needed) as the main option, **Colab** free T4 as backup, in `notebooks/`. A rough estimate (not measured) puts CPU training of a 118M-parameter model on about 10k+ query/article pairs at several hours, above the 2-hour local limit set in the kickoff, so local CPU training is only for a smoke test of a few steps.
  - No paid compute. A new `finetune` dependency group (torch CPU, sentence-transformers) will be added to `pyproject.toml` when needed, pinned.
- Budget: 10.9 of 25 USD spent. Synthetic-query estimate (to be confirmed before any paid call, Step 2): about 3,000 training articles, one call per article producing about 4 questions, roughly 1,000 input and 400 output tokens per call with a Claude Haiku 5.5 class model ($0.10 / $0.50 per million): about 0.3 + 0.6 = **under 1 USD**, far below the 5 USD confirmation line. Exact figure comes from the real article count after exclusions.
