# notebooks/

## `finetune_kaggle.ipynb`: fine-tuning on a free Kaggle GPU

The dev machine has no NVIDIA GPU, and a CPU step with batch 64 takes over 30 seconds (about 10 GB of RAM), so training runs on a free Kaggle GPU. Everything the notebook needs is one 3.4 MB zip.

### Set-up (once)

1. Build the kit (it holds the corpus, so it is git-ignored):
   ```
   uv run python notebooks/make_kaggle_kit.py
   ```
   This writes `notebooks/kit/fedlex_finetune_kit.zip` (code, the training pairs and hard negatives, the corpus, the leakage tests, `KIT_INFO.json` with the git sha and file hashes).
2. Create a free account at kaggle.com and verify your phone number (needed for GPU and internet access).
3. *Datasets* > *New Dataset* > upload `fedlex_finetune_kit.zip`, title `fedlex-finetune-kit`, **visibility Private**. Kaggle unpacks the zip itself.
4. *Code* > *New Notebook* > *File* > *Import Notebook* > upload `notebooks/finetune_kaggle.ipynb`.
5. In the notebook's right panel:
   - *Input* > *Add Input* > your private `fedlex-finetune-kit` dataset.
   - *Settings* > *Accelerator*: **GPU T4 x2** (or P100).
   - *Settings* > *Internet*: **On** (the model weights, 470 MB, download from Hugging Face).
6. In the second cell set `LEARNING_RATE` and `EPOCHS`. These are your decisions; write your prediction in `docs/learning/journal.md` first.

### Run

*Run All* (or run cell by cell). Order: settings, unpack the kit, install and GPU check, **tests gate** (leakage tests must pass), a 20-step speed check, run 1 (in-batch negatives), run 2 (plus hard negatives), learning-curve plot, data-size ablation (25% and 50%), packaging. On a T4 each run should take minutes; if the speed check shows more than about 2 s/step, tell me before running the rest. Kaggle sessions stop after about 9 hours and keep nothing between sessions, so download the output when done.

### Bring the results back

From the notebook's *Output* tab download `/kaggle/working/out` (per run: `log.jsonl`, `config.json`, `<run>_best.zip` with the best checkpoint of about 470 MB; plus `learning_curves.png`). Put them under `models/<run>/` (git-ignored) and tell me; I evaluate the checkpoints on dev with the Phase 2 harness.

Sizes: if downloading all checkpoints is too slow, download the logs and only `r1_inbatch_best.zip` and `r2_hardneg_best.zip` first.
