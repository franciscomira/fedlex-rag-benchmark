---
language: [de, fr, it, en]
license: mit
base_model: intfloat/multilingual-e5-small
tags: [sentence-embeddings, retrieval, legal, swiss-law, cross-lingual, multilingual-e5]
library_name: transformers
---

# fedlex-e5-small: cross-lingual retrieval over Swiss federal law

Numbers below come from `docs/phase3_report.md` and `docs/phase3_progress.md` in the project repository ([franciscomira/fedlex-rag-benchmark](https://github.com/franciscomira/fedlex-rag-benchmark)).

A fine-tune of [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) (118M parameters, 384 dimensions) that retrieves Swiss federal law articles (German, French, Italian) for questions in English, German, French or Italian. It is meant as the first stage of a retrieval pipeline: it finds candidate articles; it does not answer questions.

**Not legal advice. Not official.** The statute texts used for training and evaluation come from Fedlex, the official publication platform of the Swiss Confederation, but this model and its outputs are not published or endorsed by the Swiss Confederation, and only the texts published on Fedlex are authentic. Do not rely on this model for legal decisions.

## Use

Same conventions as the base model: prefix questions with `query: ` and articles with `passage: `, mean-pool the last hidden state over the attention mask, L2-normalise, rank by dot product.

```python
import torch, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("REPO_ID")
model = AutoModel.from_pretrained("REPO_ID").eval()

def embed(texts, prefix):
    batch = tok([prefix + t for t in texts], padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        hidden = model(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    return F.normalize((hidden * mask).sum(1) / mask.sum(1).clamp(min=1), dim=-1)

q = embed(["Can my landlord raise the rent during a fixed-term lease?"], "query: ")
p = embed(["Der Vermieter kann den Mietzins ... (article text)"], "passage: ")
score = q @ p.T
```

An ONNX export (fp32, 470 MB) and an int8 export (118 MB, per-channel dynamic quantisation) exist; on 200 questions on one 6-core CPU they embed a question in 22 ms (fp32, 1 thread) and 20 ms (int8, 1 thread). Padding must be masked in the pooling step (see the bug note below).

## Training data

- **Articles:** 3,201 articles of eight federal acts (Code of Obligations 220, Civil Code 210, Civil Procedure 272, Debt Enforcement and Bankruptcy 281.1 and others; German, French and Italian versions from Fedlex). Excluded by code: all 172 gold articles of the evaluation benchmark (dev and test), their neighbours (the previous and next article of the same act, 288) and 3 articles whose text overlaps a gold article by at least 50% (character 5-gram Jaccard). Tests fail on any overlap by that criterion. It does not catch a short article contained in a long gold article: 37 pool articles are at least 60% contained in one of 18 gold articles; the test gain is unchanged on items not touched by them (+0.30 [+0.20, +0.40]).
- **Questions:** 12,800 synthetic questions (4 per article: 2 English, 2 in two of German, French, Italian; six styles: keywords, short question, legal wording, lay situation, third party, confused), written by Claude Haiku 5.5 from the article text shown in a language other than the question's. No benchmark question was shown to the generator. 11,504 for training and 1,296 for validation (split by article, 324 held-out articles). One passage language (de, fr or it) per question.
- **Audit:** a judge from another model family (gpt-6-luna) found that 80 of 100 sampled questions are fully answered by their article, 20 partly, 0 not at all. The question writer's planned fixed roles for the third-party style were not applied, so the roles in those questions are the generator's choice. No human who reads German, French or Italian, and no Swiss-law expert, reviewed the questions.

## Training

Plain PyTorch, in-batch softmax ranking loss (scale 20, batch 64, no two questions of the same article in a batch, same-article columns masked), AdamW, learning rate 2e-5, 10% warmup then linear decay, 3 epochs (540 steps), fp16 autocast, passages truncated to 256 tokens, seed 20261028, one free Kaggle T4. Hard negatives were tried and did not help (+0.017 recall@10 on dev, interval touching zero), so they are not used. Training takes under 15 minutes.

## Evaluation (Fedlex cross-lingual benchmark v1, test split, 164 answerable items; 95% bootstrap intervals over question groups)

| System | Recall@1 | Recall@10 | MRR | English recall@10 |
| --- | --- | --- | --- | --- |
| multilingual-e5-small + BM25 hybrid, no model | 0.25 | 0.55 [0.46, 0.63] | 0.36 | 0.24 |
| **this model, embeddings only** | 0.48 [0.38, 0.59] | **0.81 [0.71, 0.90]** | 0.63 [0.53, 0.73] | **0.80** |
| this model + Claude Haiku 5.5 reranker | 0.79 [0.70, 0.88] | 0.89 [0.80, 0.97] | 0.87 [0.78, 0.95] | 0.89 |
| LLM query translation + hybrid + Haiku reranker | 0.76 [0.67, 0.84] | 0.87 [0.79, 0.94] | 0.84 [0.76, 0.91] | 0.86 |

The tuned model alone (embeddings only) improves recall@10 over the untuned hybrid (embedding plus BM25) by +0.26 [+0.17, +0.35]; on dev the untuned model scored the same as embedding-only and as hybrid, so this is the effect of tuning. Against the LLM-translation pipeline, recall@10 is -0.06 [-0.15, +0.02] (not distinguishable) and MRR -0.21 [-0.31, -0.11] (clearly worse at putting the right article first). With a reranker on top the two pipelines are not distinguishable (+0.015 [-0.05, +0.07] recall@10), the tuned one at a fifth of the latency (1.6 s against 8 s) and two thirds of the cost per question.

## Limitations and intended scope

- **Retrieval only.** No answer quality, citation or faithfulness measurement was made for the tuned system.
- **The benchmark questions were written by an LLM from article text**, as were the training questions (different prompt, varied styles, same model family). Part of the gain may be adaptation to this question style; real user questions have not been tested.
- **Small evaluation sets** (about 110 independent test question groups); per-language results hold about 40 items each and are directions, not measurements.
- **Eight federal acts only** (about 3,700 articles), not the whole federal law, no cantonal law, no case law. Articles are retrieved whole (no paragraph chunks).
- One training seed; no human or expert review of the training data; the data and the benchmark share a drafting model family.
- **Bug note (Phase 2 embedder):** an earlier version of the project's ONNX embedder averaged padding tokens into batched embeddings. If you serve this model with your own ONNX code, check that batched and single-text embeddings agree.

## Reproduction and files

Code, data builders, tests, evaluation runs and reports: the project repository (`src/fedlex_bench/finetune/`, `data/finetune/README.md`, `docs/phase3_report.md`). Data files: training pairs and the pool of article ids are in the repository; the Fedlex article texts come from the public Fedlex platform.

## Provenance and checks

- Base model licence (MIT) checked on the Hugging Face model page, 2026-11-02. The synthetic training questions were written with Claude Haiku 5.5; the statute texts come from Fedlex (see the notice above); the dataset itself is not published here.
- `training_config.json` and `training_log.jsonl` are the configuration (settings, seed, SHA-256 of the data files) and the per-epoch log of the run that produced these weights. Its `git_sha` field reads `unknown` (the code ran from a Kaggle kit, not a clone); the project repository at the commit of the report contains the training code.
- `onnx/model_int8.onnx` was checked on the dev split (recall@10 0.78 against 0.79 for the fp32 model, difference not claimable) and then run once on the test split: recall@10 0.79 [0.69, 0.88] against 0.81 [0.71, 0.90] for fp32 (difference -0.021 [-0.060, +0.012], not claimable) and MRR 0.58 against 0.63 (difference -0.055 [-0.097, -0.018], claimable): the int8 file orders the results slightly worse. Details: `docs/phase4_progress.md` in the project repository.
