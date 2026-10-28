"""A fine-tuned checkpoint as an ``Embedder`` for the Phase 2 harness (torch, CPU or GPU).

Same contract as ``OnnxE5Embedder``: ``"query: "`` / ``"passage: "`` prefixes, mean pooling over the
attention mask, L2 normalisation. Padding is masked (the Phase 2 ONNX embedder did not do this
before the 2026-10-28 fix; see docs/phase3_progress.md). The model id carries the folder name and
a hash of the weights, so the vector cache of one checkpoint never serves another.

Use it from the harness with ``eval run ... --embedder models/<run>/best``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class TorchE5Embedder:
    dim = 384

    def __init__(self, path: str | Path, max_length: int = 512, batch_size: int = 32) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.path = Path(path)
        weights = next(self.path.glob("*.safetensors"))
        digest = hashlib.sha256(weights.read_bytes()).hexdigest()[:12]
        self.model_id = f"torch-e5/{self.path.parent.name}/{digest}/maxlen{max_length}"
        self.max_length, self.batch_size = max_length, batch_size
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModel.from_pretrained(self.path).eval()

    @torch.no_grad()
    def _embed(self, prefix: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        order = sorted(range(len(texts)), key=lambda i: (-len(texts[i]), i))
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            batch = self.tok(
                [prefix + texts[i] for i in idx],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            hidden = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = F.normalize((hidden * mask).sum(1) / mask.sum(1).clamp(min=1), dim=-1)
            out[idx] = pooled.numpy()
        return out

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._embed("passage: ", texts)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed("query: ", texts)
