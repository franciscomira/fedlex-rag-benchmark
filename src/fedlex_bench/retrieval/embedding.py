"""Small multilingual embeddings on CPU, cached on disk by content hash.

Model: ``intfloat/multilingual-e5-small`` (118M parameters, 384 dimensions, DE/FR/IT/EN), pinned to
a revision, run with ONNX Runtime on CPU (no torch). Weights come from the Hugging Face cache
(downloaded once, ~470 MB). E5 convention: ``"query: "`` / ``"passage: "`` prefixes, mean pooling,
L2 normalisation, so a dot product is the cosine. Inputs are truncated at 512 tokens (articles with
more than about 350 words lose their tail in the ``article`` variant; the ``window`` variant exists
partly for that reason).

Cache: ``data/processed/index/<model slug>/store.npz`` holds ``hashes`` and ``vectors``. The key of
a vector is ``sha256(model_id + NUL + text)``, so changing the model, revision, max length or text
re-embeds, and identical texts across chunking variants share one vector. The directory carries a
``.gitignore`` containing ``*``. Vectors computed in different batch compositions can differ in the
last float bits; a cache hit always returns the stored vector, which is what makes reruns identical.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from fedlex_bench.ingest import PROJECT_ROOT

log = logging.getLogger("fedlex_bench.retrieval.embedding")

E5_REPO = "intfloat/multilingual-e5-small"
E5_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
E5_MAX_LENGTH = 512
INDEX_DIR = PROJECT_ROOT / "data" / "processed" / "index"
_SAVE_EVERY = 1024  # passages between checkpoints while embedding


@runtime_checkable
class Embedder(Protocol):
    model_id: str  # identifies model, revision and settings; part of every cache key
    dim: int

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...

    def embed_queries(self, texts: list[str]) -> np.ndarray: ...


def content_hash(model_id: str, text: str) -> str:
    return hashlib.sha256(f"{model_id}\0{text}".encode()).hexdigest()


class OnnxE5Embedder:
    """multilingual-e5-small via ONNX Runtime (CPU). Lazy: the model loads on first use."""

    dim = 384

    def __init__(
        self,
        repo: str = E5_REPO,
        revision: str = E5_REVISION,
        max_length: int = E5_MAX_LENGTH,
        batch_size: int = 32,
        threads: int | None = None,
    ) -> None:
        self.repo, self.revision = repo, revision
        self.max_length, self.batch_size, self.threads = max_length, batch_size, threads
        self.model_id = f"{repo}@{revision[:12]}/maxlen{max_length}/onnx-fp32"
        self._session = None
        self._tok = None

    def _load(self) -> None:
        if self._session is not None:
            return
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        model = hf_hub_download(self.repo, "onnx/model.onnx", revision=self.revision)
        tok_path = hf_hub_download(self.repo, "onnx/tokenizer.json", revision=self.revision)
        tok = Tokenizer.from_file(tok_path)
        tok.enable_truncation(max_length=self.max_length)
        tok.enable_padding(pad_id=tok.token_to_id("<pad>"), pad_token="<pad>")
        opts = ort.SessionOptions()
        if self.threads:
            opts.intra_op_num_threads = self.threads
        opts.log_severity_level = 3
        self._session = ort.InferenceSession(
            model, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._tok = tok
        self._input_names = {i.name for i in self._session.get_inputs()}

    def _embed(self, prefix: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        self._load()
        encs = self._tok.encode_batch([prefix + t for t in texts])
        lengths = [len(e.ids) for e in encs]
        order = sorted(range(len(texts)), key=lambda i: (-lengths[i], i))  # long first, stable
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            width = max(lengths[i] for i in idx)
            ids = np.full((len(idx), width), self._tok.token_to_id("<pad>"), dtype=np.int64)
            mask = np.zeros((len(idx), width), dtype=np.int64)
            for row, i in enumerate(idx):
                n = lengths[i]
                ids[row, :n] = encs[i].ids
                mask[row, :n] = 1
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._input_names:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self._session.run(None, feed)[0]
            m = mask[:, :, None].astype(np.float32)
            pooled = (hidden * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1.0)
            pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
            out[idx] = pooled
        return out

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._embed("passage: ", texts)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed("query: ", texts)


def _slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")


class CachedEmbedder:
    """Wraps an :class:`Embedder`; passage vectors are stored on disk by content hash."""

    def __init__(self, inner: Embedder, cache_dir: Path | str = INDEX_DIR) -> None:
        self.inner = inner
        self.model_id = inner.model_id
        self.dim = inner.dim
        root = Path(cache_dir)
        root.mkdir(parents=True, exist_ok=True)
        gi = root / ".gitignore"
        if not gi.exists():
            gi.write_text("*\n", encoding="utf-8")
        self.dir = root / _slug(self.model_id)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._path = self.dir / "store.npz"
        self._hashes: list[str] = []
        self._vecs = np.zeros((0, self.dim), dtype=np.float32)
        if self._path.exists():
            with np.load(self._path, allow_pickle=False) as z:
                self._hashes = [str(h) for h in z["hashes"]]
                self._vecs = z["vectors"].astype(np.float32, copy=False)
        self._row = {h: i for i, h in enumerate(self._hashes)}

    def __len__(self) -> int:
        return len(self._hashes)

    def _save(self) -> None:
        tmp = self.dir / "store.tmp.npz"
        np.savez(tmp, hashes=np.array(self._hashes, dtype="U64"), vectors=self._vecs)
        os.replace(tmp, self._path)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        hashes = [content_hash(self.model_id, t) for t in texts]
        missing: dict[str, str] = {}
        for h, t in zip(hashes, texts, strict=True):
            if h not in self._row and h not in missing:
                missing[h] = t
        if missing:
            self._embed_missing(list(missing.items()))
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._vecs[[self._row[h] for h in hashes]]

    def _embed_missing(self, items: list[tuple[str, str]]) -> None:
        # Length-sorted chunks so each inner call batches similar lengths; checkpoint after each.
        items = sorted(items, key=lambda kv: (-len(kv[1]), kv[0]))
        t0 = time.perf_counter()
        for start in range(0, len(items), _SAVE_EVERY):
            part = items[start : start + _SAVE_EVERY]
            vecs = self.inner.embed_passages([t for _, t in part])
            base = len(self._hashes)
            self._hashes.extend(h for h, _ in part)
            self._vecs = np.vstack([self._vecs, vecs.astype(np.float32, copy=False)])
            self._row.update({h: base + i for i, (h, _) in enumerate(part)})
            self._save()
            log.info(
                "embedded %d/%d new passages (%.0fs)",
                start + len(part),
                len(items),
                time.perf_counter() - t0,
            )

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.inner.embed_queries(texts)
