from __future__ import annotations

import numpy as np
import pytest
from retrieval_fixtures import FakeEmbedder

from fedlex_bench.retrieval.embedding import CachedEmbedder, content_hash


def test_content_hash_depends_on_model_and_text():
    assert content_hash("m", "t") == content_hash("m", "t")
    assert content_hash("m", "t") != content_hash("m2", "t")
    assert content_hash("m", "t") != content_hash("m", "u")


def test_cache_computes_once_and_persists(tmp_path):
    inner = FakeEmbedder()
    c = CachedEmbedder(inner, tmp_path)
    first = c.embed_passages(["alpha beta", "gamma", "alpha beta"])
    assert inner.passage_calls == 2  # duplicate text embedded once
    assert first.shape == (3, inner.dim)
    assert np.array_equal(first[0], first[2])

    inner2 = FakeEmbedder()
    c2 = CachedEmbedder(inner2, tmp_path)  # new object, same directory
    again = c2.embed_passages(["gamma", "alpha beta", "delta"])
    assert inner2.passage_calls == 1  # only "delta" is new
    assert np.array_equal(again[0], first[1]) and np.array_equal(again[1], first[0])

    assert (tmp_path / ".gitignore").read_text().strip() == "*"


def test_cache_partitioned_by_model_id(tmp_path):
    class Other(FakeEmbedder):
        model_id = "fake/other"

    CachedEmbedder(FakeEmbedder(), tmp_path).embed_passages(["a"])
    other = Other()
    CachedEmbedder(other, tmp_path).embed_passages(["a"])
    assert other.passage_calls == 1


def test_queries_pass_through_uncached(tmp_path):
    inner = FakeEmbedder()
    c = CachedEmbedder(inner, tmp_path)
    q = c.embed_queries(["frage"])
    assert q.shape == (1, inner.dim) and inner.query_calls == 1


def test_empty_input(tmp_path):
    c = CachedEmbedder(FakeEmbedder(), tmp_path)
    assert c.embed_passages([]).shape == (0, c.dim)


def _model_available() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        from fedlex_bench.retrieval.embedding import E5_REPO, E5_REVISION

        p = try_to_load_from_cache(E5_REPO, "onnx/model.onnx", revision=E5_REVISION)
        return isinstance(p, str)
    except Exception:
        return False


@pytest.mark.skipif(not _model_available(), reason="e5 ONNX model not in local HF cache")
def test_e5_onnx_semantic_sanity():
    from fedlex_bench.retrieval.embedding import OnnxE5Embedder

    emb = OnnxE5Embedder()
    q = emb.embed_queries(["Wie hoch darf der Mietzins sein?"])
    p = emb.embed_passages(
        [
            "Le loyer est abusif lorsqu'il permet d'obtenir un rendement excessif.",
            "Le mariage est celebre devant l'officier de l'etat civil.",
        ]
    )
    assert q.shape == (1, 384) and p.shape == (2, 384)
    assert np.allclose(np.linalg.norm(p, axis=1), 1.0, atol=1e-4)
    sims = (q @ p.T)[0]
    assert sims[0] > sims[1]
    assert np.allclose(emb.embed_passages(["abc"]), emb.embed_passages(["abc"]))


@pytest.mark.skipif(not _model_available(), reason="e5 ONNX model not in local HF cache")
def test_e5_onnx_batch_equals_single_text():
    """Padding must not change a vector: a short text embedded in a batch with a long one equals
    the same text embedded alone (the Phase 2 embedder averaged pad tokens into batched texts)."""
    from fedlex_bench.retrieval.embedding import OnnxE5Embedder

    emb = OnnxE5Embedder()
    short = "Der Mieter hat den Mietzins zu bezahlen."
    long = " ".join(
        ["Der Vermieter darf den Mietzins nur nach Massgabe des Gesetzes erhoehen."] * 12
    )
    batch = emb.embed_passages([short, long, short])
    alone = emb.embed_passages([short])[0]
    assert float(batch[0] @ alone) > 0.9999 and float(batch[2] @ alone) > 0.9999
    # the legacy mode keeps the old behaviour and its own cache id
    legacy = OnnxE5Embedder(mask_padding=False)
    assert legacy.model_id != emb.model_id
    assert float(legacy.embed_passages([short, long])[0] @ alone) < 0.999
