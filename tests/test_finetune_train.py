"""Batching and loss guards of the fine-tuning loop (no model, no network)."""

from __future__ import annotations

import torch

from fedlex_bench.finetune.train import make_batches, ranking_loss


def _pairs():
    return [
        {"pair_id": f"{a}#{s}", "article_id": a}
        for a in ("1:1", "1:2", "1:3", "1:4", "1:5")
        for s in range(1, 5)
    ]


def test_batches_cover_everything_once_and_never_repeat_an_article():
    pairs = _pairs()
    batches = make_batches(pairs, 4, seed=1, epoch=1)
    flat = [p["pair_id"] for b in batches for p in b]
    assert sorted(flat) == sorted(p["pair_id"] for p in pairs)
    for b in batches:
        assert len(b) <= 4
        assert len({p["article_id"] for p in b}) == len(b)


def test_batches_are_deterministic_and_change_with_epoch():
    a = make_batches(_pairs(), 4, seed=1, epoch=1)
    assert a == make_batches(_pairs(), 4, seed=1, epoch=1)
    assert a != make_batches(_pairs(), 4, seed=1, epoch=2)


def test_loss_is_low_for_a_correct_ranking_and_ignores_same_article_columns():
    q = torch.eye(3)
    p = torch.eye(3)
    low = ranking_loss(q, p, ["a", "b", "c"], ["a", "b", "c"], scale=20.0)
    assert low < 1e-6
    # A fourth column that is the same article as row 0 would score as high as its positive;
    # without the mask the loss would be log(2), with it the row is unaffected.
    q2 = torch.eye(3)
    p2 = torch.cat([torch.eye(3), torch.tensor([[1.0, 0.0, 0.0]])])
    masked = ranking_loss(q2, p2, ["a", "b", "c"], ["a", "b", "c", "a"], scale=20.0)
    unmasked = ranking_loss(q2, p2, ["a", "b", "c"], ["a", "b", "c", "d"], scale=20.0)
    assert masked < 1e-6 < unmasked


def test_torch_embedder_batch_equals_single_text():
    """Padding must not change a vector (skipped when no trained checkpoint is on this machine)."""
    import pytest

    from fedlex_bench.ingest import PROJECT_ROOT

    path = PROJECT_ROOT / "models" / "r1_inbatch" / "best"
    if not (path / "model.safetensors").exists():
        pytest.skip("no checkpoint")
    from fedlex_bench.finetune.embedder import TorchE5Embedder

    emb = TorchE5Embedder(path)
    short = "Der Mieter hat den Mietzins zu bezahlen."
    long = " ".join(
        ["Der Vermieter darf den Mietzins nur nach Massgabe des Gesetzes erhoehen."] * 12
    )
    batch = emb.embed_passages([short, long])
    assert float(batch[0] @ emb.embed_passages([short])[0]) > 0.9999
