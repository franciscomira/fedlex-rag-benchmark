"""Export a fine-tuned checkpoint to ONNX (fp32 and dynamic int8) for CPU serving.

    uv run python -m fedlex_bench.finetune.export_onnx models/r1_inbatch/best

Writes next to the checkpoint's parent: ``models/<run>/onnx/model.onnx`` (fp32),
``model_int8.onnx`` (int8 weights per channel, dynamic activation quantisation) and ``tokenizer.json``.
Inputs and pooling are those of ``OnnxE5Embedder`` (``input_ids``, ``attention_mask``,
``token_type_ids`` in; ``last_hidden_state`` out; the embedder applies the masked mean and the
L2 normalisation). The script checks the fp32 export against the torch model on a few texts and
reports the cosine of both ONNX variants to torch; ``--check-only`` skips the export.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

SAMPLE_TEXTS = [
    "query: Kann mein Vermieter den Mietzins einseitig erhoehen?",
    "query: Can my landlord raise the rent during a fixed-term lease?",
    "passage: Der Mieter hat den Mietzins und allfaellige Nebenkosten monatlich zu bezahlen.",
    "passage: Le bailleur peut resilier le bail moyennant un preavis de trois mois.",
]


class _HiddenStates(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        ).last_hidden_state


def export(checkpoint: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModel, AutoTokenizer

    out = checkpoint.parent / "onnx"
    out.mkdir(exist_ok=True)
    tok = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint).eval()
    batch = tok(SAMPLE_TEXTS[:2], padding=True, return_tensors="pt")
    args = (batch["input_ids"], batch["attention_mask"], torch.zeros_like(batch["input_ids"]))
    torch.onnx.export(
        _HiddenStates(model),
        args,
        str(out / "model.onnx"),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "token_type_ids": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=17,
        dynamo=False,
    )
    quantize_dynamic(
        out / "model.onnx", out / "model_int8.onnx", weight_type=QuantType.QInt8, per_channel=True
    )
    shutil.copy(checkpoint / "tokenizer.json", out / "tokenizer.json")
    return out


def check(checkpoint: Path, onnx_dir: Path) -> None:
    from fedlex_bench.finetune.embedder import TorchE5Embedder
    from fedlex_bench.retrieval.embedding import OnnxE5Embedder

    reference = TorchE5Embedder(checkpoint)
    texts = [t.split(": ", 1)[1] for t in SAMPLE_TEXTS]
    ref = np.vstack([reference.embed_queries(texts[:2]), reference.embed_passages(texts[2:])])
    for name in ("model.onnx", "model_int8.onnx"):
        emb = OnnxE5Embedder(local_dir=onnx_dir, onnx_file=name)
        got = np.vstack([emb.embed_queries(texts[:2]), emb.embed_passages(texts[2:])])
        cos = (ref * got).sum(1)
        size = (onnx_dir / name).stat().st_size / 1e6
        print(f"{name}: {size:.0f} MB, cosine to torch min {cos.min():.5f} mean {cos.mean():.5f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    onnx_dir = args.checkpoint.parent / "onnx" if args.check_only else export(args.checkpoint)
    check(args.checkpoint, onnx_dir)


if __name__ == "__main__":
    main()
