"""CPU latency of embedding ONE query, per model variant and thread count.

    uv run python -m fedlex_bench.finetune.latency models/r1_inbatch

Measures wall time of ``embed_queries([question])`` (tokenisation, model, pooling) for 200
validation questions after 10 warm-up calls, for torch fp32, ONNX fp32 and ONNX int8, with 1
thread and with all cores. Nothing else should run on the machine. Writes
``eval/results/latency_<run>.json`` (hardware, library versions, p50, p95, mean per variant)
and prints a table. The search step itself (a matrix product over about 11,000 vectors) is
timed separately and added in the table as ``search``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from fedlex_bench.finetune.dataset import PAIRS_FILE, read_jsonl
from fedlex_bench.ingest import PROJECT_ROOT


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(q * len(ordered)), len(ordered) - 1)]


def time_queries(embed, questions: list[str]) -> list[float]:
    for q in questions[:10]:
        embed([q])
    out = []
    for q in questions:
        t0 = time.perf_counter()
        embed([q])
        out.append((time.perf_counter() - t0) * 1000)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()
    from fedlex_bench.finetune.embedder import TorchE5Embedder
    from fedlex_bench.retrieval.embedding import OnnxE5Embedder

    pairs = [p for p in read_jsonl(PAIRS_FILE) if p["split"] == "val"]
    questions = [p["query"] for p in pairs[::6]][:200]
    onnx_dir = args.run_dir / "onnx"
    cores = os.cpu_count() or 1
    results: dict = {}
    for threads in (1, cores):
        torch.set_num_threads(threads)
        torch_emb = TorchE5Embedder(args.run_dir / "best")
        variants = {
            "torch fp32": torch_emb.embed_queries,
            "onnx fp32": OnnxE5Embedder(
                local_dir=onnx_dir, onnx_file="model.onnx", threads=threads
            ).embed_queries,
            "onnx int8": OnnxE5Embedder(
                local_dir=onnx_dir, onnx_file="model_int8.onnx", threads=threads
            ).embed_queries,
        }
        for name, fn in variants.items():
            ms = time_queries(fn, questions)
            results[f"{name}, {threads} thread{'s' if threads > 1 else ''}"] = {
                "p50_ms": round(statistics.median(ms), 1),
                "p95_ms": round(percentile(ms, 0.95), 1),
                "mean_ms": round(statistics.fmean(ms), 1),
            }
    matrix = np.random.default_rng(0).standard_normal((11000, 384)).astype(np.float32)
    vec = matrix[0]
    search = []
    for _ in range(200):
        t0 = time.perf_counter()
        np.argsort(-(matrix @ vec))[:10]
        search.append((time.perf_counter() - t0) * 1000)
    results["search over 11,000 vectors"] = {
        "p50_ms": round(statistics.median(search), 2),
        "p95_ms": round(percentile(search, 0.95), 2),
    }
    sizes = {f.name: round(f.stat().st_size / 1e6) for f in sorted(onnx_dir.glob("*.onnx"))}
    meta = {
        "cpu": platform.processor(),
        "cores": cores,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "queries": len(questions),
        "file_sizes_mb": sizes,
    }
    try:
        import onnxruntime

        meta["onnxruntime"] = onnxruntime.__version__
    except ImportError:
        pass
    out = PROJECT_ROOT / "eval" / "results" / f"latency_{args.run_dir.name}.json"
    out.write_text(json.dumps({"meta": meta, "ms": results}, indent=1) + "\n", encoding="utf-8")
    for name, r in results.items():
        print(f"{name:34s} p50 {r['p50_ms']:>8} ms  p95 {r['p95_ms']:>8} ms")
    print(meta)


if __name__ == "__main__":
    main()
