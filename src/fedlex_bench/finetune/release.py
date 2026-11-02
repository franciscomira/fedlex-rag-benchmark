"""Assemble (and optionally upload) the Hugging Face release folder for a trained run.

    uv run python -m fedlex_bench.finetune.release models/r1_inbatch --repo-id USER/NAME
    uv run python -m fedlex_bench.finetune.release models/r1_inbatch --repo-id USER/NAME --upload

Builds ``models/release/<repo name>/``: the checkpoint (``model.safetensors``, config, tokenizer),
``onnx/model_int8.onnx`` plus ``onnx/tokenizer.json``, ``training_config.json``,
``training_log.jsonl`` and ``README.md`` (``docs/model_card.md`` with ``REPO_ID`` replaced). It
refuses to continue if the card still has a placeholder. ``--upload`` creates the repository
(public unless ``--private``) and uploads the folder; it needs a prior ``hf auth login`` and is the
only step that sends anything off this machine.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from fedlex_bench.ingest import PROJECT_ROOT

CARD = PROJECT_ROOT / "docs" / "model_card.md"
CHECKPOINT_FILES = ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json")


def build(run_dir: Path, repo_id: str) -> Path:
    out = PROJECT_ROOT / "models" / "release" / repo_id.split("/")[-1]
    if out.exists():
        shutil.rmtree(out)
    (out / "onnx").mkdir(parents=True)
    for name in CHECKPOINT_FILES:
        shutil.copy(run_dir / "best" / name, out / name)
    shutil.copy(run_dir / "onnx" / "model_int8.onnx", out / "onnx" / "model_int8.onnx")
    shutil.copy(run_dir / "onnx" / "tokenizer.json", out / "onnx" / "tokenizer.json")
    shutil.copy(run_dir / "config.json", out / "training_config.json")
    shutil.copy(run_dir / "log.jsonl", out / "training_log.jsonl")
    card = CARD.read_text(encoding="utf-8").replace("REPO_ID", repo_id)
    for placeholder in ("PATH_OR_REPO", "REPO_ID", "TODO", "draft model card"):
        if placeholder in card:
            raise SystemExit(f"model card still contains {placeholder!r}")
    (out / "README.md").write_text(card, encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--repo-id", required=True, help="<hf user>/<model name>")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    out = build(args.run_dir, args.repo_id)
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"release folder: {out} ({total:.0f} MB)")
    for f in sorted(out.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(out).as_posix():28s} {f.stat().st_size / 1e6:8.1f} MB")
    if args.upload:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(
            folder_path=str(out),
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Fine-tuned multilingual-e5-small for Swiss federal law retrieval",
        )
        print(f"uploaded: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
