"""Fine-tune multilingual-e5-small for cross-lingual retrieval with an in-batch ranking loss.

Plain PyTorch on purpose (about 250 lines, no trainer abstraction), so every number in the log can
be traced to a line here. One step: encode the batch's questions and passages, build the B x B
table of cosines, and train row i to pick column i (cross-entropy over ``scale * cosine``). With
``--hard-negatives`` each question that has one gets an extra column. Two guards keep correct
answers from being used as negatives: batches never hold two questions of the same article, and
any column that is the *same article* as a row's positive (other language, hard negative that is
another row's positive) is masked out of that row.

Format identical to search time (``fedlex_bench.retrieval``): ``"query: "`` / ``"passage: "``
prefixes, the passage is ``title + newline + text`` in the pair's passage language, mean pooling,
L2 normalisation.

Each epoch (epoch 0 = the untrained model) is evaluated on the validation split: in-batch loss on
fixed validation batches, and recall@1/10 and MRR of the positive among the 324 validation
articles (best of the three language versions), per question language. The best epoch by
validation recall@10 is saved to ``models/<run>/best`` (git-ignored) with ``config.json`` (all
arguments, SHA-256 of the data files, git sha, library versions) and ``log.jsonl``.

``--max-steps N`` only times N steps and saves nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from fedlex_bench.finetune.dataset import PAIRS_FILE, _h, read_jsonl
from fedlex_bench.finetune.exclusions import FINETUNE_DIR
from fedlex_bench.finetune.hardneg import HARDNEG_FILE
from fedlex_bench.finetune.synth import load_pool_texts, read_pool
from fedlex_bench.ingest import PROJECT_ROOT
from fedlex_bench.retrieval.chunking import _with_title
from fedlex_bench.retrieval.embedding import E5_REPO, E5_REVISION

MODELS_DIR = PROJECT_ROOT / "models"
LANGS = ("de", "fr", "it")


class Encoder:
    """The e5 model plus tokenizer; ``encode`` returns L2-normalised mean-pooled vectors."""

    def __init__(self, source: str, revision: str | None, device: str) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(source, revision=revision)
        self.model = AutoModel.from_pretrained(source, revision=revision).to(device)
        self.device = device

    def encode(self, prefix: str, texts: list[str], max_len: int) -> torch.Tensor:
        batch = self.tok(
            [prefix + t for t in texts],
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(self.device)
        hidden = self.model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return F.normalize(pooled, dim=-1)


def make_batches(pairs: list[dict], batch_size: int, seed: int, epoch: int) -> list[list[dict]]:
    """Seeded shuffle, then greedy batches without two questions of the same article."""
    order = sorted(pairs, key=lambda p: p["pair_id"])
    random.Random(f"{seed}:{epoch}").shuffle(order)
    batches: list[list[dict]] = []
    waiting = order
    while waiting:
        batch, seen, later = [], set(), []
        for p in waiting:
            if len(batch) < batch_size and p["article_id"] not in seen:
                batch.append(p)
                seen.add(p["article_id"])
            else:
                later.append(p)
        batches.append(batch)
        waiting = later
    return batches


def ranking_loss(q, p, row_articles, col_articles, scale: float) -> torch.Tensor:
    """In-batch softmax over ``scale * cosine``; the target of row i is column i."""
    scores = scale * q @ p.T  # (B, B + number of hard negatives)
    n = q.shape[0]
    same = torch.tensor([[c == r for c in col_articles] for r in row_articles])
    same[torch.arange(n), torch.arange(n)] = False  # keep each row's own positive
    scores = scores.masked_fill(same.to(scores.device), float("-inf"))
    return F.cross_entropy(scores, torch.arange(n, device=scores.device))


class Data:
    def __init__(self, args) -> None:
        pairs = read_jsonl(PAIRS_FILE)
        self.hard = {r["pair_id"]: r["negatives"] for r in read_jsonl(HARDNEG_FILE)}
        pool = read_pool()
        raw = load_pool_texts(pool)
        self.text = {(a, lang): _with_title(*raw[(a, lang)]) for a in pool for lang in LANGS}
        train = [p for p in pairs if p["split"] == "train"]
        if args.fraction < 1:  # nested subsets: the same articles stay in when the fraction grows
            articles = sorted({p["article_id"] for p in train}, key=lambda a: _h("subset", a))
            keep = set(articles[: math.ceil(args.fraction * len(articles))])
            train = [p for p in train if p["article_id"] in keep]
        self.train = train
        self.val = [p for p in pairs if p["split"] == "val"]
        self.val_articles = sorted({p["article_id"] for p in self.val})


def batch_tensors(enc: Encoder, batch: list[dict], data: Data, args, hard: bool):
    q = enc.encode("query: ", [p["query"] for p in batch], args.max_len_query)
    cols = [(p["article_id"], data.text[(p["article_id"], p["passage_lang"])]) for p in batch]
    if hard:
        for p in batch:
            negs = data.hard.get(p["pair_id"], [])
            if negs:
                cols.append((negs[0], data.text[(negs[0], p["passage_lang"])]))
    pas = enc.encode("passage: ", [t for _, t in cols], args.max_len_passage)
    return q, pas, [p["article_id"] for p in batch], [a for a, _ in cols]


@torch.no_grad()
def evaluate(enc: Encoder, data: Data, args) -> dict:
    enc.model.eval()
    losses = [
        ranking_loss(
            *batch_tensors(enc, b, data, args, False)[:2],
            [p["article_id"] for p in b],
            [p["article_id"] for p in b],
            args.scale,
        ).item()
        for b in make_batches(data.val, args.batch_size, args.seed, 0)
    ]
    keys = [(a, lang) for a in data.val_articles for lang in LANGS]
    pas = torch.cat(
        [
            enc.encode("passage: ", [data.text[k] for k in keys[i : i + 64]], args.max_len_passage)
            for i in range(0, len(keys), 64)
        ]
    )
    queries = torch.cat(
        [
            enc.encode("query: ", [p["query"] for p in data.val[i : i + 64]], args.max_len_query)
            for i in range(0, len(data.val), 64)
        ]
    )
    sims = (queries @ pas.T).view(len(data.val), len(data.val_articles), len(LANGS)).max(-1).values
    position = {a: i for i, a in enumerate(data.val_articles)}
    ranks = [
        int((sims[i] > sims[i, position[p["article_id"]]]).sum()) + 1
        for i, p in enumerate(data.val)
    ]
    out: dict = {"val_loss": sum(losses) / len(losses)}
    groups = defaultdict(list)
    for p, r in zip(data.val, ranks, strict=True):
        groups["all"].append(r)
        groups[p["query_lang"]].append(r)
    for name, rs in groups.items():
        out[f"val_r1_{name}"] = sum(r == 1 for r in rs) / len(rs)
        out[f"val_r10_{name}"] = sum(r <= 10 for r in rs) / len(rs)
        out[f"val_mrr_{name}"] = sum(1 / r for r in rs) / len(rs)
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def train(args) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = Encoder(E5_REPO, E5_REVISION, device)
    data = Data(args)
    hard = args.hard_negatives
    steps_per_epoch = len(make_batches(data.train, args.batch_size, args.seed, 0))
    total = steps_per_epoch * args.epochs
    warmup = max(int(args.warmup_ratio * total), 1)
    opt = torch.optim.AdamW(enc.model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else max((total - s) / (total - warmup), 0.0)
    )
    use_amp = args.amp and device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    out_dir = MODELS_DIR / args.run
    log: list[dict] = []
    print(
        f"device {device}; {len(data.train)} train pairs, {steps_per_epoch} steps/epoch, "
        f"{total} steps; hard negatives {hard}",
        flush=True,
    )
    if not args.max_steps:
        out_dir.mkdir(parents=True, exist_ok=True)
        first = {"epoch": 0, "train_loss": None, **evaluate(enc, data, args)}
        log.append(first)
        print(first, flush=True)
    best, step, t0 = -1.0, 0, time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        enc.model.train()
        running = []
        for batch in make_batches(data.train, args.batch_size, args.seed, epoch):
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                q, pas, rows, cols = batch_tensors(enc, batch, data, args, hard)
                loss = ranking_loss(q, pas, rows, cols, args.scale)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(enc.model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running.append(loss.item())
            step += 1
            if step % 20 == 0:
                print(
                    f"epoch {epoch} step {step}/{total} loss {sum(running[-20:]) / 20:.4f} "
                    f"{(time.perf_counter() - t0) / step:.2f}s/step",
                    flush=True,
                )
            if args.max_steps and step >= args.max_steps:
                print(f"{(time.perf_counter() - t0) / step:.2f} s/step over {step} steps")
                return
        row = {
            "epoch": epoch,
            "train_loss": sum(running) / len(running),
            **evaluate(enc, data, args),
        }
        log.append(row)
        print(row, flush=True)
        if row["val_r10_all"] > best:
            best = row["val_r10_all"]
            enc.model.save_pretrained(out_dir / "best")
            enc.tok.save_pretrained(out_dir / "best")
            row["saved_best"] = True
        (out_dir / "log.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in log), encoding="utf-8"
        )
    config = {
        "args": vars(args),
        "git_sha": git_sha(),
        "torch": torch.__version__,
        "base_model": f"{E5_REPO}@{E5_REVISION}",
        "device": device,
        "data_sha256": {
            p.name: sha256(p)
            for p in (PAIRS_FILE, HARDNEG_FILE, FINETUNE_DIR / "train_pool_ids.txt")
        },
        "train_pairs": len(data.train),
        "val_pairs": len(data.val),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=1) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, help="folder name under models/")
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--hard-negatives", action="store_true")
    ap.add_argument("--fraction", type=float, default=1.0, help="share of training articles")
    ap.add_argument("--max-len-query", type=int, default=128)
    ap.add_argument("--max-len-passage", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--amp", action="store_true", help="fp16 autocast on CUDA")
    ap.add_argument("--max-steps", type=int, default=0, help="time N steps, save nothing")
    train(ap.parse_args(argv))


if __name__ == "__main__":
    main()
