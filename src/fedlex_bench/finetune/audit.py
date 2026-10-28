"""Audit of the training data by a model from another family (the questions came from Claude).

Two commands, both with an English output, because nobody on the project reads German, French
and Italian well enough to review legal text:

``audit``: for a seeded sample of training questions, a judge (``gpt-6-luna``) says whether a
candidate article could answer the question: ``yes`` (it contains what is needed), ``partly`` or
``no``, with a one-sentence reason in English. Four sets, 100 questions each (30 for the last):

- **filtered**: the first hard negative chosen by ``hardneg.py``. ``yes`` = a false negative.
- **raw**: the base model's best wrong article, with none of the filter's rules. The difference
  to ``filtered`` is what the filter buys.
- **positive**: the labelled article. ``yes`` = the question is answerable from its own article;
  this measures the quality of the synthetic questions, not of the filter.
- **random**: a random pool article. Should be ``no``; a control for the judge.

``gloss``: English glosses of the 20-example review sheet (``review_hardneg_en.md``).

Run ``uv run python -m fedlex_bench.finetune.audit audit --estimate`` first; ``--yes`` pays.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from fedlex_bench.finetune.dataset import PAIRS_FILE, read_jsonl
from fedlex_bench.finetune.exclusions import FINETUNE_DIR
from fedlex_bench.finetune.hardneg import HARDNEG_FILE
from fedlex_bench.finetune.mine import MINED_FILE
from fedlex_bench.finetune.synth import load_pool_texts, read_pool
from fedlex_bench.llm.factory import build_client
from fedlex_bench.llm.jsonutil import extract_json
from fedlex_bench.types import LLMRequest

JUDGE = "gpt-6-luna"
GLOSS_MODEL = "claude-haiku-5-5"
SEED = 20261028
SET_SIZES = {"filtered": 100, "raw": 100, "positive": 100, "random": 30}
MAX_CHARS = 3000
AUDIT_FILE = FINETUNE_DIR / "audit.json"

SYSTEM = (
    "You check whether a statute article from Swiss federal law can answer a question. The "
    "question may be in English, German, French or Italian; the article is in German. Judge by "
    "meaning, not by shared words.\n"
    "Reply with a JSON object and nothing else: "
    '{"verdict": "yes" | "partly" | "no", "reason": "<one sentence in English>"}\n'
    '"yes": the article contains the rule or fact needed to answer the question.\n'
    '"partly": it answers only a part of the question, or is a closely related rule that would '
    "need another article to answer.\n"
    '"no": it is about something else, or it does not state what the question asks for.'
)
USER = "Question:\n{question}\n\nArticle {article_id} ({title}):\n{text}"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(centre - half, 0.0), min(centre + half, 1.0)


def build_items() -> list[dict]:
    """The audit items: a sample of train questions with a candidate article per set."""
    pairs = {p["pair_id"]: p for p in read_jsonl(PAIRS_FILE)}
    hard = {r["pair_id"]: r for r in read_jsonl(HARDNEG_FILE)}
    mined = {r["pair_id"]: r for r in read_jsonl(MINED_FILE)}
    pool = read_pool()
    rng = random.Random(SEED)
    with_neg = sorted(pid for pid, r in hard.items() if r["negatives"])
    sample = rng.sample(with_neg, SET_SIZES["filtered"])  # the same questions in every set
    items = []
    for i, pid in enumerate(sample):
        p, m = pairs[pid], mined[pid]
        cands = {
            "filtered": hard[pid]["negatives"][0],
            "raw": m["emb"][0][0],
            "positive": p["article_id"],
        }
        if i < SET_SIZES["random"]:
            cands["random"] = rng.choice([a for a in pool if a != p["article_id"]])
        for name, article in cands.items():
            items.append(
                {
                    "set": name,
                    "pair_id": pid,
                    "article_id": article,
                    "style": p["style"],
                    "query_lang": p["query_lang"],
                    "question": p["query"],
                }
            )
    return items


def request_for(item: dict, texts: dict, attempt: int = 0) -> LLMRequest:
    title, text = texts[(item["article_id"], "de")]
    user = USER.format(
        question=item["question"], article_id=item["article_id"], title=title, text=text[:MAX_CHARS]
    )
    if attempt:
        user += '\n\nReply with only the JSON object {"verdict": ..., "reason": ...}.'
    return LLMRequest(system=SYSTEM, user=user, max_output_tokens=1500, json_output=True)


def judge_one(client, item: dict, texts: dict) -> dict:
    for attempt in range(2):
        resp = client.complete(request_for(item, texts, attempt))
        obj = extract_json(resp.text)
        if isinstance(obj, dict) and obj.get("verdict") in ("yes", "partly", "no"):
            return {**item, "verdict": obj["verdict"], "reason": str(obj.get("reason", ""))}
    return {**item, "verdict": "failed", "reason": ""}


def cmd_audit(args) -> None:
    items = build_items()
    texts = load_pool_texts(read_pool())
    client = build_client(JUDGE)
    proj = client.project([request_for(i, texts) for i in items], 150)
    print(
        f"{len(items)} judge calls on {JUDGE}; projected {proj.cost_usd:.3f} USD; remaining "
        f"budget {proj.remaining_usd:.2f}; needs confirmation: {proj.needs_confirmation}"
    )
    if args.estimate or not args.yes:
        return
    if proj.exceeds_remaining or proj.needs_confirmation:
        raise SystemExit("needs approval or exceeds the budget")
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(lambda i: judge_one(client, i, texts), items))
    (FINETUNE_DIR / "audit_rows.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    summary: dict = {"judge": JUDGE, "seed": SEED, "sets": {}}
    for name in SET_SIZES:
        part = [r for r in rows if r["set"] == name]
        counts = Counter(r["verdict"] for r in part)
        n = sum(counts[v] for v in ("yes", "partly", "no"))
        yes, partly = counts["yes"], counts["partly"]
        summary["sets"][name] = {
            "n": n,
            "failed": counts["failed"],
            "yes": yes,
            "partly": partly,
            "no": counts["no"],
            "yes_rate": round(yes / n, 3),
            "yes_ci95": [round(x, 3) for x in wilson(yes, n)],
            "yes_or_partly_rate": round((yes + partly) / n, 3),
        }
        print(
            f"{name:9s} n={n:3d}  yes {yes:3d} ({yes / n:.2f})  partly {partly:3d}  "
            f"no {counts['no']:3d}  failed {counts['failed']}  95% CI for yes "
            f"{wilson(yes, n)[0]:.2f} to {wilson(yes, n)[1]:.2f}"
        )
    AUDIT_FILE.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")


GLOSS_SYSTEM = (
    "Translate the texts into plain English for a reader who knows no German, French or Italian. "
    "Keep legal terms accurate and keep numbers. Reply with a JSON object and nothing else: "
    '{"translations": ["...", "..."]} with one entry per input text, in the same order.'
)


def cmd_gloss(args) -> None:
    from fedlex_bench.finetune.hardneg import REVIEW_FILE, _snippet, write_review  # noqa: F401

    pairs = {p["pair_id"]: p for p in read_jsonl(PAIRS_FILE)}
    rows = read_jsonl(HARDNEG_FILE)
    step = max(len(rows) // args.show, 1)
    chosen = rows[::step][: args.show]
    pool = read_pool()
    german = load_pool_texts(pool)
    client = build_client(GLOSS_MODEL)
    out = [
        "# Hard-negative review sheet, English glosses (machine translation, Claude Haiku 5.5)\n",
        "Question and article texts are translated; the audit verdicts are in `audit.json`.\n",
    ]

    def one(r: dict) -> str:
        p = pairs[r["pair_id"]]
        keys = [p["article_id"], *r["negatives"]]
        texts = [
            p["query"],
            *[f"{german[(a, 'de')][0]}: {_snippet(german[(a, 'de')][1], 400)}" for a in keys],
        ]
        req = LLMRequest(
            system=GLOSS_SYSTEM,
            user=json.dumps(texts, ensure_ascii=False),
            max_output_tokens=3000,
            json_output=True,
        )
        obj = extract_json(client.complete(req).text) or {}
        tr = obj.get("translations", [])
        if len(tr) != len(texts):
            return f"## {r['pair_id']}\n(translation failed)\n"
        lines = [
            f"## {r['pair_id']}  [{p['query_lang']} / {p['style']}]  pos_rank {r['pos_rank']}",
            f"**Question:** {tr[0]}\n",
            f"**Positive {keys[0]}:** {tr[1]}\n",
        ]
        lines += [f"- **Negative {a}:** {t}" for a, t in zip(keys[1:], tr[2:], strict=True)]
        if not r["negatives"]:
            lines.append("- (no negative kept)")
        return "\n".join(lines) + "\n"

    with ThreadPoolExecutor(max_workers=8) as ex:
        out += list(ex.map(one, chosen))
    path = FINETUNE_DIR / "review_hardneg_en.md"
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit")
    a.add_argument("--estimate", action="store_true")
    a.add_argument("--yes", action="store_true")
    g = sub.add_parser("gloss")
    g.add_argument("--show", type=int, default=20)
    args = parser.parse_args(argv)
    {"audit": cmd_audit, "gloss": cmd_gloss}[args.cmd](args)


if __name__ == "__main__":
    main()
