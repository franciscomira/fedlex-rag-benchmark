"""Synthetic training questions: four per training article, in EN/DE/FR/IT, in varied styles.

Plan per article (deterministic, from a hash of the article id and ``SEED``):

- Slots 1 and 2 are English; slots 3 and 4 are two of DE/FR/IT. The language left out is the one
  the article is **shown** in, so every question is written in a language other than the text
  the model reads. That removes the easiest copy route (reusing the article's own words) and
  makes DE/FR/IT questions cross-lingual, as most benchmark questions are.
- Each slot gets one of six styles, four distinct styles per article (see the system prompt).
  The benchmark's drafting style is first-person, two to three sentences; only ``lay_situation``
  resembles it, so most training questions do not.

Code, not the model, checks every reply: JSON shape, one question per slot, length bounds, no
article/SR numbers, no duplicates. Failures are retried once with a different request, then
counted and reported. Output: ``data/finetune/queries.jsonl`` (one row per question, sorted).

Commands (paid calls need ``--yes``; the cache makes reruns free)::

    uv run python -m fedlex_bench.finetune.synth estimate
    uv run python -m fedlex_bench.finetune.synth run --limit 20 --yes
    uv run python -m fedlex_bench.finetune.synth run --yes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fedlex_bench.benchmark.common import id_key
from fedlex_bench.chunk import ARTICLES_PARQUET
from fedlex_bench.finetune.exclusions import FINETUNE_DIR
from fedlex_bench.llm.factory import build_client
from fedlex_bench.llm.jsonutil import extract_json
from fedlex_bench.types import LLMRequest

MODEL = "claude-haiku-5-5"
SEED = 20261028
LANGS = ("de", "fr", "it")
LANG_NAMES = {"en": "English", "de": "German", "fr": "French", "it": "Italian"}
STYLES = ("keywords", "short_question", "legal_wording", "lay_situation", "third_party", "confused")
STYLE_HINT = {  # repeated next to each slot so the model cannot drift to its default style
    "keywords": "2 to 8 words, search-engine style",
    "short_question": "one direct neutral question, at most 20 words",
    "legal_wording": "precise legal terminology, one or two sentences",
    "lay_situation": "first person, everyday situation, no legal terms",
    "third_party": "asked for someone else or from a small business owner's view",
    "confused": "casual, imprecise, maybe a wrong assumption",
}
PROMPT_DIR = Path(__file__).with_name("prompts")
MAX_ARTICLE_CHARS = 6000
EXPECTED_OUTPUT_TOKENS = (
    700  # visible JSON (~250) plus reasoning at effort "low"; cost estimate only
)

_LEAK = re.compile(
    r"\b(art\.?|artt\.?|artikel|articles?|articolo|articoli)\s*\d|\bSR\s*\d|\b\d{3}(\.\d+)*:\d",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ArticlePlan:
    article_id: str
    shown_lang: str
    slots: tuple[tuple[str, str], ...]  # (question language, style), slot number = index + 1


def _h(article_id: str, salt: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{SEED}:{salt}:{article_id}".encode()).digest()[:8], "big"
    )


def plan_article(article_id: str) -> ArticlePlan:
    shown = LANGS[_h(article_id, "shown") % 3]
    other = [lang for lang in LANGS if lang != shown]
    styles = random.Random(_h(article_id, "styles")).sample(STYLES, 4)
    langs = ["en", "en", *other]
    return ArticlePlan(article_id, shown, tuple(zip(langs, styles, strict=True)))


def prompt_hash() -> str:
    h = hashlib.sha256()
    for name in ("synth_system.txt", "synth_user.txt"):
        h.update((PROMPT_DIR / name).read_bytes())
    return h.hexdigest()


def build_request(plan: ArticlePlan, title: str, text: str, attempt: int = 0) -> LLMRequest:
    slots = "\n".join(
        f"{i}. language: {LANG_NAMES[lang]}; style: {style} ({STYLE_HINT[style]})"
        for i, (lang, style) in enumerate(plan.slots, 1)
    )
    user = (
        (PROMPT_DIR / "synth_user.txt")
        .read_text(encoding="utf-8")
        .format(
            article_language=LANG_NAMES[plan.shown_lang],
            title=title,
            text=text[:MAX_ARTICLE_CHARS],
            slots=slots,
        )
    )
    if attempt:
        user += (
            "\nYour previous reply could not be used. Reply with only the JSON object, one "
            "question per slot, no article numbers, each in the language of its slot."
        )
    return LLMRequest(
        system=(PROMPT_DIR / "synth_system.txt").read_text(encoding="utf-8"),
        user=user,
        max_output_tokens=1500,
        json_output=True,
    )


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[\W_]+", " ", text.lower()).split())


def validate_reply(text: str, plan: ArticlePlan) -> tuple[list[str] | None, str]:
    """Return ``(questions in slot order, "")`` or ``(None, reason)``."""
    obj = extract_json(text)
    items = obj.get("questions") if isinstance(obj, dict) else None
    if not isinstance(items, list) or len(items) != len(plan.slots):
        return None, "shape"
    by_slot: dict[int, str] = {}
    for it in items:
        if not isinstance(it, dict) or not isinstance(it.get("slot"), int):
            return None, "shape"
        if not isinstance(it.get("text"), str):
            return None, "shape"
        by_slot[it["slot"]] = it["text"].strip()
    if set(by_slot) != set(range(1, len(plan.slots) + 1)):
        return None, "slots"
    questions = [by_slot[i] for i in range(1, len(plan.slots) + 1)]
    for q, (_, style) in zip(questions, plan.slots, strict=True):
        n_words = len(q.split())
        if n_words < (2 if style == "keywords" else 4) or n_words > 80:
            return None, "length"
        if _LEAK.search(q):
            return None, "id_leak"
    if len({_norm(q) for q in questions}) != len(questions):
        return None, "duplicate"
    return questions, ""


def load_pool_texts(article_ids: Sequence[str]) -> dict[tuple[str, str], tuple[str, str]]:
    """``{(article_id, lang): (title, text)}`` for the pool, from the retrieval-eligible rows."""
    df = pd.read_parquet(ARTICLES_PARQUET)
    df = df[(df["kind"] == "article") & ~df["is_repealed"] & ~df["is_historical"]]
    df = df[df["article_id"].isin(set(article_ids))].drop_duplicates(["article_id", "lang"])
    return {
        (r.article_id, r.lang): (r.title or "", r.text)
        for r in df.itertuples()
        if isinstance(r.text, str) and r.text.strip()
    }


def read_pool() -> list[str]:
    path = FINETUNE_DIR / "train_pool_ids.txt"
    ids = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    return sorted(ids, key=id_key)


def _requests(ids: list[str], texts) -> list[tuple[ArticlePlan, LLMRequest]]:
    out = []
    for a in ids:
        plan = plan_article(a)
        title, text = texts[(a, plan.shown_lang)]
        out.append((plan, build_request(plan, title, text)))
    return out


def generate_one(client, plan: ArticlePlan, title: str, text: str) -> dict:
    reason = "no_reply"
    cost = 0.0
    for attempt in range(2):
        resp = client.complete(build_request(plan, title, text, attempt))
        cost += resp.usage.cost_usd
        questions, reason = validate_reply(resp.text, plan)
        if questions is not None:
            rows = [
                {
                    "article_id": plan.article_id,
                    "slot": i,
                    "query_lang": lang,
                    "style": style,
                    "shown_lang": plan.shown_lang,
                    "text": q,
                }
                for i, ((lang, style), q) in enumerate(zip(plan.slots, questions, strict=True), 1)
            ]
            return {"rows": rows, "failed": None, "cost": cost, "attempts": attempt + 1}
    return {"rows": [], "failed": reason, "cost": cost, "attempts": 2}


def _select(args) -> list[str]:
    """The sorted pool; with --limit N, N articles spread evenly over it, not the first N."""
    ids = read_pool()
    if args.limit:
        step = max(len(ids) // args.limit, 1)
        ids = ids[::step][: args.limit]
    return ids


def cmd_estimate(args) -> None:
    ids = _select(args)
    texts = load_pool_texts(ids)
    client = build_client(MODEL, transport=_OfflineTransport())
    proj = client.project([r for _, r in _requests(ids, texts)], EXPECTED_OUTPUT_TOKENS)
    langs = [lang for a in ids for lang, _ in plan_article(a).slots]
    print(
        f"articles {len(ids)}  questions {len(langs)}  model {MODEL}  prompt {prompt_hash()[:12]}"
    )
    print({lang: langs.count(lang) for lang in ("en", *LANGS)})
    print(
        f"projected live cost {proj.cost_usd:.3f} USD (assumes {EXPECTED_OUTPUT_TOKENS} output "
        f"tokens per call); budget remaining {proj.remaining_usd:.2f} USD; "
        f"needs confirmation: {proj.needs_confirmation}; "
        f"exceeds remaining: {proj.exceeds_remaining}"
    )


class _OfflineTransport:
    def call(self, spec, request):  # pragma: no cover - estimate never calls out
        raise RuntimeError("offline transport: estimate does not call a model")


def cmd_run(args) -> None:
    if not args.yes:
        raise SystemExit("paid calls: run `estimate` first, then repeat with --yes")
    ids = _select(args)
    texts = load_pool_texts(ids)
    client = build_client(MODEL)
    proj = client.project([r for _, r in _requests(ids, texts)], EXPECTED_OUTPUT_TOKENS)
    if proj.exceeds_remaining or (proj.needs_confirmation and not args.confirm_over_limit):
        raise SystemExit(f"projected {proj.cost_usd:.2f} USD needs approval or exceeds the budget")

    def work(article_id: str) -> dict:
        plan = plan_article(article_id)
        title, text = texts[(article_id, plan.shown_lang)]
        return generate_one(client, plan, title, text)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(work, ids))
    rows = [r for res in results for r in res["rows"]]
    rows.sort(key=lambda r: (id_key(r["article_id"]), r["slot"]))
    failed = {a: res["failed"] for a, res in zip(ids, results, strict=True) if res["failed"]}
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    print(
        f"articles {len(ids)}  questions {len(rows)}  failed articles {len(failed)}  "
        f"retried {sum(r['attempts'] > 1 for r in results)}  "
        f"live cost {sum(r['cost'] for r in results):.3f} USD  -> {out}"
    )
    if failed:
        (out.parent / "queries_failed.json").write_text(
            json.dumps(failed, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    est = sub.add_parser("estimate", help="projected cost, no calls")
    est.add_argument("--limit", type=int, default=0)
    run = sub.add_parser("run", help="generate (paid unless cached)")
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--yes", action="store_true")
    run.add_argument("--confirm-over-limit", action="store_true")
    run.add_argument("--workers", type=int, default=8)
    run.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.cmd == "run" and args.out is None:  # a limited run must not overwrite the full file
        args.out = FINETUNE_DIR / ("queries_sample.jsonl" if args.limit else "queries.jsonl")
    {"estimate": cmd_estimate, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
