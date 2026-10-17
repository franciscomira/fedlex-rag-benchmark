"""Cross-family check of benchmark v1 (Phase 2, Step 3).

A judge from another model family reviews every v1 item with the same rubric as the Phase 1
`question-judge` agent. That agent could search the corpus; here code supplies the corpus access:
each request carries the gold article text (all three languages) plus the best-retrieved
competing articles, so "another article answers it" and "the trap is really unanswerable" can be
checked. Output records match the machine-review sidecar format (`<judge>-<n>.jsonl`).

    uv run python -m fedlex_bench.benchmark.cross_family --model gemini-3.6-flash --dry-run
    uv run python -m fedlex_bench.benchmark.cross_family --model gemini-3.6-flash --name gemini-judge
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

from fedlex_bench.benchmark.machine_review import MACHINE_DIR, content_sha256
from fedlex_bench.types import LLMClient, LLMRequest

ISSUE_CODES = (
    "fact_unsupported", "fact_not_asked", "question_not_answered", "copies_wording",
    "other_article_answers", "ambiguous", "unnatural_or_wrong_language", "trap_answerable",
    "trap_implausible", "other",
)
MAX_CHARS = 4000
COMPETITOR_CHARS = 1500
N_COMPETITORS = 6

RUBRIC = """You are an independent reviewer of benchmark items for a Swiss federal law retrieval benchmark. You have no history with these items. Judge each item only from the material in the message.

Decide approve or reject.

For single and cross_ref items, approve only if all hold:
1. Every gold fact is stated by (or follows directly from) the text of the article it is tied to, and is true in all three language versions.
2. The question is answered by the gold article(s) alone; for cross_ref both articles are really needed. The question asks for what the facts say, and every fact answers part of the question.
3. The question is written in its stated language, reads like a real person's question, and does not just repeat the article's distinctive wording.
4. It is not ambiguous in a way that a different, reasonable reading would make the gold facts wrong. If one of the "other articles" shown answers it better or equally, reject with other_article_answers.

For traps, approve only if the other articles shown (the closest the corpus has) really do not state the answer, and the question is a plausible real-person question. A pointer article that only says where to look is fine; say so in the reason.

Be strict but fair. Reject only for a concrete, stated problem.

Reply with JSON only: {"verdict": "approve" or "reject", "issues": [codes], "reason": "one sentence, max 200 chars", "confidence": "high" or "medium" or "low"}
Issue codes: """ + ", ".join(ISSUE_CODES) + ". issues is empty when you approve."

TextLookup = Callable[[Iterable[str]], Mapping[str, str]]


def build_request(item: Mapping, texts: TextLookup, competitors: Sequence[str]) -> LLMRequest:
    """The judge sees: question, type, facts (answerable) or trap_reason, gold text, competitors."""
    parts = [
        f"Item type: {item['qtype']}",
        f"Question language: {item['query_lang']}",
        f"Question: {item['question']}",
    ]
    if item["is_trap"]:
        parts.append(f"Claimed reason the corpus does not answer it: {item['trap_reason']}")
    else:
        facts = "\n".join(f"- [{f['article_id']}] {f['text']}" for f in item["gold_facts"])
        parts.append(f"Gold facts (English), each tied to an article:\n{facts}")
        gold = texts(item["gold_article_ids"])
        parts.append("Gold article text:\n" + "\n\n".join(
            f"=== {i} ===\n{gold[i][:MAX_CHARS]}" for i in item["gold_article_ids"]))
    if competitors:
        other = texts(competitors)
        parts.append("Other articles of the corpus (closest by retrieval, excluding gold):\n" +
                     "\n\n".join(f"=== {i} ===\n{other[i][:COMPETITOR_CHARS]}" for i in competitors))
    return LLMRequest(RUBRIC, "\n\n".join(parts), max_output_tokens=600, temperature=0.0,
                      json_output=True)


def parse_verdict(text: str) -> dict | None:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict) or raw.get("verdict") not in ("approve", "reject"):
        return None
    issues = [i if i in ISSUE_CODES else "other" for i in raw.get("issues") or []]
    return {
        "verdict": raw["verdict"],
        "issues": issues,
        "reason": str(raw.get("reason", ""))[:200],
        "confidence": raw.get("confidence", "medium"),
    }


def judge_items(
    items: Sequence[Mapping],
    client: LLMClient,
    texts: TextLookup,
    competitors_of: Callable[[Mapping], Sequence[str]],
) -> tuple[list[dict], list[str]]:
    """Sidecar records for the items, and the ids whose judge output stayed unparseable."""
    records, failed = [], []
    for item in sorted(items, key=lambda i: i["id"]):
        request = build_request(item, texts, competitors_of(item))
        verdict = None
        for _ in range(2):  # one retry on unparseable output
            verdict = parse_verdict(client.complete(request).text)
            if verdict:
                break
        if verdict is None:
            failed.append(item["id"])
            continue
        records.append({"id": item["id"], **verdict, "content_sha256": content_sha256(item)})
    return records, failed


def main(argv: list[str] | None = None) -> None:
    from fedlex_bench.eval.guard import article_text_lookup
    from fedlex_bench.eval.runner import load_items
    from fedlex_bench.eval.__main__ import BENCH, PARQUET
    from fedlex_bench.llm import build_client
    from fedlex_bench.retrieval import RetrievalEnv, retrieve_articles

    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--name", default="crossfamily-judge", help="sidecar judge name")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args(argv)

    items = load_items(BENCH / "dev.jsonl") + load_items(BENCH / "test.jsonl")
    items = sorted(items, key=lambda i: i["id"])[: args.limit or None]
    retriever = RetrievalEnv().retriever("hybrid", "all_lang", "article")

    def competitors_of(item: Mapping) -> list[str]:
        ids = retrieve_articles(retriever, item["question"], item["query_lang"], 3 * N_COMPETITORS)
        gold = set(item["gold_article_ids"])
        return [i for i in ids if i not in gold][:N_COMPETITORS]

    texts = article_text_lookup(PARQUET)
    client = build_client(args.model)
    requests = [build_request(i, texts, competitors_of(i)) for i in items]
    proj = client.project(requests, expected_output_tokens=150)
    print(f"{len(requests)} requests; projected {proj.cost_usd:.2f} USD "
          f"(remaining {proj.remaining_usd:.2f}; confirm needed: {proj.needs_confirmation})")
    if args.dry_run or proj.exceeds_remaining or proj.needs_confirmation:
        return
    records, failed = judge_items(items, client, texts, competitors_of)
    out = MACHINE_DIR / f"{args.name}-1.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=True) + "\n" for r in records), "utf-8")
    print(f"wrote {len(records)} verdicts to {out}; unparseable: {failed}")


if __name__ == "__main__":
    main()
