"""Record the "Recorded full pipeline" examples for the demo page.

    uv run python -m fedlex_bench.serve.record_examples            # shows the plan, calls nothing
    uv run python -m fedlex_bench.serve.record_examples --yes      # runs it (needs a key)

The public demo is retrieval only. This script runs the full pipeline (tuned retrieval, LLM
rerank, cited answer) once per example question and saves the results to demo/examples.json,
which the page shows as pre-recorded answers. It costs about 0.002 USD per question, so about
0.02 USD for the ten below. The file holds questions, the drafted answers and article ids only;
it never holds statute text.

If any question fails, nothing is written, so the file is never half old and half new.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fedlex_bench.serve.pipeline import HUB_REVISION, LLM_MODEL, ServingPipeline
from fedlex_bench.serve.web import DEFAULT_EXAMPLES

COST_PER_QUESTION_USD = 0.002  # rough, from the Phase 3 runs; the real cost is recorded per item

# Two or three questions per language, on the Code of Obligations (SR 220), plus two that the
# covered acts cannot answer. `expected_status` is what we hope for; the file records what the
# pipeline actually did, and the page shows that, not the hope.
EXAMPLE_QUESTIONS: list[dict[str, str]] = [
    # English
    {
        "lang": "en",
        "expected_status": "answered",
        "question": "Can a landlord terminate a residential lease without giving a reason?",
    },
    {
        "lang": "en",
        "expected_status": "answered",
        "question": (
            "What notice period applies when an employer ends an open-ended employment "
            "contract after the probation period?"
        ),
    },
    {
        "lang": "en",
        "expected_status": "answered",
        "question": (
            "How quickly must a buyer inspect the goods and tell the seller about defects?"
        ),
    },
    # German
    {
        "lang": "de",
        "expected_status": "answered",
        "question": (
            "Darf der Vermieter den Mietzins während des Mietverhältnisses einseitig erhöhen?"
        ),
    },
    {
        "lang": "de",
        "expected_status": "answered",
        "question": "Wann haftet der Verkäufer für Mängel der verkauften Sache?",
    },
    {
        "lang": "de",
        "expected_status": "refused",  # value added tax is not in the covered acts
        "question": "Wie hoch ist der Mehrwertsteuersatz auf Lebensmitteln in der Schweiz?",
    },
    # French
    {
        "lang": "fr",
        "expected_status": "answered",
        "question": (
            "Quel est le délai de congé d'un contrat de travail de durée indéterminée "
            "pendant la première année de service ?"
        ),
    },
    {
        "lang": "fr",
        "expected_status": "answered",
        "question": "Le locataire peut-il sous-louer son appartement sans l'accord du bailleur ?",
    },
    # Italian
    {
        "lang": "it",
        "expected_status": "answered",
        "question": "Quale forma è richiesta per un contratto di fideiussione?",
    },
    {
        "lang": "it",
        "expected_status": "refused",  # naturalisation law is not in the covered acts
        "question": (
            "Quali sono le condizioni per ottenere la cittadinanza svizzera per naturalizzazione?"
        ),
    },
]


def record(
    pipeline: Any,
    questions: list[dict[str, str]] = EXAMPLE_QUESTIONS,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Run `pipeline.ask` once per question and return the file content (nothing is written)."""
    items = []
    for ex in questions:
        res = pipeline.ask(ex["question"], ex["lang"])
        items.append(
            {
                "lang": ex["lang"],
                "question": ex["question"],
                "expected_status": ex["expected_status"],
                "status": res.status,
                "answer": res.answer,
                "citations": [h.article_id for h in res.citations],
                "pointers": [h.article_id for h in res.pointers],
                "top_ranked": [h.article_id for h in res.hits[:5]],
                "n_invalid_ids": res.n_invalid_ids,
                "parse_failed": res.parse_failed,
                "cost_usd": round(res.cost_usd, 6),
                "latency_s": round(res.latency_s, 2),
            }
        )
    stamp = (now or datetime.now(UTC)).date().isoformat()
    return {
        "recorded_at": stamp,
        "model": LLM_MODEL,
        "embedder_revision": HUB_REVISION,
        "total_cost_usd": round(sum(i["cost_usd"] for i in items), 6),
        "examples": items,
    }


def write_examples(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--yes", action="store_true", help="really call the API (costs money)")
    ap.add_argument("--out", type=Path, default=DEFAULT_EXAMPLES)
    args = ap.parse_args(argv)

    n = len(EXAMPLE_QUESTIONS)
    print(f"{n} questions, about {n * COST_PER_QUESTION_USD:.3f} USD with {LLM_MODEL}.")
    print(f"Output: {args.out}")
    if not args.yes:
        for ex in EXAMPLE_QUESTIONS:
            print(f"  [{ex['lang']}] {ex['question']}")
        print("Dry run: nothing was called. Add --yes to run it.")
        return 0

    from dotenv import load_dotenv

    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set (put it in .env). Nothing was called.", file=sys.stderr)
        return 1
    data = record(ServingPipeline())
    write_examples(data, args.out)
    print(f"Wrote {len(data['examples'])} examples, cost {data['total_cost_usd']:.4f} USD.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
