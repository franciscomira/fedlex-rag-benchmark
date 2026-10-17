"""LLM judge roles (docs/eval_spec.md section 5). The client is injected; prompts are hashed."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fedlex_bench.types import LLMClient, LLMRequest, Usage

PROMPT_DIR = Path(__file__).parent / "prompts"
ROLES = ("facts", "faithfulness", "trap")
MAX_ARTICLE_CHARS = 6000


def _prompt(role: str) -> str:
    return (PROMPT_DIR / f"{role}.txt").read_text(encoding="utf-8").strip()


def prompt_hashes() -> dict[str, str]:
    return {r: hashlib.sha256(_prompt(r).encode("utf-8")).hexdigest()[:16] for r in ROLES}


@dataclass(frozen=True)
class FactResult:
    verdicts: list[bool]
    score: float | None  # None if the judge output could not be parsed
    failed: bool = False
    usage: Usage = field(default_factory=Usage)


@dataclass(frozen=True)
class FaithResult:
    score: float | None  # None: no claims (excluded from the mean) or failure
    n_claims: int = 0
    claims: list[dict[str, Any]] = field(default_factory=list)
    failed: bool = False
    usage: Usage = field(default_factory=Usage)


@dataclass(frozen=True)
class TrapResult:
    leaked: bool | None
    failed: bool = False
    usage: Usage = field(default_factory=Usage)


class Judge:
    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def _ask(self, role: str, user: str) -> tuple[dict | None, Usage]:
        """One call, one retry on unparseable output. Returns (parsed or None, usage)."""
        usage = Usage()
        for _ in range(2):
            resp = self.client.complete(
                LLMRequest(system=_prompt(role), user=user, temperature=0.0, json_output=True)
            )
            usage = usage + resp.usage
            try:
                parsed = json.loads(resp.text)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed, usage
        return None, usage

    def fact_recall(self, question: str, answer: str, gold_facts: Sequence[Mapping]) -> FactResult:
        listed = "\n".join(f"{i}. {f['text']}" for i, f in enumerate(gold_facts))
        user = f"Question:\n{question}\n\nAnswer:\n{answer}\n\nGold facts:\n{listed}"
        parsed, usage = self._ask("facts", user)
        if parsed is None or not isinstance(parsed.get("facts"), list):
            return FactResult([], None, True, usage)
        said = {e.get("index"): e.get("supported") is True for e in parsed["facts"]
                if isinstance(e, dict)}
        verdicts = [said.get(i, False) for i in range(len(gold_facts))]
        return FactResult(verdicts, sum(verdicts) / len(verdicts), False, usage)

    def faithfulness(self, answer: str, cited_texts: Mapping[str, str]) -> FaithResult:
        arts = "\n\n".join(
            f"[{aid}]\n{text[:MAX_ARTICLE_CHARS]}" for aid, text in sorted(cited_texts.items())
        )
        parsed, usage = self._ask("faithfulness", f"Answer:\n{answer}\n\nCited articles:\n{arts}")
        if parsed is None or not isinstance(parsed.get("claims"), list):
            return FaithResult(None, 0, [], True, usage)
        claims = [c for c in parsed["claims"] if isinstance(c, dict)]
        if not claims:
            return FaithResult(None, 0, [], False, usage)
        ok = sum(c.get("supported") is True for c in claims)
        return FaithResult(ok / len(claims), len(claims), claims, False, usage)

    def trap_leak(self, question: str, answer: str, trap_reason: str) -> TrapResult:
        user = f"Question:\n{question}\n\nWhy it is unanswerable:\n{trap_reason}\n\nReply:\n{answer}"
        parsed, usage = self._ask("trap", user)
        if parsed is None or not isinstance(parsed.get("asserts_answer"), bool):
            return TrapResult(None, True, usage)
        return TrapResult(parsed["asserts_answer"], False, usage)
