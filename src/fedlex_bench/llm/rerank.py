"""Listwise LLM reranker.

Why listwise rather than pointwise: one call per query (latency and cost scale with the query,
not with the candidate count: pointwise over 30 candidates is 30 calls repeating the question),
and the model compares passages directly, which is what choosing the specific provision over a
general one needs. Costs: position bias and a long prompt (bounded by `max_candidates` and
`max_chars_per_passage`). The model ranks by passage *number*; code maps numbers back to
candidates, so ids and text can never be invented or altered by the model.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

from fedlex_bench.llm.jsonutil import extract_json
from fedlex_bench.types import (
    Candidate,
    LLMClient,
    LLMRequest,
    QueryLang,
    RetrievalResult,
    Usage,
)

PROMPT_DIR = Path(__file__).with_name("prompts")
LANG_NAMES = {"en": "English", "de": "German", "fr": "French", "it": "Italian"}
_RETRY_NOTE = (
    "\n\nYour previous reply was not a valid JSON object of the form "
    '{"ranking": [<numbers>]}. Reply with only that JSON object.'
)


def prompt_hash() -> str:
    """sha256 over the reranker prompt files (goes into config.json)."""
    h = hashlib.sha256()
    for name in ("rerank_system.txt", "rerank_user.txt"):
        h.update((PROMPT_DIR / name).read_bytes())
    return h.hexdigest()


class LLMReranker:
    def __init__(
        self,
        client: LLMClient,
        *,
        max_candidates: int = 30,
        max_chars_per_passage: int = 1500,
        max_output_tokens: int = 1024,
    ) -> None:
        self.client = client
        self.name = f"llm_rerank/{client.model_id}"
        self.max_candidates = max_candidates
        self.max_chars = max_chars_per_passage
        self.max_output_tokens = max_output_tokens
        self.n_parse_failures = 0  # queries that fell back to the input order
        self._system = (PROMPT_DIR / "rerank_system.txt").read_text(encoding="utf-8")
        self._user = (PROMPT_DIR / "rerank_user.txt").read_text(encoding="utf-8")

    def rerank(
        self, query: str, query_lang: QueryLang, candidates: list[Candidate], k: int
    ) -> RetrievalResult:
        if not candidates:
            return RetrievalResult([], Usage())
        window = candidates[: self.max_candidates]
        tail = candidates[self.max_candidates :]
        passages = "\n\n".join(
            f"[{i}] ({c.lang.upper()}) {c.text[: self.max_chars]}" for i, c in enumerate(window, 1)
        )
        user = self._user.format(
            language=LANG_NAMES[query_lang],
            question=query,
            passages=passages,
            k=min(k, len(window)),
        )
        usage = Usage()
        order: list[int] | None = None
        for attempt in range(2):
            req = LLMRequest(
                system=self._system,
                user=user + (_RETRY_NOTE if attempt else ""),
                max_output_tokens=self.max_output_tokens,
                json_output=True,
            )
            resp = self.client.complete(req)
            usage = usage + resp.usage
            order = _parse_ranking(resp.text, len(window))
            if order is not None:
                break
        if order is None:
            self.n_parse_failures += 1
            order = []  # fall back to the retriever's order
        seen = set(order)
        order = order + [i for i in range(len(window)) if i not in seen]
        ranked = [window[i] for i in order] + tail
        top = ranked[:k]
        n = max(len(top), 1)
        top = [replace(c, score=(n - r) / n) for r, c in enumerate(top)]
        return RetrievalResult(top, usage)


def _parse_ranking(text: str, n_candidates: int) -> list[int] | None:
    """0-based indices, first occurrence wins, anything not a valid number is dropped."""
    obj = extract_json(text)
    if not isinstance(obj, dict) or not isinstance(obj.get("ranking"), list):
        return None
    out: list[int] = []
    for v in obj["ranking"]:
        if isinstance(v, bool) or not isinstance(v, int):
            continue
        if 1 <= v <= n_candidates and v - 1 not in out:
            out.append(v - 1)
    return out
