"""Cited answer generation with code-side validation of every article id."""

from __future__ import annotations

import hashlib
from pathlib import Path

from fedlex_bench.answer.parse import parse_answer_json
from fedlex_bench.llm.rerank import LANG_NAMES
from fedlex_bench.types import AnswerResult, Candidate, LLMClient, LLMRequest, QueryLang, Usage

PROMPT_DIR = Path(__file__).with_name("prompts")
PROMPT_FILES = ("answer_system.txt", "answer_user.txt", "retry_note.txt")


def prompt_hash() -> str:
    """sha256 over the answer prompt files (goes into config.json)."""
    h = hashlib.sha256()
    for name in PROMPT_FILES:
        h.update((PROMPT_DIR / name).read_bytes())
    return h.hexdigest()


def _dedupe(ids: list[str]) -> list[str]:
    return list(dict.fromkeys(i.strip() for i in ids))


class LLMAnswerer:
    """Prompts a model with the numbered candidates and validates what comes back.

    - `citations` / `pointers` only ever contain ids that are in `candidates`.
    - Any other id the model produced (citation or pointer) lands in `invalid_citations`.
    - Unparseable output is retried once; if it fails again the result has `parse_failed=True`
      (status "refused", empty answer). Scorers must check `parse_failed` before `status`.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        max_chars_per_candidate: int = 4000,
        max_output_tokens: int = 2048,
    ) -> None:
        self.client = client
        self.name = f"answer/{client.model_id}"
        self.max_chars = max_chars_per_candidate
        self.max_output_tokens = max_output_tokens
        self._system = (PROMPT_DIR / "answer_system.txt").read_text(encoding="utf-8")
        self._user = (PROMPT_DIR / "answer_user.txt").read_text(encoding="utf-8")
        self._retry = (PROMPT_DIR / "retry_note.txt").read_text(encoding="utf-8")

    def _passages(self, candidates: list[Candidate]) -> str:
        if not candidates:
            return "(no passages were retrieved)"
        return "\n\n".join(
            f"[{i}] id: {c.article_id} | language: {c.lang}\n{c.text[: self.max_chars]}"
            for i, c in enumerate(candidates, 1)
        )

    def answer(
        self, query: str, query_lang: QueryLang, candidates: list[Candidate]
    ) -> AnswerResult:
        user = self._user.format(
            language=LANG_NAMES[query_lang], question=query, passages=self._passages(candidates)
        )
        usage = Usage()
        parsed = None
        for attempt in range(2):
            resp = self.client.complete(
                LLMRequest(
                    system=self._system,
                    user=user + (self._retry if attempt else ""),
                    max_output_tokens=self.max_output_tokens,
                    json_output=True,
                )
            )
            usage = usage + resp.usage
            parsed = parse_answer_json(resp.text)
            if parsed is not None:
                break
        if parsed is None:
            return AnswerResult("refused", "", [], [], [], parse_failed=True, usage=usage)

        allowed = {c.article_id for c in candidates}
        citations = _dedupe(parsed["citations"])
        pointers = _dedupe(parsed["pointers"])
        invalid = [i for i in citations + pointers if i not in allowed]
        return AnswerResult(
            status=parsed["status"],
            answer=parsed["answer"],
            citations=[i for i in citations if i in allowed],
            pointers=[i for i in pointers if i in allowed],
            invalid_citations=list(dict.fromkeys(invalid)),
            usage=usage,
        )
