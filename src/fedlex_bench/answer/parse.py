"""Parse and shape-check the answer object (docs/eval_spec.md section 3)."""

from __future__ import annotations

from typing import Any

from fedlex_bench.llm.jsonutil import extract_json


def _str_list(value: Any) -> list[str] | None:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        return None
    return value


def parse_answer_json(text: str) -> dict | None:
    """Return {status, answer, citations, pointers} or None if the shape is wrong.

    `pointers` may be absent (defaults to []). Ids are NOT validated here; that needs the
    candidate list and is done by the answerer.
    """
    obj = extract_json(text)
    if not isinstance(obj, dict):
        return None
    status = obj.get("status")
    if not isinstance(status, str) or status.strip().lower() not in ("answered", "refused"):
        return None
    answer = obj.get("answer")
    if not isinstance(answer, str):
        return None
    citations = _str_list(obj.get("citations"))
    pointers = _str_list(obj.get("pointers", []))
    if citations is None or pointers is None:
        return None
    return {
        "status": status.strip().lower(),
        "answer": answer,
        "citations": citations,
        "pointers": pointers,
    }
