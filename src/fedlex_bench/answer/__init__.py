"""Cited answer generation (docs/eval_spec.md section 3)."""

from fedlex_bench.answer.answerer import LLMAnswerer, prompt_hash
from fedlex_bench.answer.parse import parse_answer_json

__all__ = ["LLMAnswerer", "parse_answer_json", "prompt_hash"]
