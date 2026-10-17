"""Tolerant extraction of one JSON object from model text."""

from __future__ import annotations

import json
from typing import Any


def extract_json(text: str) -> Any | None:
    """Return the first JSON object found in `text` (fenced, or surrounded by prose), else None."""
    if not text:
        return None
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(text, start)
            return obj
        except ValueError:
            start = text.find("{", start + 1)
    return None
