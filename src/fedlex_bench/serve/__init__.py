"""Serving layer: the measured pipeline behind the demo and the MCP server.

Names are imported lazily so that `from fedlex_bench.serve import notices` (the results page, the
tests) does not load pandas and the retrieval stack.
"""

from __future__ import annotations

from importlib import import_module

_LAZY = {
    "NOT_LEGAL_ADVICE": "notices",
    "NOT_OFFICIAL_DE": "notices",
    "NOT_OFFICIAL_EN": "notices",
    "notice_text": "notices",
    "ArticleHit": "pipeline",
    "AskResult": "pipeline",
    "Corpus": "pipeline",
    "ServingPipeline": "pipeline",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    if name in _LAZY:
        return getattr(import_module(f"fedlex_bench.serve.{_LAZY[name]}"), name)
    raise AttributeError(f"module 'fedlex_bench.serve' has no attribute {name!r}")
