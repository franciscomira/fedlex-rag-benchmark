"""On-disk response cache keyed by (provider, model id, prompt, params).

A hit replays the stored usage (tokens, cost, latency) so reports are stable on re-runs.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from fedlex_bench.llm.config import repo_root
from fedlex_bench.types import LLMRequest, Usage


def default_cache_dir() -> Path:
    return repo_root() / "eval" / "cache"


def request_key(provider: str, model_id: str, request: LLMRequest, extra: dict) -> str:
    """sha256 over a canonical JSON of everything that can change the response."""
    payload = {
        "provider": provider,
        "model_id": model_id,
        "system": request.system,
        "user": request.user,
        "max_output_tokens": request.max_output_tokens,
        "temperature": request.temperature,
        "json_output": request.json_output,
        "extra": extra,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheEntry:
    text: str
    model_id: str
    usage: Usage


class DiskCache:
    def __init__(self, directory: Path | None = None) -> None:
        self.directory = Path(directory) if directory else default_cache_dir()

    def _path(self, key: str) -> Path:
        return self.directory / "responses" / key[:2] / f"{key}.json"

    def has(self, key: str) -> bool:
        return self.get(key) is not None

    def get(self, key: str) -> CacheEntry | None:
        path = self._path(key)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return CacheEntry(data["text"], data["model_id"], Usage(**data["usage"]))
        except (OSError, ValueError, KeyError, TypeError):
            return None  # absent or corrupt: treated as a miss

    def put(self, key: str, *, text: str, model_id: str, usage: Usage) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(
            json.dumps(
                {"text": text, "model_id": model_id, "usage": asdict(usage)}, ensure_ascii=False
            ),
            encoding="utf-8",
        )
        os.replace(tmp, path)  # atomic: a crash never leaves a half-written entry
