"""Per-run mutable state shared by the ADK agents of one question, plus small helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from fedlex_bench.agent.trace import TraceRecorder
from fedlex_bench.types import AnswerResult, Candidate, Lang, QueryLang, Usage

LANGS: tuple[Lang, ...] = ("de", "fr", "it")
Tool = Literal["search", "get_article", "follow_reference"]
TOOLS: tuple[str, ...] = ("search", "get_article", "follow_reference")
_ID_RE = re.compile(r"^[0-9][0-9.]*:[0-9]+[a-z]*(?:bis|ter|quater|quinquies|sexies)?$", re.I)


@dataclass(frozen=True)
class Action:
    """One tool call the planner asks a language retriever to perform."""

    tool: Tool
    lang: Lang  # which per-language retriever sub-agent executes it
    query: str = ""  # search only
    article_id: str = ""  # get_article / follow_reference
    k: int = 8
    query_lang: QueryLang | None = None  # language of `query`; defaults to `lang`


def valid_article_id(value: object) -> bool:
    return isinstance(value, str) and bool(_ID_RE.match(value.strip()))


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from model text (tolerates code fences and prose)."""
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_actions(
    obj: dict[str, Any] | None, *, default_k: int, max_actions: int
) -> tuple[list[Action], bool]:
    """Validate planner actions. Returns (actions, done). Invalid entries are dropped."""
    if not obj:
        return [], False
    actions: list[Action] = []
    seen: set[tuple[Any, ...]] = set()
    raw = obj.get("actions")
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        tool, lang = item.get("tool"), item.get("lang")
        if tool not in TOOLS or lang not in LANGS:
            continue
        try:
            k = max(1, min(20, int(item.get("k", default_k))))
        except (TypeError, ValueError):
            k = default_k
        if tool == "search":
            query = str(item.get("query", "")).strip()
            if not query:
                continue
            act = Action("search", lang, query=query, k=k, query_lang=lang)
        else:
            art = str(item.get("id", item.get("article_id", ""))).strip()
            if not valid_article_id(art):
                continue
            act = Action(tool, lang, article_id=art, k=k)
        key = (act.tool, act.lang, act.query, act.article_id)
        if key in seen:
            continue
        seen.add(key)
        actions.append(act)
        if len(actions) >= max_actions:
            break
    return actions, bool(obj.get("done", False))


class Pool:
    """Evidence pool: best candidate per (article_id, lang), ranked across all tool calls."""

    def __init__(self) -> None:
        self._cands: dict[tuple[str, str], Candidate] = {}
        self._rank: dict[tuple[str, str], tuple[int, float, int]] = {}
        self._order = 0

    def add(self, candidates: list[Candidate], *, boost: bool = False) -> list[str]:
        """Add candidates of one tool call; returns article ids new to the pool."""
        before = self.article_ids()
        for pos, cand in enumerate(candidates):
            key = (cand.article_id, cand.lang)
            rank = -1 if boost else pos
            old = self._rank.get(key)
            if old is None:
                self._cands[key] = cand
                self._rank[key] = (rank, -cand.score, self._order)
                self._order += 1
            elif (rank, -cand.score) < old[:2]:
                self._cands[key] = cand
                self._rank[key] = (rank, -cand.score, old[2])
        return [a for a in self.article_ids() if a not in before]

    def ordered(self) -> list[Candidate]:
        keys = sorted(self._cands, key=lambda key: self._rank[key])
        return [self._cands[k] for k in keys]

    def article_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for cand in self.ordered():
            seen.setdefault(cand.article_id)
        return list(seen)

    def top(self, n: int) -> list[Candidate]:
        return self.ordered()[:n]

    def get(self, article_id: str, lang: str | None = None) -> list[Candidate]:
        found = [c for c in self.ordered() if c.article_id == article_id]
        if lang:
            same = [c for c in found if c.lang == lang]
            return same or found
        return found

    def texts_for(self, article_ids: list[str]) -> dict[str, list[Candidate]]:
        return {a: self.get(a) for a in article_ids}


@dataclass
class RunState:
    """Everything the agents of one run read and write. Not shared across runs."""

    query: str
    query_lang: QueryLang
    trace: TraceRecorder
    pool: Pool = field(default_factory=Pool)
    usage: Usage = field(default_factory=Usage)
    round: int = 0
    tool_calls: int = 0
    plan: list[Action] = field(default_factory=list)
    planner_done: bool = False
    feedback: str = ""
    past_queries: list[str] = field(default_factory=list)
    new_ids_per_round: list[int] = field(default_factory=list)
    draft: AnswerResult | None = None
    verified: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    missing_hint: str = ""
    final: AnswerResult | None = None
    stop_reason: str = ""
    step_limit_hit: bool = False
    refs_seen: set[str] = field(default_factory=set)

    def add_usage(self, usage: Usage) -> None:
        self.usage = self.usage + usage


_REF_MARK = re.compile(
    r"\b(?:Art(?:ikel|t)?\.?|art(?:icles?|\.)|articol[oi])\s+"
    r"(\d+[a-z]{0,9}(?:\s*(?:,|und|et|e|bis|à|a|-|–)\s*\d+[a-z]{0,9})*)",
    re.I,
)
_NUM = re.compile(r"\d+[a-z]{0,9}", re.I)
_EXTERNAL = re.compile(
    r"^(?:\s+(?:Abs\.?|Absatz|al\.|alinéa|cpv\.|Bst\.|lit\.|Ziff\.|ch\.|let\.|n\.)\s*\d+\w*)*"
    r"\s+(?:des|der|du|de la|de l'|della|dello|del|dei)\s*(\w+)",
    re.I,
)
_SAME_ACT = {
    "vorliegenden", "vorliegende", "gleichen", "selben", "dieses", "présente", "presente",
    "même", "meme", "medesimo", "stessa", "dieser", "ce", "cette", "questa", "questo",
}  # fmt: skip


def extract_references(text: str, source_id: str) -> list[str]:
    """Best-effort same-act article references in `text` as "<SR>:<n>" ids.

    References that the following words attach to another named act ("Art. 5 des Zivilgesetzbuches")
    are skipped; this is heuristic and may miss or over-include. Never returns `source_id`.
    """
    sr = source_id.split(":", 1)[0]
    out: dict[str, None] = {}
    for m in _REF_MARK.finditer(text or ""):
        tail = text[m.end() : m.end() + 80]
        ext = _EXTERNAL.match(tail)
        if ext and ext.group(1).lower() not in _SAME_ACT:
            continue
        for num in _NUM.findall(m.group(1)):
            ref = f"{sr}:{num.lower()}"
            if ref != source_id:
                out.setdefault(ref)
    return list(out)
