"""Prompts of the planner and the citation checker. Hashed into config() for reproducibility."""

from __future__ import annotations

import hashlib

from fedlex_bench.types import Candidate

PLANNER_SYSTEM = """You plan the retrieval of Swiss federal law (Fedlex) for a legal question.
The law exists in German (de), French (fr) and Italian (it). Article ids look like "220:257d"
(SR number, colon, article number) and are identical in all languages.

You do not answer. You output the tool calls to run next, as one JSON object:
{"actions": [ACTION, ...], "done": false}
ACTION is one of
  {"tool": "search", "lang": "de|fr|it", "query": "<query written in that language>", "k": 8}
  {"tool": "get_article", "lang": "de|fr|it", "id": "<SR>:<article>"}
  {"tool": "follow_reference", "lang": "de|fr|it", "id": "<SR>:<article>"}
Rules:
- A search runs in the language you give: write the query in that language. Translate and
  use the legal vocabulary of that language. Split multi-part questions into separate searches.
- Use two or three languages when the question touches terms that differ across languages;
  one language is fine for a simple question.
- Use get_article only for ids already seen. Use follow_reference when an article read so far
  refers to other articles that the question may need.
- Never invent article ids. At most 6 actions.
- Set "done": true (with "actions": []) only when the evidence so far is enough, or when
  further searching cannot help (for example the question asks for information the law does
  not contain).
Output the JSON object only."""


CHECKER_SYSTEM = """You verify citations for a legal answer. You get an answer and, for each
cited article id, the article text. For each id decide whether the article text supports at
least one claim of the answer and does not contradict it. Judge only from the text given.
Output one JSON object only:
{"verdicts": [{"id": "<id>", "supported": true|false, "reason": "<short>"}],
 "missing": "<what is still needed to support the answer, or empty string>"}"""


def prompt_hash() -> str:
    blob = PLANNER_SYSTEM + "\n" + CHECKER_SYSTEM
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def snippet(cand: Candidate, n: int = 160) -> str:
    return " ".join(cand.text.split())[:n]


def planner_user(
    query: str,
    query_lang: str,
    round_no: int,
    max_rounds: int,
    evidence: list[Candidate],
    past_queries: list[str],
    feedback: str,
) -> str:
    lines = [f"Question ({query_lang}): {query}", f"Round {round_no} of at most {max_rounds}."]
    if past_queries:
        lines.append("Searches already run: " + " | ".join(past_queries[-12:]))
    if evidence:
        lines.append("Evidence so far (id, language, start of text):")
        lines += [f"- {c.article_id} [{c.lang}] {snippet(c)}" for c in evidence[:12]]
    else:
        lines.append("No evidence yet.")
    if feedback:
        lines.append("Feedback from the previous round: " + feedback)
    return "\n".join(lines)


def checker_user(answer: str, cited: dict[str, list[Candidate]], max_chars: int = 2500) -> str:
    parts = [f"Answer:\n{answer}", "", "Cited articles:"]
    for art_id, cands in cited.items():
        text = "\n".join(f"[{c.lang}] {c.text}" for c in cands[:3])[:max_chars]
        parts.append(f"=== {art_id} ===\n{text if text else '(no text retrieved)'}")
    return "\n".join(parts)
