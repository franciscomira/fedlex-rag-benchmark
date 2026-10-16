# Translator prompt (used for the 72 translated variants)

Run as general-purpose Claude Code subagents with the Sonnet model alias, one per input chunk (`in-1.jsonl` to `in-6.jsonl`, 12 rows each). The agents did not read any other repository file.

> Translate benchmark questions. Input: benchmark/translations/in-N.jsonl (12 rows: id, question, source_lang, target_lang). Translate each question faithfully from source_lang into target_lang (en, de, fr or it): keep exactly the same meaning, situation, conditions and level of detail; natural everyday wording as a layperson would write; do not add or drop conditions; do not add article numbers or legal citations; keep length within about 20 percent of the original and under 400 characters. Output: benchmark/translations/drafts/out-N.jsonl, one JSON line per input row in the same order: {"id": <input id>, "target_lang": ..., "question": <translation>}.

The translations were then checked for fidelity by the fresh-context judge (`.claude/agents/question-judge.md`, extra instruction: reject if the translation changes the meaning, adds or drops a condition or fact, or is unnatural). No human has reviewed the translations.
