"""Synthetic corpora and items for the benchmark tests (no real Fedlex text)."""

from __future__ import annotations

import pandas as pd

from fedlex_bench.benchmark.common import ACT_QUOTA, split_of

LANGS = ("de", "fr", "it")
COLUMNS = [
    "sr",
    "article_id",
    "lang",
    "title",
    "text",
    "source_url",
    "consolidation_date",
    "kind",
    "is_repealed",
    "is_historical",
    "block_heading",
    "footnotes",
]
# Articles per act of the synthetic corpus: big enough that every act x split cell can be filled.
SIZES = {
    "210": 220,
    "220": 420,
    "221.214.1": 70,
    "221.229.1": 90,
    "235.1": 90,
    "241": 70,
    "272": 130,
    "281.1": 130,
}
REF = {"de": "Siehe Artikel {n}.", "fr": "Voir l'article {n}.", "it": "Vedi l'articolo {n}."}


def _row(sr: str, number: str, lang: str, text: str, **kw) -> dict:
    row = {
        "sr": sr,
        "article_id": f"{sr}:{number}",
        "lang": lang,
        "title": f"Titel {number}",
        "text": text,
        "source_url": f"https://x/{lang}.xml#art_{number}",
        "consolidation_date": "2026-10-01",
        "kind": "article",
        "is_repealed": False,
        "is_historical": False,
        "block_heading": "",
        "footnotes": "",
    }
    row.update(kw)
    return row


def make_corpus(references: bool = True, sizes: dict[str, int] | None = None) -> pd.DataFrame:
    """Eligible articles ``1..n`` per act in DE, FR and IT. Every even article i > 2 refers to
    article i-1 of the same act in all three languages. Extra ids that must never be chosen:
    ``<sr>:r1`` (repealed), ``<sr>:s1`` (only 60 characters) and ``<sr>:u1/1`` (transitional)."""
    rows: list[dict] = []
    for sr, n in (sizes or SIZES).items():
        for i in range(1, n + 1):
            for lang in LANGS:
                text = f"{lang} wortlaut {sr} nummer {i} " + f"inhalt{i} " * 40
                if references and i > 2 and i % 2 == 0:
                    text += REF[lang].format(n=i - 1)
                rows.append(_row(sr, str(i), lang, text))
        for lang in LANGS:
            rows.append(_row(sr, "r1", lang, "", is_repealed=True))
            rows.append(_row(sr, "s1", lang, "kurz " * 12))
            rows.append(_row(sr, "u1/1", lang, "uebergang " * 60, kind="transitional"))
    return pd.DataFrame(rows, columns=COLUMNS)


def make_valid_items(df: pd.DataFrame) -> list[dict]:
    """A full, valid benchmark (150 answerable + 20 traps) over ``df``: gold ids taken by code
    from the corpus, splits from the rule, ranks from the BM25 module, the language balance
    38/37/38/37, source languages as in the spec, act quotas exactly as in the table."""
    from fedlex_bench.benchmark.bm25 import build_indexes, compute_ranks
    from fedlex_bench.benchmark.common import (
        DEV_ANSWERABLE,
        LANG_FINAL,
        QUERY_LANGS,
        TRAP_FINAL,
        TRAP_TYPES,
        largest_remainder,
    )
    from fedlex_bench.eligibility import benchmark_eligible

    pool = list(benchmark_eligible(df)["article_id"])
    indexes = build_indexes(df)
    by_cell: dict[tuple[str, str], list[str]] = {}
    for aid in pool:
        by_cell.setdefault((aid.partition(":")[0], split_of(aid)), []).append(aid)

    dev_by_act = largest_remainder(DEV_ANSWERABLE, ACT_QUOTA, order=list(ACT_QUOTA))
    langs_cycle = [lang for lang in QUERY_LANGS for _ in range(LANG_FINAL[lang])]
    # interleave so that each act gets a spread of languages
    order = sorted(range(len(langs_cycle)), key=lambda i: (i % 38, i))
    langs = [langs_cycle[i] for i in order]
    counters = {lang: 0 for lang in QUERY_LANGS}
    items: list[dict] = []
    n = 0
    for act, quota in ACT_QUOTA.items():
        for split, count in (("dev", dev_by_act[act]), ("test", quota - dev_by_act[act])):
            for gold in by_cell[(act, split)][:count]:
                lang = langs[n]
                n += 1
                k = counters[lang]
                counters[lang] += 1
                if lang == "en":
                    source = ("de", "fr", "it")[k % 3]
                elif k % 4 == 3:
                    source = lang
                else:
                    source = [x for x in LANGS if x != lang][k % 2]
                question = words(f"{gold}:{lang}") + "?"
                ranks, overlap = compute_ranks(question, [gold], lang, indexes)
                items.append(
                    {
                        "id": "",
                        "split": split,
                        "qtype": "single",
                        "is_trap": False,
                        "query_lang": lang,
                        "question": question,
                        "source_lang": source,
                        "gold_article_ids": [gold],
                        "gold_facts": [{"text": f"Fact for {gold}.", "article_id": gold}],
                        "source_act": act,
                        "difficulty": ("easy", "medium", "hard")[n % 3],
                        "lexical_overlap": overlap,
                        "bm25_rank": ranks,
                        "trap_type": None,
                        "trap_reason": None,
                        "translation_of": None,
                        "provenance": provenance(f"s-{n:04d}"),
                        "review_status": "unreviewed",
                        "reviewed_by": None,
                        "reviewed_at": None,
                        "review_note": None,
                    }
                )
    t = 0
    for split, count in TRAP_FINAL.items():
        for _ in range(count):
            lang = QUERY_LANGS[t % 4]
            items.append(
                {
                    "id": "",
                    "split": split,
                    "qtype": "trap",
                    "is_trap": True,
                    "query_lang": lang,
                    "question": words(f"trap-{t}:{lang}") + "?",
                    "source_lang": None,
                    "gold_article_ids": [],
                    "gold_facts": [],
                    "source_act": None,
                    "difficulty": None,
                    "lexical_overlap": None,
                    "bm25_rank": None,
                    "trap_type": TRAP_TYPES[t % 4],
                    "trap_reason": "No act regulates this.",
                    "translation_of": None,
                    "provenance": provenance(f"s-{500 + t:04d}"),
                    "review_status": "unreviewed",
                    "reviewed_by": None,
                    "reviewed_at": None,
                    "review_note": None,
                }
            )
            t += 1
    items.sort(
        key=lambda it: (
            it["split"],
            it["source_act"] is None,
            it["source_act"] or "",
            it["question"],
        )
    )
    for i, item in enumerate(items, 1):
        item["id"] = f"v1-{i:04d}"
    return items


def words(seed: str, n: int = 8) -> str:
    """Pseudo-words derived from ``seed``: questions that do not resemble each other."""
    import hashlib

    return " ".join(hashlib.sha256(f"{seed}/{k}".encode()).hexdigest()[:9] for k in range(n))


def provenance(slot_id: str) -> dict:
    return {
        "slot_id": slot_id,
        "drafter": "question-writer",
        "model": "opus",
        "agent_file_sha256": "a" * 64,
        "plan_sha256": "b" * 64,
        "drafted_on": "2026-10-16",
        "critic_verdict": "keep",
        "critic_rounds": 1,
        "critic_notes": "ok",
    }
