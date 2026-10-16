"""``benchmark plan``: the deterministic draft plan (docs/benchmark_spec.md, sections 1 to 4, 7).

The plan decides, by code and before anything is drafted, which articles are drafted from, in
which query language and from which source language. A *slot* is one planned draft. Writers
never choose or see an article id.

Algorithm (all inputs sorted by natural order, one ``random.Random(20261015)`` consumed in the
fixed order below):

0. **Twins.** Pool articles whose German text has a character 5-gram Jaccard of at least 0.7 with
   another pool article (``twins.py``) are removed as gold candidates, primary and secondary. They
   stay in the retrieval corpus. The removed articles are listed in ``twins.jsonl``.
1. **Cells.** One cell per (act, split). Final quotas: the act table of the spec, split into dev
   (40) and test (110) by largest remainder (ties go to the earlier act). Draft slots per cell:
   ``ceil(1.5 x quota)``, so 228 answerable slots for 150 final items.
2. **cross_ref.** 22 final cross_ref items are spread over the cells by largest remainder in
   proportion to the quotas. Candidates: an eligible article A whose text (DE, FR and IT agree)
   refers to another article B of the same act (``refs.same_act_references``), where B resolves
   with ``eligibility.resolve_gold`` and falls in the same split; an A with more than five
   distinct references is skipped. Per cell the distinct A articles are shuffled (seeded) and each
   draws one B at random, while neither article is used. A cell with too few disjoint pairs gets
   fewer cross_ref slots; the shortfall is reported, not hidden.
3. **single.** The other articles of the cell are drawn with ``rng.sample`` from the unused
   articles. An article is used at most once in the whole plan, as gold or as secondary gold.
4. **Order inside a cell.** The first ``quota`` slots (the *final prefix*) already have the final
   composition (languages, cross_ref count); the surplus slots follow. ``select`` keeps the first
   ``quota`` slots the critic kept, in slot order, so a clean run yields the final mix.
5. **Query language.** Final prefix: 38 EN / 37 DE / 38 FR / 37 IT over all cells (a cell-by-cell
   proportional rounding with carry, exact in total and within one or two per act). The 78
   surplus slots are balanced the same way.
6. **Source language.** EN questions: DE, FR, IT in turn. DE/FR/IT questions: every fourth is
   drafted from the same language (control group, 25%), the others alternate between the two
   other languages. Counters run over the final-prefix slots and the surplus slots separately.
7. **Traps.** 30 trap slots (``s-0229`` ...). Their split is the hash rank of the slot id
   (lowest 8 hashes dev, the other 22 test), never a property of a text. Inside a split the
   query languages follow the same final/surplus balancing (5 per language in the final prefix)
   and the trap type rotates over the four types.
8. **Pilot.** Ten slots from the dev split (8 answerable, 2 traps), the one with the lowest
   ``sha256("fedlex-bench-v1-pilot:" + slot_id)`` that meets each requirement in turn.

Stop conditions of spec section 14 enforced here: an act whose pool is smaller than twice the
number of articles it uses, or a cell too small to draw from, raises :class:`PlanError`.
"""

from __future__ import annotations

import argparse
import hashlib
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from fedlex_bench.benchmark.common import (
    ACT_QUOTA,
    CORPUS_LANGS,
    CROSS_REF_FINAL,
    DEV_ANSWERABLE,
    LANG_FINAL,
    QUERY_LANGS,
    SEED,
    SLOTS_PATH,
    SPLITS,
    TRAP_DRAFT_TOTAL,
    TRAP_FINAL,
    TRAP_LANG_FINAL,
    TRAP_TYPES,
    TWINS_PATH,
    act_key,
    draft_count,
    id_key,
    largest_remainder,
    load_corpus,
    read_jsonl,
    sha256_file,
    slots_sha256,
    split_of,
    write_jsonl,
)
from fedlex_bench.benchmark.refs import same_act_references
from fedlex_bench.benchmark.twins import THRESHOLD as TWIN_THRESHOLD
from fedlex_bench.benchmark.twins import find_twins, twin_records
from fedlex_bench.chunk import ARTICLES_PARQUET
from fedlex_bench.eligibility import benchmark_eligible, resolve_gold, retrieval_eligible

Cell = tuple[str, str]
CELLS: list[Cell] = [(act, split) for act in ACT_QUOTA for split in SPLITS]
TRAP_SPLIT_PREFIX = "fedlex-bench-v1-trap:"
PILOT_PREFIX = "fedlex-bench-v1-pilot:"
PILOT_SIZE = 10


class PlanError(RuntimeError):
    """The corpus cannot support the plan (a stop condition of the spec)."""


@dataclass
class PlanResult:
    slots: list[dict]
    cross_report: list[dict] = field(default_factory=list)
    pool_report: list[dict] = field(default_factory=list)
    twins: dict[str, tuple[str, float]] = field(default_factory=dict)
    twin_report: list[dict] = field(default_factory=list)
    seed: int = SEED


# ------------------------------------------------------------------ quotas


def final_quotas() -> dict[Cell, int]:
    """Final answerable quota per (act, split): the act table, dev total 40, test total 110."""
    dev = largest_remainder(DEV_ANSWERABLE, ACT_QUOTA, order=list(ACT_QUOTA))
    quotas: dict[Cell, int] = {}
    for act, total in ACT_QUOTA.items():
        quotas[(act, "dev")] = dev[act]
        quotas[(act, "test")] = total - dev[act]
    return quotas


def cross_ref_targets() -> dict[Cell, int]:
    """Final cross_ref items per cell: 22 in total, in proportion to the cell quotas (largest
    remainder, ties to the earlier cell)."""
    quotas = final_quotas()
    return largest_remainder(CROSS_REF_FINAL, quotas, order=CELLS)


def assign_matrix(rows: list[tuple[object, int]], cols: dict[str, int]) -> dict[object, dict]:
    """Distribute integer row totals over the columns so that the column totals are met exactly
    and every row is as proportional to the remaining column totals as rounding allows (carry
    from row to row; ties to the earlier column). Needs ``sum(rows) == sum(cols)``."""
    if sum(n for _, n in rows) != sum(cols.values()):
        raise ValueError("row totals and column totals differ")
    remaining = dict(cols)
    out: dict[object, dict] = {}
    for key, n in rows:
        total = sum(remaining.values())
        if n == total:
            alloc = dict(remaining)
        elif n == 0:
            alloc = {c: 0 for c in remaining}
        else:
            alloc = largest_remainder(n, remaining, order=list(remaining))
            for c in alloc:  # a column can never take more than it has left
                alloc[c] = min(alloc[c], remaining[c])
            short = n - sum(alloc.values())
            for c in remaining:
                if short and alloc[c] < remaining[c]:
                    take = min(short, remaining[c] - alloc[c])
                    alloc[c] += take
                    short -= take
        for c, v in alloc.items():
            remaining[c] -= v
        out[key] = alloc
    return out


def interleave(counts: dict[str, int], order: tuple[str, ...] = QUERY_LANGS) -> list[str]:
    """Spread the items of ``counts`` evenly over a sequence (Bresenham style: at every step take
    the one furthest behind its ideal share; ties to the earlier key in ``order``)."""
    total = sum(counts.values())
    placed = {k: 0 for k in counts}
    seq: list[str] = []
    for step in range(total):
        best = max(
            (k for k in order if k in counts and counts[k] > placed[k]),
            key=lambda k: (counts[k] * (step + 1) - placed[k] * total, -order.index(k)),
        )
        placed[best] += 1
        seq.append(best)
    return seq


def spread(n: int, k: int) -> set[int]:
    """``k`` evenly spread positions out of ``n``."""
    if k > n:
        raise ValueError("cannot spread more positions than there are")
    return {(2 * i + 1) * n // (2 * k) for i in range(k)}


# ------------------------------------------------------------------ candidate pairs


MAX_DISTINCT_REFERENCES = 5


def candidate_pairs(df: pd.DataFrame, pool_ids: list[str]) -> dict[Cell, dict[str, list[str]]]:
    """Per cell, ``{referencing article A: [articles B it cites]}`` among the eligible articles of
    one split. A reference must be found in DE, FR and IT alike. An article with more than five
    distinct same-act references is skipped as A: it is unclear which one the question needs."""
    pool = set(pool_ids)
    known = set(df["article_id"])
    live = retrieval_eligible(df)
    live = live[live["article_id"].isin(pool)]
    texts = {(r.article_id, r.lang): r.text for r in live.itertuples(index=False)}
    pairs: dict[Cell, dict[str, list[str]]] = {cell: {} for cell in CELLS}
    for aid in pool_ids:
        sr = aid.partition(":")[0]
        found = [same_act_references(texts[(aid, lang)], lang) for lang in CORPUS_LANGS]
        if max(len(f) for f in found) > MAX_DISTINCT_REFERENCES:
            continue
        per_lang = [{f"{sr}:{n}" for n in f} for f in found]
        cited: list[str] = []
        for target in sorted(set.intersection(*per_lang), key=id_key):
            gold = resolve_gold(target, pool, known)
            if gold is None or gold == aid or split_of(gold) != split_of(aid):
                continue
            cited.append(gold)
        cell = (sr, split_of(aid))
        if cited and cell in pairs:
            pairs[cell][aid] = cited
    return pairs


def draw_pairs(
    candidates: dict[str, list[str]], want: int, rng: random.Random, used: set[str]
) -> list[tuple[str, str]]:
    """Up to ``want`` disjoint (A, B) pairs: the distinct A articles are shuffled (seeded) and each
    gets one B drawn at random from the articles it cites that are still unused."""
    order = sorted(candidates, key=id_key)
    rng.shuffle(order)
    chosen: list[tuple[str, str]] = []
    for a in order:
        if len(chosen) == want:
            break
        if a in used:
            continue
        options = [b for b in candidates[a] if b not in used and b != a]
        if not options:
            continue
        b = rng.choice(options)
        chosen.append((a, b))
        used.update((a, b))
    return chosen


# ------------------------------------------------------------------ the plan


def find_pool_twins(df: pd.DataFrame, pool_ids: list[str]) -> dict[str, tuple[str, float]]:
    """Twins among the pool articles, by the German text (see ``twins.py``)."""
    wanted = set(pool_ids)
    de = df[(df["lang"] == "de") & df["article_id"].isin(wanted)]
    return find_twins(dict(zip(de["article_id"], de["text"], strict=True)), TWIN_THRESHOLD)


def build_plan(df: pd.DataFrame, seed: int = SEED) -> PlanResult:
    eligible = sorted(benchmark_eligible(df)["article_id"], key=id_key)
    twins_found = find_pool_twins(df, eligible)
    pool_ids = [a for a in eligible if a not in twins_found]
    twin_report = []
    for act in ACT_QUOTA:
        before = sum(1 for a in eligible if a.partition(":")[0] == act)
        removed = sum(1 for a in twins_found if a.partition(":")[0] == act)
        twin_report.append(
            {"act": act, "pool_before": before, "removed": removed, "pool_after": before - removed}
        )
    cell_articles: dict[Cell, list[str]] = {cell: [] for cell in CELLS}
    for aid in pool_ids:
        cell = (aid.partition(":")[0], split_of(aid))
        if cell in cell_articles:
            cell_articles[cell].append(aid)
    quotas = final_quotas()
    cross_final = cross_ref_targets()
    drafts = {cell: draft_count(quotas[cell]) for cell in CELLS}

    # query language per cell: final prefix, then surplus
    final_cols = dict(LANG_FINAL)
    draft_cols = largest_remainder(sum(drafts.values()), LANG_FINAL, order=list(QUERY_LANGS))
    surplus_cols = {lang: draft_cols[lang] - final_cols[lang] for lang in QUERY_LANGS}
    final_langs = assign_matrix([(c, quotas[c]) for c in CELLS], final_cols)
    surplus_langs = assign_matrix([(c, drafts[c] - quotas[c]) for c in CELLS], surplus_cols)

    pairs_by_cell = candidate_pairs(df, pool_ids)
    rng = random.Random(seed)
    used: set[str] = set()
    cross_langs: Counter = Counter()  # cross_ref slots per query language so far
    answerable: list[dict] = []
    cross_report: list[dict] = []
    for cell in CELLS:
        act, split = cell
        q, nd = quotas[cell], drafts[cell]
        candidates = pairs_by_cell[cell]
        want = min(nd, draft_count(cross_final[cell]))
        chosen = draw_pairs(candidates, want, rng, used)
        free = [a for a in cell_articles[cell] if a not in used]
        n_single = nd - len(chosen)
        if len(free) < n_single:
            raise PlanError(
                f"cell {act}/{split}: {len(free)} unused eligible articles left for {n_single} "
                f"single slots (pool {len(cell_articles[cell])})"
            )
        singles = rng.sample(free, n_single)
        used.update(singles)

        cross_prefix = min(cross_final[cell], len(chosen))
        cross_extra = len(chosen) - cross_prefix
        flags = [False] * nd
        for i in spread(q, cross_prefix):
            flags[i] = True
        for i in spread(nd - q, cross_extra):
            flags[q + i] = True
        langs = _cell_languages(flags, q, final_langs[cell], surplus_langs[cell], cross_langs)
        pair_iter, single_iter = iter(chosen), iter(singles)
        for pos in range(nd):
            gold = list(next(pair_iter)) if flags[pos] else [next(single_iter)]
            answerable.append(
                {
                    "act": act,
                    "gold_article_ids": gold,
                    "pilot": False,
                    "qtype": "cross_ref" if flags[pos] else "single",
                    "query_lang": langs[pos],
                    "slot_id": "",
                    "source_lang": None,
                    "split": split,
                    "trap_type": None,
                    "_final": pos < q,
                }
            )
        cross_report.append(
            {
                "cell": cell,
                "act": act,
                "split": split,
                "candidate_articles": len(candidates),
                "candidate_pairs": sum(len(v) for v in candidates.values()),
                "final_target": cross_final[cell],
                "draft_target": want,
                "draft_slots": len(chosen),
                "final_slots": cross_prefix,
                "shortfall": cross_final[cell] - cross_prefix,
            }
        )

    pool_report = []
    for act in ACT_QUOTA:
        pool_n = sum(len(cell_articles[(act, s)]) for s in SPLITS)
        used_n = sum(len(s["gold_article_ids"]) for s in answerable if s["act"] == act)
        pool_report.append({"act": act, "pool": pool_n, "used": used_n})
        if pool_n < 2 * used_n:
            raise PlanError(
                f"act {act}: pool of {pool_n} articles is below twice the {used_n} articles the "
                "plan uses (spec section 14: stop and ask)"
            )

    _assign_source_languages(answerable)
    for i, slot in enumerate(answerable, 1):
        slot["slot_id"] = f"s-{i:04d}"
    trap_slots = _trap_slots(len(answerable))
    slots = answerable + trap_slots
    _mark_pilot(slots)
    for slot in slots:
        slot.pop("_final", None)
    return PlanResult(
        slots=slots,
        cross_report=cross_report,
        pool_report=pool_report,
        twins=twins_found,
        twin_report=twin_report,
        seed=seed,
    )


def _cell_languages(
    flags: list[bool],
    quota: int,
    final_counts: dict[str, int],
    surplus_counts: dict[str, int],
    cross_langs: Counter,
) -> list[str]:
    """Query language per slot position of a cell. The cross_ref positions take the language that
    has had the fewest cross_ref slots so far (among those still owed to the block), the other
    positions of the block are interleaved from what is left. The final-prefix block and the
    surplus block keep their own counts."""
    langs: list[str] = [""] * len(flags)
    for lo, hi, counts in (
        (0, quota, dict(final_counts)),
        (quota, len(flags), dict(surplus_counts)),
    ):
        for pos in range(lo, hi):
            if flags[pos]:
                options = [lang for lang in QUERY_LANGS if counts.get(lang, 0) > 0]
                lang = min(options, key=lambda x: (cross_langs[x], QUERY_LANGS.index(x)))
                counts[lang] -= 1
                cross_langs[lang] += 1
                langs[pos] = lang
        rest = [pos for pos in range(lo, hi) if not flags[pos]]
        sequence = interleave({lang: n for lang, n in counts.items() if n > 0})
        for pos, lang in zip(rest, sequence, strict=True):
            langs[pos] = lang
    return langs


def _assign_source_languages(answerable: list[dict]) -> None:
    """EN: DE, FR, IT in turn. DE/FR/IT: every fourth question from the same language, the others
    alternating between the two other languages. Final-prefix and surplus slots count apart."""
    seen: Counter = Counter()
    cross_seen: Counter = Counter()
    for slot in answerable:
        lang, group = slot["query_lang"], slot["_final"]
        k = seen[(lang, group)]
        seen[(lang, group)] += 1
        if lang == "en":
            slot["source_lang"] = CORPUS_LANGS[k % 3]
        elif k % 4 == 3:
            slot["source_lang"] = lang
        else:
            others = [x for x in CORPUS_LANGS if x != lang]
            slot["source_lang"] = others[cross_seen[(lang, group)] % 2]
            cross_seen[(lang, group)] += 1


def _trap_slots(n_answerable: int) -> list[dict]:
    ids = [f"s-{n_answerable + i:04d}" for i in range(1, TRAP_DRAFT_TOTAL + 1)]
    n_dev = draft_count(TRAP_FINAL["dev"])
    ranked = sorted(ids, key=lambda i: hashlib.sha256((TRAP_SPLIT_PREFIX + i).encode()).hexdigest())
    dev_ids = set(ranked[:n_dev])
    split_of_slot = {i: ("dev" if i in dev_ids else "test") for i in ids}

    drafts = {s: sum(1 for i in ids if split_of_slot[i] == s) for s in SPLITS}
    cols_draft = largest_remainder(TRAP_DRAFT_TOTAL, TRAP_LANG_FINAL, order=list(QUERY_LANGS))
    surplus_cols = {lang: cols_draft[lang] - TRAP_LANG_FINAL[lang] for lang in QUERY_LANGS}
    final_langs = assign_matrix([(s, TRAP_FINAL[s]) for s in SPLITS], dict(TRAP_LANG_FINAL))
    surplus_langs = assign_matrix([(s, drafts[s] - TRAP_FINAL[s]) for s in SPLITS], surplus_cols)

    slots: dict[str, dict] = {}
    for split in SPLITS:
        cell_ids = sorted(i for i in ids if split_of_slot[i] == split)
        langs = interleave(final_langs[split]) + interleave(surplus_langs[split])
        for pos, slot_id in enumerate(cell_ids):
            slots[slot_id] = {
                "act": None,
                "gold_article_ids": [],
                "pilot": False,
                "qtype": "trap",
                "query_lang": langs[pos],
                "slot_id": slot_id,
                "source_lang": None,
                "split": split,
                "trap_type": TRAP_TYPES[(pos + pos // 4) % 4],
                "_final": pos < TRAP_FINAL[split],
            }
    return [slots[i] for i in ids]


# ------------------------------------------------------------------ pilot


def _pilot_hash(slot_id: str) -> str:
    return hashlib.sha256((PILOT_PREFIX + slot_id).encode("utf-8")).hexdigest()


def _mark_pilot(slots: list[dict]) -> None:
    """Ten dev slots: 5 from SR 220 (one cross_ref, EN, a same-language DE control, a cross-lingual
    FR and IT question), one each from SR 210, 272 and 281.1, and two traps (different type and
    query language). Each pick is the lowest pilot hash among the slots that fit."""
    picked: list[dict] = []

    def pick(predicates: list) -> None:
        for predicate in predicates:  # strict first, then relaxed
            options = [s for s in slots if s not in picked and s["split"] == "dev" and predicate(s)]
            if options:
                picked.append(min(options, key=lambda s: _pilot_hash(s["slot_id"])))
                return
        raise PlanError("no dev slot is available for the pilot")

    def act_q(act: str, qtype: str, lang: str | None = None, same: bool | None = None):
        def strict(s: dict) -> bool:
            if s["act"] != act or s["qtype"] != qtype:
                return False
            if lang is not None and s["query_lang"] != lang:
                return False
            return same is None or (s["source_lang"] == s["query_lang"]) == same

        return strict

    def any_act(qtype: str):
        return lambda s: s["qtype"] == qtype

    # without any cross_ref slot (no candidate pairs) the pilot has none either
    pick([act_q("220", "cross_ref"), any_act("cross_ref"), act_q("220", "single")])
    pick([act_q("220", "single", "en"), act_q("220", "single")])
    pick([act_q("220", "single", "de", True), act_q("220", "single", None, True)])
    pick([act_q("220", "single", "fr", False), act_q("220", "single", None, False)])
    pick([act_q("220", "single", "it", False), act_q("220", "single", None, False)])
    for act in ("210", "272", "281.1"):
        pick([act_q(act, "single")])

    def trap(different_from: list[dict] | None = None):
        def fits(s: dict) -> bool:
            if s["qtype"] != "trap":
                return False
            return not different_from or all(
                s["trap_type"] != t["trap_type"] and s["query_lang"] != t["query_lang"]
                for t in different_from
            )

        return fits

    pick([trap()])
    first_trap = picked[-1]
    pick([trap([first_trap]), trap()])
    for s in picked:
        s["pilot"] = True
    assert len(picked) == PILOT_SIZE


# ------------------------------------------------------------------ files and summary


def final_prefix(slots: list[dict]) -> list[dict]:
    """The first ``quota`` slots of every answerable cell and the first 5 / 15 trap slots of the
    dev / test split, in slot order: the final mix if the critic keeps every slot."""
    quotas = final_quotas()
    seen: Counter = Counter()
    out: list[dict] = []
    for s in slots:
        cell = (s["act"], s["split"])
        limit = TRAP_FINAL[s["split"]] if s["qtype"] == "trap" else quotas[cell]
        if seen[cell] < limit:
            out.append(s)
        seen[cell] += 1
    return out


def write_slots(slots: list[dict], path: Path) -> str:
    """Write ``slots.jsonl`` (sorted keys, UTF-8, ``\\n``); returns its sha256."""
    return write_jsonl(path, slots)


def read_slots(path: Path) -> list[dict]:
    return read_jsonl(path)


def _count_table(rows: list[tuple[str, list[str]]], columns: list[str]) -> list[str]:
    """Rows of ``(label, values)`` -> aligned text lines with one count column per ``columns``."""
    width = max([len(label) for label, _ in rows] + [12])
    head = f"{'':<{width}}" + "".join(f"{c:>7}" for c in columns) + f"{'total':>7}"
    lines = [head]
    for label, values in rows:
        counts = Counter(values)
        lines.append(
            f"{label:<{width}}"
            + "".join(f"{counts.get(c, 0):>7}" for c in columns)
            + f"{len(values):>7}"
        )
    return lines


def summary_text(result: PlanResult) -> str:
    slots = result.slots
    ans = [s for s in slots if s["qtype"] != "trap"]
    out: list[str] = []
    out.append(
        f"PLAN: {len(slots)} slots = {len(ans)} answerable + {len(slots) - len(ans)} traps; "
        f"{sum(s['pilot'] for s in slots)} pilot; seed {result.seed}"
    )
    out.append("")
    out.append("act x split x qtype: query_lang | source_lang (all draft slots)")
    head = (
        f"{'act':<10}{'split':<6}{'qtype':<10}{'n':>4}  "
        + "".join(f"{x:>4}" for x in QUERY_LANGS)
        + "  |"
        + "".join(f"{x:>4}" for x in (*CORPUS_LANGS, "-"))
    )
    out.append(head)
    groups: dict[tuple, list[dict]] = {}
    for s in slots:
        groups.setdefault((s["act"] or "(trap)", s["split"], s["qtype"]), []).append(s)
    order = sorted(groups, key=lambda g: (act_key(None if g[0] == "(trap)" else g[0]), g[1], g[2]))
    for g in order:
        members = groups[g]
        q = Counter(s["query_lang"] for s in members)
        src = Counter(s["source_lang"] or "-" for s in members)
        out.append(
            f"{g[0]:<10}{g[1]:<6}{g[2]:<10}{len(members):>4}  "
            + "".join(f"{q.get(x, 0):>4}" for x in QUERY_LANGS)
            + "  |"
            + "".join(f"{src.get(x, 0):>4}" for x in (*CORPUS_LANGS, "-"))
        )
    out.append("")
    out.append("query_lang x source_lang (answerable draft slots)")
    out.extend(
        _count_table(
            [
                (f"query {q}", [s["source_lang"] for s in ans if s["query_lang"] == q])
                for q in QUERY_LANGS
            ],
            list(CORPUS_LANGS),
        )
    )
    out.append("")
    fin = final_prefix(slots)
    out.append(
        f"final prefix = the first quota slots of each cell, what `select` keeps if the critic "
        f"keeps everything: {len(fin)} slots"
    )
    out.append(
        f"{'act':<10}{'n':>4}{'xref':>6}  "
        + "".join(f"{x:>4}" for x in QUERY_LANGS)
        + "  |"
        + "".join(f"{x:>4}" for x in (*CORPUS_LANGS, "-"))
    )
    for act in [*ACT_QUOTA, None]:
        members = [s for s in fin if s["act"] == act]
        q = Counter(s["query_lang"] for s in members)
        src = Counter(s["source_lang"] or "-" for s in members)
        out.append(
            f"{act or '(trap)':<10}{len(members):>4}"
            f"{sum(s['qtype'] == 'cross_ref' for s in members):>6}  "
            + "".join(f"{q.get(x, 0):>4}" for x in QUERY_LANGS)
            + "  |"
            + "".join(f"{src.get(x, 0):>4}" for x in (*CORPUS_LANGS, "-"))
        )
    q = Counter(s["query_lang"] for s in fin)
    out.append(
        f"{'all':<10}{len(fin):>4}{sum(s['qtype'] == 'cross_ref' for s in fin):>6}  "
        + "".join(f"{q.get(x, 0):>4}" for x in QUERY_LANGS)
    )
    out.append("")
    out.append("cross_ref allocation by cell (final target, drafted, shortfall)")
    out.append(f"{'cell':<16}{'pairs':>7}{'final':>7}{'draft':>7}{'got':>5}{'short':>7}")
    by_act: Counter = Counter()
    short_act: Counter = Counter()
    for row in result.cross_report:
        by_act[row["act"]] += row["final_slots"]
        short_act[row["act"]] += row["shortfall"]
        out.append(
            f"{row['act'] + '/' + row['split']:<16}{row['candidate_pairs']:>7}"
            f"{row['final_target']:>7}{row['draft_target']:>7}{row['draft_slots']:>5}"
            f"{row['shortfall']:>7}"
        )
    total_short = sum(short_act.values())
    out.append(
        "final cross_ref by act: "
        + ", ".join(f"{a}={by_act[a]}" for a in ACT_QUOTA)
        + f"; total {sum(by_act.values())} of {CROSS_REF_FINAL}; shortfall {total_short}"
    )
    out.append("")
    out.append("twins: articles removed as gold candidates (DE 5-gram Jaccard >= 0.7)")
    out.append(f"  {'act':<10}{'eligible':>9}{'removed':>9}{'pool':>7}")
    for row in result.twin_report:
        out.append(
            f"  {row['act']:<10}{row['pool_before']:>9}{row['removed']:>9}{row['pool_after']:>7}"
        )
    total_removed = sum(r["removed"] for r in result.twin_report)
    out.append(f"  removed in total: {total_removed}")
    out.append("")
    out.append("pool check (articles in the pool vs articles used by the plan)")
    for row in result.pool_report:
        out.append(f"  {row['act']:<10}pool {row['pool']:>5}  used {row['used']:>4}")
    return "\n".join(out) + "\n"


def run(argv: list[str] | None = None, corpus: pd.DataFrame | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="benchmark plan", description="Write the draft plan (slots.jsonl)."
    )
    ap.add_argument("--parquet", type=Path, default=ARTICLES_PARQUET)
    ap.add_argument("--out", type=Path, default=SLOTS_PATH)
    ap.add_argument("--twins-out", type=Path, default=TWINS_PATH)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--force", action="store_true", help="overwrite an existing slots file that differs"
    )
    args = ap.parse_args(argv)
    df = corpus if corpus is not None else load_corpus(args.parquet)
    try:
        result = build_plan(df, seed=args.seed)
    except PlanError as exc:
        print(f"STOP: {exc}")
        return 2
    new_sha = slots_sha256(result.slots)
    if args.out.exists() and sha256_file(args.out) != new_sha and not args.force:
        print(
            f"REFUSED: {args.out} exists and differs from the new plan "
            f"(existing sha256 {sha256_file(args.out)}, new {new_sha}). Drafts and briefs refer to "
            "the existing plan; use --force to replace it."
        )
        return 1
    sha = write_slots(result.slots, args.out)
    write_jsonl(args.twins_out, twin_records(result.twins))
    print(summary_text(result))
    print(f"{args.out.name} sha256 {sha}")
    print(f"{args.twins_out.name}: {len(result.twins)} twin articles")
    return 0
