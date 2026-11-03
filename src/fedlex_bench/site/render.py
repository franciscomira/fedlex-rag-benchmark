"""HTML for the results page. Every number is read from a ``Report``; prose carries no digits."""

from __future__ import annotations

from html import escape
from typing import Any

from fedlex_bench.serve.notices import NOT_LEGAL_ADVICE, NOT_OFFICIAL_DE, NOT_OFFICIAL_EN
from fedlex_bench.site.build import (
    Report,
    clopper_pearson,
    expand_commands,
    expand_text,
)
from fedlex_bench.site.data import (
    Diff,
    Run,
    Stat,
    fmt_diff,
    fmt_latency,
    fmt_stat,
)

e = escape

# ---------------------------------------------------------------- small pieces


def stat_span(stat: Stat | Diff | None, digits: int = 2, diff: bool = False) -> str:
    """``0.81 [0.71, 0.90]`` with the interval styled as secondary text."""
    if stat is None:
        return '<span class="na">n/a</span>'
    text = fmt_diff(stat, digits) if diff else fmt_stat(stat, digits)
    head, _, tail = text.partition(" ")
    return f'<span class="v">{e(head)}</span> <span class="ci">{e(tail)}</span>'


def lang_attr(lang: str | None) -> str:
    return f' data-lang="{lang}"' if lang else ""


def pending_cell(span: int, reason: str) -> str:
    return f'<td class="pend" colspan="{span}">{e(reason)}</td>'


def run_header(run: Run) -> str:
    sub = ""
    if run.ok:
        frozen = f"frozen as {run.frozen_name}, " if run.frozen_name else ""
        sub = f'<small class="id">{e(frozen)}config {e(run.config_hash)}</small>'
    elif run.run_id:
        sub = f'<small class="id">{e(run.run_id)}</small>'
    return (
        f'<th scope="row"><span class="lab">{e(run.label)}</span> '
        f'<span class="nm">{e(run.name)}</span>{sub}</th>'
    )


def claim_word(diff: Diff | None) -> str:
    if diff is None:
        return "n/a"
    if not diff.claimable:
        return "not claimable"
    return "claimable (first higher)" if diff.mean > 0 else "claimable (first lower)"


def tag_html(tag: str) -> str:
    return f'<span class="tag tag-{e(tag.lower())}">{e(tag)}</span>'


# ---------------------------------------------------------------- generic tables

# (header, metric, language or None, digits)
Col = tuple[str, str, str | None, int]

RETRIEVAL_COLS: tuple[Col, ...] = (
    ("Recall@1", "recall@1", None, 2),
    ("Recall@10", "recall@10", None, 2),
    ("MRR", "mrr", None, 2),
    ("English recall@10", "recall@10", "en", 2),
)
COST_COL: Col = ("Cost per question (USD)", "cost_usd", None, 4)
ANSWER_COLS: tuple[Col, ...] = (
    ("Fact recall", "fact_recall", None, 2),
    ("Faithfulness", "faithfulness", None, 2),
    ("Citation precision", "cit_precision", None, 2),
    ("Over-refusal", "over_refusal", None, 2),
)


def run_table(rep: Report, run_keys: list[str], cols: list[Col], traps: bool = False,
              latency: bool = True) -> str:
    heads = [c[0] for c in cols]
    if traps:
        heads.append("Traps handled (exact 95% interval)")
    if latency:
        heads.append("Latency p50 / p95")
    out = ['<div class="tablewrap" tabindex="0"><table class="data">']
    out.append("<thead><tr><th scope=\"col\">System</th>")
    out.extend(f'<th scope="col" class="num">{e(h)}</th>' for h in heads)
    out.append("</tr></thead><tbody>")
    for key in run_keys:
        run = rep.runs[key]
        out.append(f'<tr data-run="{e(key)}"' + ("" if run.ok else ' class="pending"') + ">")
        out.append(run_header(run))
        if not run.ok:
            reason = f"{run.reason}" if run.run_id else "run id not yet in the manifest"
            out.append(pending_cell(len(heads), f"pending: {reason}"))
        else:
            for _, metric, lang, digits in cols:
                stat = run.stat(metric, lang)
                out.append(
                    f'<td class="num" data-run="{e(key)}" data-metric="{e(metric)}"'
                    f'{lang_attr(lang)}>{stat_span(stat, digits)}</td>'
                )
            if traps:
                out.append(f'<td class="num" data-run="{e(key)}" data-metric="traps">'
                           f"{trap_cell(run)}</td>")
            if latency:
                out.append(f'<td class="num" data-run="{e(key)}" data-metric="latency">'
                           f"{e(fmt_latency(run))}</td>")
        out.append("</tr>")
    out.append("</tbody></table></div>")
    out.append(legend(rep, run_keys))
    return "".join(out)


def legend(rep: Report, run_keys: list[str]) -> str:
    """Full system names as a list; shown only on narrow screens, where the table hides them."""
    items = "".join(
        f"<li><strong>{e(rep.runs[k].label)}</strong>: {e(rep.runs[k].name)}</li>" for k in run_keys
    )
    return f'<ul class="legend">{items}</ul>'


def trap_cell(run: Run) -> str:
    t = run.traps
    if t is None:
        return '<span class="na">n/a</span>'
    k, n = t
    lo, hi = clopper_pearson(k, n)
    return (f'<span class="v">{k} of {n}</span> '
            f'<span class="ci">[{lo:.2f}, {hi:.2f}]</span>')


def diff_table(rep: Report, section: str, metrics: list[tuple[str, str]]) -> str:
    """Paired differences (first minus second) with a claimable verdict per cell."""
    span = len(metrics)
    out = ['<div class="tablewrap" tabindex="0"><table class="data">']
    out.append('<thead><tr><th scope="col">First minus second</th>')
    out.extend(f'<th scope="col" class="num">{e(h)}</th>' for h, _ in metrics)
    out.append("</tr></thead><tbody>")
    for comp in rep.section_comparisons(section):
        a, b = comp.a, comp.b
        pair = f"{a.key}-{b.key}"
        head = (f'<th scope="row"><span class="lab">{e(a.label)} minus {e(b.label)}</span></th>')
        if not comp.ready:
            out.append(f'<tr class="pending" data-cmp="{e(pair)}">{head}'
                       f"{pending_cell(span, comp.error)}</tr>")
            continue
        out.append(f'<tr data-cmp="{e(pair)}">{head}')
        for _, metric in metrics:
            d = comp.diffs.get(metric)
            if d is None:
                out.append(f'<td class="num" data-cmp="{e(pair)}" data-metric="{e(metric)}">'
                           '<span class="na">n/a</span></td>')
                continue
            cls = "yes" if d.claimable else "no"
            out.append(
                f'<td class="num" data-cmp="{e(pair)}" data-metric="{e(metric)}">'
                f'{stat_span(d, 3, diff=True)}'
                f'<span class="verdict {cls}">{e(claim_word(d))}</span></td>'
            )
        out.append("</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def not_claimable(rep: Report, section: str, metrics: list[tuple[str, str]]) -> str:
    """Plain list of the differences whose interval includes 0."""
    items = []
    for comp in rep.section_comparisons(section):
        if not comp.ready:
            continue
        names = [h for h, m in metrics if (d := comp.diffs.get(m)) is not None and not d.claimable]
        if names:
            items.append(f"{comp.a.label} minus {comp.b.label}: {', '.join(names)}")
    if not items:
        ready = any(c.ready for c in rep.section_comparisons(section))
        return ("<p>Every computed difference above has an interval that excludes 0.</p>"
                if ready else "")
    lis = "".join(f"<li>{e(i)}</li>" for i in items)
    return (
        "<p><strong>Not claimable</strong> (the 95% interval includes 0, so the data do not show "
        f"a difference in either direction):</p><ul>{lis}</ul>"
    )


def interval_chart(rep: Report, run_keys: list[str], metric: str, title: str) -> str:
    rows = []
    for key in run_keys:
        run = rep.runs[key]
        stat = run.stat(metric)
        if stat is None:
            continue
        rows.append(
            f'<div class="irow"><span class="ilab">{e(run.label)}</span>'
            f'<span class="itrack"><span class="iwhisk" style="left:{stat.lo * 100:.1f}%;'
            f'width:{(stat.hi - stat.lo) * 100:.1f}%"></span>'
            f'<span class="idot" style="left:{stat.mean * 100:.1f}%"></span></span>'
            f'<span class="ival">{stat.mean:.2f}</span></div>'
        )
    if not rows:
        return ""
    axis = ('<div class="irow iaxis"><span class="ilab"></span><span class="itrack">'
            '<span style="left:0">0</span><span style="left:50%">0.5</span>'
            '<span style="left:100%">1</span></span><span class="ival"></span></div>')
    return (f'<figure class="chart" aria-hidden="true"><figcaption>{e(title)}</figcaption>'
            + "".join(rows) + axis + "</figure>")


# ---------------------------------------------------------------- prose pieces


class Fmt:
    """Formatting helpers bound to a report; every string returned comes from run data."""

    def __init__(self, rep: Report):
        self.rep = rep

    def v(self, key: str, metric: str, lang: str | None = None, digits: int = 2) -> str:
        run = self.rep.runs[key]
        return fmt_stat(run.stat(metric, lang), digits) if run.ok else "pending"

    def d(self, a: str, b: str, metric: str) -> str:
        comp = self.rep.comparison(a, b)
        return fmt_diff(comp.diffs.get(metric)) if comp.ready else "pending"

    def claim(self, a: str, b: str, metric: str) -> str:
        comp = self.rep.comparison(a, b)
        diff = comp.diffs.get(metric) if comp.ready else None
        if diff is None:
            return "pending"
        if not diff.claimable:
            return "not claimable, because the interval includes 0"
        return "claimable, the interval excludes 0"

    def cost(self, key: str) -> str:
        return self.v(key, "cost_usd", digits=4)

    def lat50(self, key: str) -> str:
        from fedlex_bench.site.data import fmt_seconds

        return fmt_seconds(self.rep.runs[key].scalar("latency_p50")) + " s"

    def ratio(self, num: str, den: str, metric: str) -> float | None:
        a, b = self.rep.runs[num].scalar(metric), self.rep.runs[den].scalar(metric)
        if a is None or b is None or not a or not b:
            return None
        return a / b

    def cost_ratio(self, num: str, den: str) -> float | None:
        a, b = self.rep.runs[num].stat("cost_usd"), self.rep.runs[den].stat("cost_usd")
        if a is None or b is None or not b.mean:
            return None
        return a.mean / b.mean


def all_ok(rep: Report, *keys: str) -> bool:
    return all(rep.runs[k].ok for k in keys)


def headline(rep: Report) -> str:
    f = Fmt(rep)
    intro = (
        "<p>The question: can a small fine-tuned multilingual model match an LLM-based pipeline "
        "at finding the right article of Swiss federal law, for questions in German, French, "
        "Italian and English, at a fraction of the cost? What was tuned is the search embedder, "
        "not a reranker, because the first search step, not the reranker, limited the results "
        "of Phase 2. The only LLM in the compared pipelines is a small one"
        f"{_llm_name(rep)}; no frontier-model reranker was compared on the test questions.</p>"
    )
    if not all_ok(rep, "T1", "A3", "Cp", "T2"):
        return intro + "<p>The headline numbers are pending: a run is missing.</p>"
    n = rep.runs["T1"].stat("recall@10").n
    t2_cost = f.cost_ratio("T2", "Cp")
    t2_speed = f.ratio("Cp", "T2", "latency_p50")
    cheaper = ""
    if t2_cost is not None and t2_cost < 1:
        cheaper = f", {(1 - t2_cost) * 100:.0f}% cheaper than C'"
    faster = f", {t2_speed:.1f} times faster at the median" if t2_speed and t2_speed > 1 else ""
    claim = f.claim
    para = (
        f"<p>On {n} answerable test questions (retrieval only, 95% intervals):</p><ul>"
        f"<li><strong>T1, the fine-tuned embedder alone</strong> (no LLM call, "
        f"{e(f.cost('T1'))} USD per question, median {e(f.lat50('T1'))}): recall@10 "
        f"<strong>{e(f.v('T1', 'recall@10'))}</strong>.</li>"
        f"<li><strong>Against the untuned hybrid search</strong> (A corrected, "
        f"{e(f.v('A3', 'recall@10'))}): T1 minus A is {e(f.d('T1', 'A3', 'recall@10'))}, "
        f"{e(claim('T1', 'A3', 'recall@10'))}.</li>"
        f"<li><strong>Against LLM translation</strong> (C': an LLM translates the question first; "
        f"recall@10 {e(f.v('Cp', 'recall@10'))}, {e(f.cost('Cp'))} USD per question, median "
        f"{e(f.lat50('Cp'))}): T1 minus C' is {e(f.d('T1', 'Cp', 'recall@10'))} on recall@10 "
        f"({e(claim('T1', 'Cp', 'recall@10'))}) and {e(f.d('T1', 'Cp', 'mrr'))} on MRR, the "
        f"rank of the right article ({e(claim('T1', 'Cp', 'mrr'))}).</li>"
        f"<li><strong>With a cheap LLM reranker on top</strong> (T2: recall@10 "
        f"{e(f.v('T2', 'recall@10'))}, {e(f.cost('T2'))} USD per question{cheaper}{faster}): "
        f"T2 minus C' is {e(f.d('T2', 'Cp', 'recall@10'))} on recall@10 "
        f"({e(claim('T2', 'Cp', 'recall@10'))}).</li></ul>"
    )
    tail = (
        "<p>These are retrieval results only: they say how often the right article is found, "
        "not that final answers are better. "
        + ("Answer-level results for the tuned pipeline are in section 4."
           if rep.runs["P2"].ok
           else "Answer-level results for the tuned pipeline are pending (section 4).")
        + " The questions were written by an LLM from the text of the article they ask about.</p>"
    )
    int8 = ""
    if all_ok(rep, "P1", "T1"):
        int8 = (
            "<p><strong>What is served is not T1.</strong> The numbers above are for the "
            "full-precision model (T1). The demo and the MCP server use the smaller int8 file "
            f"(P1, retrieval only: recall@10 {e(f.v('P1', 'recall@10'))}). On the test questions "
            f"P1 minus T1 is {e(f.d('P1', 'T1', 'recall@10'))} on recall@10 "
            f"({e(claim('P1', 'T1', 'recall@10'))}) and {e(f.d('P1', 'T1', 'mrr'))} on MRR "
            f"({e(claim('P1', 'T1', 'mrr'))}): the int8 file orders the results a little worse. "
            + (
                "With the LLM reranker on top, MRR is "
                f"{e(f.v('P2', 'mrr'))} for the shipped pipeline (int8, P2) against "
                f"{e(f.v('T2', 'mrr'))} for the full-precision one (T2, retrieval only); the "
                "intervals overlap broadly, and the two runs were not compared pairwise because "
                "they differ in judge settings."
                if all_ok(rep, "P2", "T2")
                else ""
            )
            + "</p>"
        )
    return intro + para + int8 + tail


def _llm_name(rep: Report) -> str:
    run = rep.runs.get("Cp")
    if run and run.ok:
        ref = str(run.config.get("system", {}).get("reranker") or "")
        if "/" in ref:
            return f" ({e(ref.split('/', 1)[1])})"
    return ""


# ---------------------------------------------------------------- sections


def section_phase3(rep: Report) -> str:
    keys = rep.manifest["sections"]["phase3"]["runs"]
    cols = [*RETRIEVAL_COLS, COST_COL]
    metrics = [("Recall@10", "recall@10"), ("MRR", "mrr")]
    return (
        '<section id="phase3"><h2>2. Phase 3: retrieval with the fine-tuned embedder</h2>'
        "<p>Each system was run once on the test split, retrieval only (no answer was generated "
        "or judged). Recall@10 is the share of questions whose gold article is among the first "
        "ten articles returned; MRR rewards ranking it higher. English questions are the hardest "
        "case because no article is in English. Intervals are 95% bootstrap intervals that "
        "resample a question together with its translations.</p>"
        + interval_chart(rep, keys, "recall@10", "Recall@10 with 95% interval")
        + run_table(rep, keys, cols)
        + "<p class=\"fine\">Latency percentiles are single measurements from one machine and "
        "carry no interval. Cost excludes judging.</p>"
        "<h3>Paired differences</h3>"
        + diff_table(rep, "phase3", metrics)
        + not_claimable(rep, "phase3", metrics)
        + "</section>"
    )


def section_phase2(rep: Report) -> str:
    keys = rep.manifest["sections"]["phase2"]["runs"]
    cols = [("Recall@10", "recall@10", None, 2), *ANSWER_COLS, COST_COL]
    metrics = [("Recall@10", "recall@10"), ("MRR", "mrr"), ("Fact recall", "fact_recall"),
               ("Faithfulness", "faithfulness")]
    return (
        '<section id="phase2"><h2>3. Phase 2: answers, and the embedder erratum</h2>'
        '<div class="banner" role="note"><strong>Measured with the buggy embedder.</strong> '
        "Every number in this section's first two tables used a search step whose article "
        "vectors were slightly wrong (see the erratum below). Retrieval was weaker than it should "
        "have been, and the answer-level numbers of B, C and D were not re-measured.</div>"
        "<p>Systems B, C and D generate a cited answer from the retrieved articles; a judge model "
        "scores whether the answer states the expected facts (fact recall) and whether its claims "
        "are supported by the cited text (faithfulness). Citation ids are checked by code, not "
        "generated. Trap questions have no answer in the law; the system should refuse.</p>"
        + run_table(rep, keys, cols, traps=True)
        + "<h3>Paired differences</h3>"
        + diff_table(rep, "phase2", metrics)
        + not_claimable(rep, "phase2", metrics)
        + errata(rep)
        + "</section>"
    )


def errata(rep: Report) -> str:
    f = Fmt(rep)
    head = '<h3 id="errata">Erratum: the embedder padding bug</h3>'
    wrong = (
        "<h4>What was wrong</h4><p>The program that turns text into vectors (the embedder) had a "
        "bug when it handled several texts at once. Shorter texts are padded to the length of "
        "the longest one in the batch, and the padding was wrongly averaged into the vector of "
        "each text. A question embedded alone, as at search time, was correct. Every article of "
        "the search index was embedded in batches, so the article vectors were slightly off, "
        "most for short texts that shared a batch with long ones. It was found in Phase 3, when "
        "the same untrained model scored much better in a second implementation than in the "
        "project's own pipeline. It is fixed, with a regression test, and the old behaviour is "
        "kept behind a switch so the old numbers can still be reproduced.</p>"
    )
    if not all_ok(rep, "A2", "A3"):
        return head + wrong + "<p>The re-measured numbers are pending.</p>"
    rows = [("Hybrid search A: recall@1", "A2", "A3", "recall@1", None),
            ("Hybrid search A: recall@10", "A2", "A3", "recall@10", None),
            ("Hybrid search A: MRR", "A2", "A3", "mrr", None),
            ("Hybrid search A: English recall@10", "A2", "A3", "recall@10", "en")]
    for new, old, label in (("Bp", "B2", "B to B'"), ("Cp", "C2", "C to C'")):
        if all_ok(rep, new, old):
            rows += [(f"{label}: recall@1", old, new, "recall@1", None),
                     (f"{label}: recall@10", old, new, "recall@10", None),
                     (f"{label}: MRR", old, new, "mrr", None),
                     (f"{label}: English recall@10", old, new, "recall@10", "en")]
    body = []
    for label, old, new, metric, lang in rows:
        cells = "".join(
            f'<td class="num" data-run="{key}" data-metric="{e(metric)}"{lang_attr(lang)} '
            f'data-errata="{which}">{stat_span(rep.runs[key].stat(metric, lang))}</td>'
            for key, which in ((old, "old"), (new, "new"))
        )
        body.append(f'<tr><th scope="row">{e(label)}</th>{cells}</tr>')
    comp = rep.comparison("A3", "A2")
    paired = ""
    if comp.ready:
        paired = (
            "<p>Corrected A minus old A on the same test questions (paired): recall@10 "
            f"<strong>{e(f.d('A3', 'A2', 'recall@10'))}</strong> "
            f"({e(f.claim('A3', 'A2', 'recall@10'))}) and MRR {e(f.d('A3', 'A2', 'mrr'))} "
            f"({e(f.claim('A3', 'A2', 'mrr'))}).</p>"
        )
    cmp_bc = ""
    if all_ok(rep, "B2", "C2", "Bp", "Cp"):
        old_gap = _lang_gap(rep, "C2", "B2")
        new_gap = _lang_gap(rep, "Cp", "Bp")
        cmp_bc = (
            "<p>The English recall@10 gap between the translation pipeline and the single-shot "
            f"pipeline was {old_gap} with the buggy embedder (C minus B) and is {new_gap} with "
            "the corrected one (C' minus B', retrieval only). These two gaps are differences of "
            "unpaired runs and carry no interval here, because the harness refuses to pair runs "
            "that differ in the judge setting.</p>"
        )
    a3 = rep.runs["A3"]
    measured = (
        "<h4>What was re-measured</h4>"
        f"<p>Corrected system A is run {e(a3.run_id or '')}"
        + (f", frozen as {e(a3.frozen_name)}" if a3.frozen_name else "")
        + f" (config {e(a3.config_hash)}); it is retrieval only (no model call, no cost) and "
        "the second use of the test split for system A. B' and C' are B and C without the answer "
        "step, with the corrected embedder, also run once on test. The old and the new rows are "
        "different runs.</p>"
        '<div class="tablewrap" tabindex="0"><table class="data"><thead><tr>'
        '<th scope="col">Test metric</th><th scope="col" class="num">Phase 2 (buggy embedder)</th>'
        '<th scope="col" class="num">Corrected embedder</th></tr></thead><tbody>'
        + "".join(body)
        + "</tbody></table></div>"
        + paired
        + cmp_bc
    )
    unknown = (
        "<h4>What is not known</h4><p>B, C and D used the same weakened first search step, so "
        "their candidate lists were worse than the model allows. Their reranker, answerer and "
        "judge numbers are what they are, but their recall, fact recall and the gaps between "
        "the systems were measured on degraded candidates and might change in either direction. "
        "A better first stage helps the single-shot system B most, which narrows the gap that "
        "query translation closes. Fact recall, faithfulness and trap results were not "
        "re-measured, and the agent D was not rerun. Read the Phase 2 answer claims as "
        "conditional on the buggy first stage until the frozen B, C and D are rerun with the "
        "corrected embedder; that decision is still open.</p>"
    )
    return head + wrong + measured + unknown


def _lang_gap(rep: Report, a: str, b: str) -> str:
    sa, sb = rep.runs[a].stat("recall@10", "en"), rep.runs[b].stat("recall@10", "en")
    if sa is None or sb is None:
        return "n/a"
    return f"{sa.mean - sb.mean:+.3f}"


def section_phase4(rep: Report) -> str:
    keys = rep.manifest["sections"]["phase4"]["runs"]
    cols = [*RETRIEVAL_COLS, *ANSWER_COLS, COST_COL]
    metrics = [("Recall@10", "recall@10"), ("MRR", "mrr"), ("Fact recall", "fact_recall"),
               ("Faithfulness", "faithfulness")]
    pending = [rep.runs[k].label for k in keys if not rep.runs[k].ok]
    state = (
        "<p>Pending rows are filled in when the run ids are added to the results manifest, after "
        "the test runs: " + e(", ".join(pending)) + ".</p>"
        if pending
        else ""
    )
    items = "".join(
        f"<li><strong>{e(rep.runs[k].label)}</strong>: {e(rep.runs[k].name)}</li>" for k in keys
    )
    return (
        '<section id="phase4"><h2>4. Phase 4: the shipped configuration</h2>'
        "<p>These runs test what the demo and the MCP server serve: the int8 model on the test "
        "split, the full pipeline with answers, and the translation pipeline with the same answer "
        f"step.</p><ul>{items}</ul>{state}"
        + run_table(rep, keys, cols, traps=True)
        + "<h3>Paired differences</h3>"
        + diff_table(rep, "phase4", metrics)
        + not_claimable(rep, "phase4", metrics)
        + "<p>The comparisons of P2 with B and of P3 with C are exploratory: B and C are the "
        "Phase 2 runs, which used the buggy first stage, so the retrieval stage differs in "
        "several ways at once (fine-tuning, the padding fix, no keyword search, int8). They show "
        "that the shipped pipeline is better than what was published in Phase 2, not which change "
        "helped. They were not all planned before the runs.</p>"
        + "</section>"
    )


def section_limits(rep: Report) -> str:
    facts = rep.facts
    n = facts.get("n_answerable", "n/a")
    groups = facts.get("n_groups", "n/a")
    lo_hi = facts.get("lang_n")
    per_lang = f"{lo_hi[0]} to {lo_hi[1]}" if lo_hi else "n/a"
    agree = facts.get("judge_agreement")
    agree_txt = f"{agree * 100:.0f}%" if agree is not None else "an unknown share"
    judge = facts.get("judge", "the judge model")
    n_traps = facts.get("n_traps", "n/a")
    groups_html = {
        "The questions and the data": [
            "The benchmark questions were written by an LLM from the text of one article, and "
            "the training questions by the same kind of model. The results show how well "
            "systems find the article a question was written from, not how people really ask. "
            "Part of the fine-tuning gain may be the model learning that writing style.",
            f"The test set is small: {n} answerable questions in {groups} independent groups "
            f"(a question and its translations count once), and {per_lang} per language. Only "
            "large differences can be claimed.",
            "Nobody on the project reads German, French and Italian well enough to have reviewed "
            "the training questions; only a sample was audited, by a second AI model. No Swiss-law "
            "expert reviewed the benchmark, its gold facts or any answer; the one human reviewer "
            "reads English only and has no legal training.",
            "Some short training articles are partly contained in longer test gold articles, so "
            "a few test questions may be easier for the tuned model than for a fresh one. The "
            "Phase 3 report checks that the headline gain does not depend on them.",
        ],
        "How answers are scored": [
            f"One judge model ({judge}) scores all answers. It agreed with one human on "
            f"{agree_txt} of the facts in a small sample, so differences of a few points in fact "
            "recall or faithfulness are noise that the intervals do not include.",
            f"Trap questions are few ({n_traps}). They are reported as counts, not ranked.",
        ],
        "What was measured": [
            "The corrected systems of Phase 3 (A corrected, B', C', T1, T2) were measured on "
            "retrieval only. Answers were measured for two pipelines in Phase 4 (the shipped one "
            "and the translation pipeline), once each; the effect of the tuned embedder on "
            "answers alone, apart from the other changes to the first stage, was not isolated.",
            "System D, the agent, was not re-measured with the corrected embedder, and the "
            "Phase 2 answer-level results for B, C and D predate the fix.",
            "After tuning, embedding alone beat the hybrid with keyword search on the "
            "development questions, but that was not tested further and was not run on the test "
            "split. T1 and T2 use embedding only.",
            "Each configuration was run once, with one training seed; run-to-run variation is "
            "not measured. Model answers are not bit-reproducible, so numbers replay from a "
            "cache of recorded responses, not from scratch.",
            "The development split was used to choose configurations; the test split was used "
            "once per frozen configuration, twice for system A, and B' was frozen after the T1, "
            "T2 and C' results were known.",
        ],
        "Speed and cost": [
            "Latency comes from one Windows laptop on one day with serial calls; cached model "
            "calls are replayed, so it shows the relative cost of pipelines, not provider speed. "
            f"Costs use the price table of {facts.get('price_date', 'unknown')}.",
        ],
        "Provenance": [
            "The trained weights are not stored in git. The configuration of T1 records a digest "
            "of the weights, and the weights are published on Hugging Face. One training run's "
            "configuration does not record the code version.",
            "A planned feature of the synthetic training questions (a fixed asker role per "
            "question) was never applied: the generator invented the roles.",
        ],
    }
    out = ['<section id="limits"><h2>5. Limitations</h2>']
    for title, items in groups_html.items():
        out.append(f"<h3>{e(title)}</h3><ul>")
        out.extend(f"<li>{e(i)}</li>" for i in items)
        out.append("</ul>")
    out.append("</section>")
    return "".join(out)


def section_reproduce(rep: Report) -> str:
    rows = []
    for row in rep.manifest["reproduce"]:
        cmds: list[str] = []
        for template in row["commands"]:
            cmds.extend(expand_commands(rep, expand_text(rep, template)))
        if cmds:
            code = "".join(f'<code class="cmd">{e(c)}</code>' for c in cmds)
        else:
            code = '<span class="na">available once the runs exist</span>'
        rows.append(
            "<tr>"
            f'<th scope="row">{e(row["table"])}</th>'
            f'<td data-label="Cost">{tag_html(row["tag"])}</td>'
            f'<td data-label="Command">{code}</td>'
            f'<td data-label="Needs">{e(expand_text(rep, row["needs"]))}'
            + (f' <span class="fine">{e(expand_text(rep, row["note"]))}</span>'
               if row.get("note") else "")
            + "</td></tr>"
        )
    return (
        '<section id="reproduce"><h2>6. Reproduce each number</h2>'
        "<p>Every table on this page is produced by code from the committed run folders. "
        "<span class=\"tag tag-free\">FREE</span> needs no key and no GPU. "
        "<span class=\"tag tag-keys\">KEYS</span> needs provider keys in <code>.env</code> and "
        "spends money, unless the git-ignored response cache replays it. "
        "<span class=\"tag tag-gpu\">GPU</span> is training on a Kaggle T4.</p>"
        '<div class="tablewrap" tabindex="0"><table class="data repro"><thead><tr>'
        '<th scope="col">Table</th><th scope="col">Cost</th><th scope="col">Command</th>'
        '<th scope="col">Needs</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div></section>"
    )


def link(href: str | None, text: str) -> str:
    if not href:
        return f'<span class="soon">{e(text)}</span>'
    return f'<a href="{e(href, quote=True)}">{e(text)}</a>'


def section_links(rep: Report) -> str:
    links = rep.manifest["links"]
    demo: Any = links.get("demo") or {}
    items = [
        link(links.get("repo"), "Source code and data (GitHub)"),
        link(links.get("model"), "Fine-tuned model (Hugging Face)"),
        link(demo.get("url"), demo.get("text", "recorded examples (no live demo)")),
        link(links.get("mcp_docs"), "MCP server documentation"),
        link(links.get("fedlex"), "Fedlex, the official source of the law"),
    ]
    lis = "".join(f"<li>{i}</li>" for i in items)
    return f'<section id="links"><h2>7. Links</h2><ul class="links">{lis}</ul></section>'


NAV = (
    ("answer", "Answer"), ("phase3", "Phase 3"), ("phase2", "Phase 2"), ("phase4", "Phase 4"),
    ("limits", "Limitations"), ("reproduce", "Reproduce"), ("links", "Links"),
)
EXAMPLES_PAGE = "examples.html"
POST_PAGE = "post.html"
POST_LINK = f'<a href="{POST_PAGE}">Write-up</a>'


def notices_html() -> str:
    return (
        f"<p>{e(NOT_OFFICIAL_EN)} <span lang=\"de\">{e(NOT_OFFICIAL_DE)}</span> "
        f"{e(NOT_LEGAL_ADVICE)}</p>"
    )


def page_shell(title: str, description: str, nav: str, main: str, foot: str) -> str:
    """Document skeleton shared by every page: notices at the top and bottom, one stylesheet."""
    notices = notices_html()
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="color-scheme" content="light dark">'
        f"<title>{e(title)}</title>"
        f'<meta name="description" content="{e(description, quote=True)}">'
        '<link rel="stylesheet" href="style.css"></head><body>'
        f'<header class="top"><div class="wrap"><div class="notice">{notices}</div>'
        f"<h1>{e(title)}</h1><nav aria-label=\"Sections\">{nav}</nav></div></header>"
        f'<main class="wrap">{main}</main>'
        f'<footer class="wrap"><div class="notice">{notices}</div>'
        f'<p class="fine">{foot}</p></footer>'
        "</body></html>\n"
    )


def render_page(rep: Report, has_post: bool = False) -> str:
    nav = "".join(f'<a href="#{i}">{e(t)}</a>' for i, t in NAV)
    nav += f'<a href="{EXAMPLES_PAGE}">Recorded examples</a>'
    nav += POST_LINK if has_post else ""
    main = (
        f'<section id="answer"><h2>1. The answer in short</h2>{headline(rep)}</section>'
        f"{section_phase3(rep)}{section_phase2(rep)}{section_phase4(rep)}"
        f"{section_limits(rep)}{section_reproduce(rep)}{section_links(rep)}"
    )
    return page_shell(
        rep.manifest.get("title", "Results"),
        "Results of a cross-lingual retrieval benchmark over Swiss federal law, with every "
        "number rebuilt from the evaluation harness.",
        nav,
        main,
        "Statute text is not shown on this page. Built by "
        "<code>uv run python -m fedlex_bench.site build</code> from the committed files under "
        "<code>eval/</code>; no script, font or tracker is loaded from elsewhere.",
    )
