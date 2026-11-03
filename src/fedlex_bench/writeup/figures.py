"""Hand-written SVG figures, built from the same run data as the results page.

Each figure comes in two forms: ``inline`` for post.html (transparent, takes its colours from the
page's CSS variables) and ``standalone`` for docs/writeup/figures/NAME.svg (own background and
its own light and dark colours, so it reads on GitHub). Colours are the Okabe-Ito family (safe for
colour-blind readers); every point is also told apart by marker shape and a direct label.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from math import floor, hypot, log10

from fedlex_bench.site.data import Run, fmt_seconds

WIDTH = 388

# Run key -> family (colour and marker shape) and short direct label.
FAMILY = {"A3": "base", "Bp": "llm", "Cp": "llm", "P3": "llm",
          "T1": "tuned", "T2": "tuned", "P1": "shipped", "P2": "shipped"}
SHORT = {
    "A3": "A: hybrid search", "Bp": "B': hybrid + reranker", "T1": "T1: tuned embedder",
    "T2": "T2: tuned + reranker", "Cp": "C': LLM translation", "P1": "P1: int8 embedder",
    "P2": "P2: full pipeline", "P3": "P3: translation + answers",
}

# Shorter labels for the scatter plot, where text competes with points for room.
POINT_LABEL = {
    "A3": "A: hybrid", "T1": "T1: tuned", "T2": "T2: tuned + LLM", "Cp": "C': translation",
    "P2": "P2: shipped", "P3": "P3: translation",
}

RECALL_KEYS = ["A3", "Bp", "T1", "T2", "Cp", "P2"]
COST_KEYS = ["A3", "T1", "T2", "Cp", "P2", "P3"]
ENGLISH_KEYS = ["A3", "T1", "T2", "Cp"]
FIGURE_NAMES = ("fig_recall", "fig_cost_latency", "fig_english")

STYLE = (
    "svg.wf{--wf-fg:var(--text,#1c1b19);--wf-mut:var(--muted,#5d5a54);"
    "--wf-line:var(--line,#d9d5cc);--wf-bg:var(--surface,#ffffff);"
    "--wf-base:#6b6b6b;--wf-llm:#b85c00;--wf-tuned:#0072b2;--wf-shipped:#00795a}"
    "@media (prefers-color-scheme:dark){svg.wf{--wf-fg:var(--text,#ece9e2);"
    "--wf-mut:var(--muted,#aaa69c);--wf-line:var(--line,#3a3832);--wf-bg:var(--surface,#1d1c18);"
    "--wf-base:#b5b5b5;--wf-llm:#f2a43a;--wf-tuned:#5eb4ea;--wf-shipped:#3fcfa4}}"
    "svg.wf text{fill:var(--wf-fg);font-family:system-ui,-apple-system,'Segoe UI',Roboto,"
    "sans-serif;font-size:12.5px;paint-order:stroke;stroke:var(--wf-bg);stroke-width:3px;"
    "stroke-linejoin:round}"
    "svg.wf .wf-ttl{font-size:14px;font-weight:700}"
    "svg.wf .wf-sub{fill:var(--wf-mut);font-size:11.5px}"
    "svg.wf .wf-mut{fill:var(--wf-mut)}svg.wf .wf-b{font-weight:700}"
    "svg.wf .wf-bgrect{fill:var(--wf-bg);stroke:var(--wf-line)}"
    "svg.wf .wf-grid{stroke:var(--wf-line);stroke-width:1;fill:none}"
    "svg.wf .wf-axis{stroke:var(--wf-mut);stroke-width:1;fill:none}"
    "svg.wf .wf-whisk{stroke-width:2.5;fill:none;stroke-linecap:round}"
    "svg.wf .wf-base{fill:var(--wf-base);stroke:var(--wf-base)}"
    "svg.wf .wf-llm{fill:var(--wf-llm);stroke:var(--wf-llm)}"
    "svg.wf .wf-tuned{fill:var(--wf-tuned);stroke:var(--wf-tuned)}"
    "svg.wf .wf-shipped{fill:var(--wf-shipped);stroke:var(--wf-shipped)}"
)


class FigureError(Exception):
    """A figure could not be built (a run is pending or lacks a number)."""


@dataclass(frozen=True)
class Figure:
    name: str
    title: str
    desc: str
    inline: str
    standalone: str


def esc(text: str) -> str:
    return escape(text, quote=False)


def f1(value: float) -> str:
    return f"{value:.1f}"


def _short(key: str, run: Run) -> str:
    return SHORT.get(key, run.label)


def _need(runs: dict[str, Run], key: str) -> Run:
    run = runs.get(key)
    if run is None:
        raise FigureError(f"unknown run '{key}'")
    if not run.ok:
        raise FigureError(f"run '{key}' is pending ({run.reason or 'no run yet'})")
    return run


def marker(family: str, x: float, y: float) -> str:
    cls = f"wf-{family}"
    if family == "llm":
        return f'<rect class="{cls}" x="{f1(x - 5)}" y="{f1(y - 5)}" width="10" height="10"/>'
    if family == "tuned":
        pts = f"{f1(x)},{f1(y - 7)} {f1(x + 7)},{f1(y)} {f1(x)},{f1(y + 7)} {f1(x - 7)},{f1(y)}"
        return f'<polygon class="{cls}" points="{pts}"/>'
    if family == "shipped":
        pts = f"{f1(x)},{f1(y - 6.5)} {f1(x + 6.5)},{f1(y + 5)} {f1(x - 6.5)},{f1(y + 5)}"
        return f'<polygon class="{cls}" points="{pts}"/>'
    return f'<circle class="{cls}" cx="{f1(x)}" cy="{f1(y)}" r="5.5"/>'


def wrap(name: str, title: str, desc: str, height: float, body: str, standalone: bool) -> str:
    tid, did = f"{name}-title", f"{name}-desc"
    style = f"<style>{STYLE}</style>"
    bg = (f'<rect class="wf-bgrect" x="0.5" y="0.5" width="{WIDTH - 1}" height="{f1(height - 1)}"'
          ' rx="6"/>') if standalone else ""
    ns = ' xmlns="http://www.w3.org/2000/svg"' if standalone else ""
    return (
        f'<svg class="wf"{ns} viewBox="0 0 {WIDTH} {f1(height)}" role="img" '
        f'aria-labelledby="{tid} {did}">'
        f'<title id="{tid}">{esc(title)}</title><desc id="{did}">{esc(desc)}</desc>'
        f"{style}{bg}{body}</svg>"
    )


def head(title: str, subtitle: str) -> str:
    return (f'<text class="wf-ttl" x="14" y="22">{esc(title)}</text>'
            f'<text class="wf-sub" x="14" y="39">{esc(subtitle)}</text>')


# ---------------------------------------------------------------- interval chart


def interval_figure(name: str, runs: dict[str, Run], keys: list[str], lang: str | None,
                    title: str, subtitle: str) -> Figure:
    x0, x1, vx = 14.0, 240.0, 252.0
    y0, row_h = 58.0, 46.0
    rows = []
    desc_parts = []
    for i, key in enumerate(keys):
        run = _need(runs, key)
        stat = run.stat("recall@10", lang)
        if stat is None:
            raise FigureError(f"run '{key}' has no recall@10" + (f" for {lang}" if lang else ""))
        fam = FAMILY.get(key, "base")
        top = y0 + i * row_h
        cy = top + 31
        sx = lambda v: x0 + (x1 - x0) * v  # noqa: E731
        rows.append(
            f'<text x="{f1(x0)}" y="{f1(top + 13)}"><tspan class="wf-b">{esc(run.label)}</tspan>'
            f'<tspan class="wf-mut"> {esc(_short(key, run).split(": ", 1)[-1])}</tspan></text>'
            f'<line class="wf-whisk wf-{fam}" x1="{f1(sx(stat.lo))}" y1="{f1(cy)}" '
            f'x2="{f1(sx(stat.hi))}" y2="{f1(cy)}"/>'
            f'<line class="wf-whisk wf-{fam}" x1="{f1(sx(stat.lo))}" y1="{f1(cy - 5)}" '
            f'x2="{f1(sx(stat.lo))}" y2="{f1(cy + 5)}"/>'
            f'<line class="wf-whisk wf-{fam}" x1="{f1(sx(stat.hi))}" y1="{f1(cy - 5)}" '
            f'x2="{f1(sx(stat.hi))}" y2="{f1(cy + 5)}"/>'
            + marker(fam, sx(stat.mean), cy)
            + f'<text x="{f1(vx)}" y="{f1(cy + 4)}"><tspan class="wf-b">{stat.mean:.2f}</tspan>'
            f'<tspan class="wf-mut" dx="4">[{stat.lo:.2f}, {stat.hi:.2f}]</tspan></text>'
        )
        desc_parts.append(
            f"{run.label} {stat.mean:.2f}, 95% interval {stat.lo:.2f} to {stat.hi:.2f}")
    bottom = y0 + len(keys) * row_h - 4
    grid = "".join(
        f'<line class="wf-grid" x1="{f1(x0 + (x1 - x0) * v)}" y1="{f1(y0 - 4)}" '
        f'x2="{f1(x0 + (x1 - x0) * v)}" y2="{f1(bottom)}"/>'
        for v in (0.0, 0.25, 0.5, 0.75, 1.0)
    )
    ticks = "".join(
        f'<text class="wf-sub" text-anchor="middle" x="{f1(x0 + (x1 - x0) * v)}" '
        f'y="{f1(bottom + 15)}">{label}</text>'
        for v, label in ((0.0, "0"), (0.5, "0.5"), (1.0, "1"))
    )
    height = bottom + 24
    body = head(title, subtitle) + grid + "".join(rows) + ticks
    desc = f"{title}. {subtitle} " + "; ".join(desc_parts) + "."
    return Figure(name, title, desc,
                  wrap(name, title, desc, height, body, False),
                  wrap(name, title, desc, height, body, True))


# ---------------------------------------------------------------- cost and latency


def nice_ticks(vmax: float, target: int = 4) -> tuple[list[float], int]:
    """Ticks from 0 in a 1-2-5 step, and the decimals to print them with."""
    raw = vmax / target
    mag = 10 ** floor(log10(raw))
    step = next(m * mag for m in (1, 2, 5, 10) if m * mag >= raw)
    count = int(vmax / step + 1e-9)
    decimals = max(0, -floor(log10(step) + 1e-9))
    return [round(step * k, 10) for k in range(count + 1)], decimals


def log_ticks(vmin: float, vmax: float) -> list[float]:
    out = []
    for p in range(floor(log10(vmin)) - 1, floor(log10(vmax)) + 2):
        for m in (1, 2, 5):
            v = m * 10.0**p
            if vmin <= v <= vmax:
                out.append(v)
    return out


def _tick_text(v: float) -> str:
    return f"{v:g}"


def place_label(x: float, y: float, w: float, placed: list[tuple[float, float, float, float]],
                bounds: tuple[float, float, float, float],
                others: list[tuple[float, float]] = ()):
    """Label position (x, y, anchor, box, dy): the first candidate, in order of preference, that
    stays inside ``bounds`` and overlaps no earlier label or marker; when none is free, the one
    that overlaps least."""
    cands = []
    near = [oy for ox, oy in others if abs(ox - x) < 14 and abs(oy - y) < 16 and (ox, oy) != (x, y)]
    if near:  # nearly the same spot: labels go apart, the upper marker's label up
        up = -1 if y <= min(near) else 1
        offsets = (15 * up, 30 * up, 45 * up, -15 * up, -30 * up)
    else:
        offsets = (0, -15, 15, -30, 30, -45, 45)
    for dy in offsets:
        cands.append((x + 10, y + 4 + dy, "start", (x + 10, y + dy - 8, x + 10 + w, y + dy + 8),
                      dy))
        cands.append((x - 10, y + 4 + dy, "end", (x - 10 - w, y + dy - 8, x - 10, y + dy + 8), dy))
    for dy in (-24, 24):  # above or below, centred or ending/starting at the marker
        for anchor, x_from, x_to in (("middle", x - w / 2, x + w / 2), ("end", x + 8 - w, x + 8),
                                     ("start", x - 8, x - 8 + w)):
            lx = {"middle": x, "end": x + 8, "start": x - 8}[anchor]
            cands.append((lx, y + dy + 4, anchor, (x_from, y + dy - 8, x_to, y + dy + 8), dy))

    def overlap(box: tuple[float, float, float, float]) -> float:
        """Area overlapping earlier boxes, plus a heavy penalty for leaving the bounds."""
        area = sum(max(0.0, min(box[2], b[2]) - max(box[0], b[0]))
                   * max(0.0, min(box[3], b[3]) - max(box[1], b[1])) for b in placed)
        out = (max(0.0, bounds[0] - box[0]) + max(0.0, box[2] - bounds[2])
               + max(0.0, bounds[1] - box[1]) + max(0.0, box[3] - bounds[3]))
        return area + 1000 * out

    best = min(cands, key=lambda c: (overlap(c[3]), cands.index(c)))  # first of the least bad
    return best


def cost_latency_figure(name: str, runs: dict[str, Run], keys: list[str]) -> Figure:
    pts = []
    for key in keys:
        run = _need(runs, key)
        cost, lat = run.stat("cost_usd"), run.scalar("latency_p50")
        if cost is None or lat is None or lat <= 0:
            raise FigureError(f"run '{key}' has no cost or median latency")
        pts.append((key, run, cost.mean, lat))
    title = "Cost and speed per question"
    subtitle = "Each point is one system; lower left is cheaper and faster."
    px0, px1, py0, py1 = 56.0, 372.0, 70.0, 276.0
    cmax = max(p[2] for p in pts) or 1.0
    xticks, dec = nice_ticks(cmax * 1.08)
    xmax = xticks[-1] if xticks[-1] >= cmax * 1.02 else xticks[-1] + (xticks[1] - xticks[0])
    if xticks[-1] < xmax:
        xticks.append(xmax)
    lats = [p[3] for p in pts]
    use_log = max(lats) / min(lats) > 15
    if use_log:
        lo, hi = min(lats) / 4, max(lats) * 3.2
        yticks = log_ticks(lo, hi)

        def sy(v: float) -> float:
            return py1 - (py1 - py0) * (log10(v) - log10(lo)) / (log10(hi) - log10(lo))
        ylabel = "Median latency (seconds, log scale)"
    else:
        yticks, _ = nice_ticks(max(lats) * 1.1)
        hi = yticks[-1]

        def sy(v: float) -> float:
            return py1 - (py1 - py0) * v / hi
        ylabel = "Median latency (seconds)"

    def sx(v: float) -> float:
        return px0 + (px1 - px0) * v / xmax

    grid, axes = [], []
    for v in yticks:
        y = sy(v)
        grid.append(f'<line class="wf-grid" x1="{f1(px0)}" y1="{f1(y)}" x2="{f1(px1)}" '
                    f'y2="{f1(y)}"/>')
        axes.append(f'<text class="wf-sub" text-anchor="end" x="{f1(px0 - 6)}" '
                    f'y="{f1(y + 4)}">{_tick_text(v)}</text>')
    for v in xticks:
        x = sx(v)
        grid.append(f'<line class="wf-grid" x1="{f1(x)}" y1="{f1(py0)}" x2="{f1(x)}" '
                    f'y2="{f1(py1)}"/>')
        axes.append(f'<text class="wf-sub" text-anchor="middle" x="{f1(x)}" '
                    f'y="{f1(py1 + 16)}">{v:.{dec}f}</text>')
    axes.append(f'<line class="wf-axis" x1="{f1(px0)}" y1="{f1(py1)}" x2="{f1(px1)}" '
                f'y2="{f1(py1)}"/><line class="wf-axis" x1="{f1(px0)}" y1="{f1(py0)}" '
                f'x2="{f1(px0)}" y2="{f1(py1)}"/>')
    axes.append(f'<text class="wf-sub" x="14" y="{f1(py0 - 10)}">{esc(ylabel)}</text>')
    axes.append(f'<text class="wf-sub" text-anchor="middle" x="{f1((px0 + px1) / 2)}" '
                f'y="{f1(py1 + 36)}">Cost per question (USD, judging not included)</text>')

    # Direct labels: first free spot near the marker, with a leader line when it is moved away.
    placed: list[tuple[float, float, float, float]] = [
        (sx(c) - 9, sy(lat) - 9, sx(c) + 9, sy(lat) + 9) for _, _, c, lat in pts]
    placed.append((14.0, py0 - 24, 14.0 + len(ylabel) * 6.0, py0 - 4))  # the axis caption
    bounds = (4.0, py0 - 14, WIDTH - 4.0, py1 - 2)
    centres = [(sx(c), sy(lat)) for _, _, c, lat in pts]
    marks, labels, desc_parts = [], [], []
    for key, run, c, lat in pts:
        x, y = sx(c), sy(lat)
        fam = FAMILY.get(key, "base")
        text = POINT_LABEL.get(key, _short(key, run))
        lx, ly, anchor, box, dy = place_label(x, y, len(text) * 6.3 + 6, placed, bounds, centres)
        placed.append(box)
        marks.append(marker(fam, x, y))
        if abs(dy) > 6:  # a label that is not beside its marker gets a leader line
            qx, qy = min(max(x, box[0]), box[2]), min(max(y, box[1]), box[3])
            dist = hypot(qx - x, qy - y) or 1.0
            sx0, sy0 = x + 7 * (qx - x) / dist, y + 7 * (qy - y) / dist
            labels.append(f'<line class="wf-grid" x1="{f1(sx0)}" y1="{f1(sy0)}" '
                          f'x2="{f1(qx)}" y2="{f1(qy)}"/>')
        labels.append(f'<text class="wf-b" x="{f1(lx)}" y="{f1(ly)}" text-anchor="{anchor}">'
                      f"{esc(text)}</text>")
        desc_parts.append(f"{run.label} {c:.4f} USD, median {fmt_seconds(lat)} s")
    height = py1 + 46
    body = head(title, subtitle) + "".join(grid) + "".join(axes) + "".join(marks) + "".join(labels)
    desc = (f"{title}. Horizontal axis: cost per question in US dollars. Vertical axis: median "
            f"latency in seconds{', on a log scale' if use_log else ''}. "
            + "; ".join(desc_parts) + ".")
    return Figure(name, title, desc,
                  wrap(name, title, desc, height, body, False),
                  wrap(name, title, desc, height, body, True))


# ---------------------------------------------------------------- the three figures


def make_figures(runs: dict[str, Run]) -> dict[str, Figure]:
    return {
        "fig_recall": interval_figure(
            "fig_recall", runs, RECALL_KEYS, None, "Recall@10 on the test questions",
            "Dot: mean. Bar: 95% interval."),
        "fig_cost_latency": cost_latency_figure("fig_cost_latency", runs, COST_KEYS),
        "fig_english": interval_figure(
            "fig_english", runs, ENGLISH_KEYS, "en", "Recall@10, English questions only",
            "No article is in English. Dot: mean. Bar: 95% interval."),
    }
