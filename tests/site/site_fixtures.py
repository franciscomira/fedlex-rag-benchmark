"""A tiny fake project root with run folders, for the results page tests."""

from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path

LINKS = {
    "repo": "https://github.com/franciscomira/fedlex-rag-benchmark",
    "model": "https://huggingface.co/franciscomira/fedlex-e5-small",
    "demo": {"text": "recorded examples (no live demo)", "url": "examples.html"},
    "mcp_docs": "https://github.com/franciscomira/fedlex-rag-benchmark/blob/main/docs/mcp.md",
    "fedlex": "https://www.fedlex.admin.ch",
}

# Distinctive values so a wrong source cannot match by accident.
FAKE_ANSWER_TEXT = "SECRET-ANSWER-TEXT-ABOUT-A-STATUTE"


def stat(mean: float, lo: float, hi: float, n: int = 12) -> dict:
    return {"mean": mean, "lo": lo, "hi": hi, "n": n}


def metrics_json(base: float, answers: bool = False, judge_failures: int = 0) -> dict:
    """metrics.json with the keys the page reads; every number is derived from ``base``."""
    overall = {
        "recall@1": stat(base - 0.30, base - 0.40, base - 0.20),
        "recall@10": stat(base, base - 0.08, base + 0.07),
        "mrr": stat(base - 0.15, base - 0.22, base - 0.07),
        "cost_usd": stat(0.0012 * base, 0.001 * base, 0.0014 * base),
        "latency_p50": 1.5 * base,
        "latency_p95": 4.5 * base,
        "fact_recall": stat(base - 0.1, base - 0.2, base) if answers else {"n": 0},
        "faithfulness": stat(0.85, 0.8, 0.9) if answers else {"n": 0},
        "cit_precision": stat(0.55, 0.5, 0.6) if answers else {"n": 0},
        "over_refusal": stat(0.11, 0.05, 0.17) if answers else {"n": 0},
    }
    return {
        "overall": overall,
        "by_query_lang": {
            lang: {"recall@10": stat(base - 0.05 * i, base - 0.15, base + 0.02, n=3 + i)}
            for i, lang in enumerate(("de", "en", "fr", "it"))
        },
        "originals_only": {},
        "n_items": 14,
        "n_answerable": 12,
        "judge_failures": {"fact": judge_failures, "faith": 0, "trap": 0},
        "traps": {"correct": 1, "n": 2} if answers else {"correct": 0, "n": 0},
    }


def per_item_rows(offset: float, answers: bool = False) -> list[dict]:
    """12 answerable items in 6 groups plus 2 traps; values vary so a bootstrap has spread."""
    rows = []
    for i in range(12):
        group = f"g{i // 2}"
        r10 = 1.0 if (i * 7 + int(offset * 10)) % 3 else 0.0
        row = {
            "id": f"q{i:02d}", "group": group, "is_trap": False, "qtype": "single",
            "query_lang": ("de", "en", "fr", "it")[i % 4],
            "recall@10": r10, "recall@1": float(r10 and i % 2), "mrr": r10 / (1 + i % 3),
            "cost_usd": 0.001 + 0.0001 * (i % 4) + offset / 1000,
            "fact_recall": (0.5 + 0.5 * ((i + int(offset * 10)) % 2)) if answers else None,
            "faithfulness": 0.8 if answers else None,
        }
        if answers:
            row["answer"] = FAKE_ANSWER_TEXT
        rows.append(row)
    for j in range(2):
        rows.append({"id": f"t{j}", "group": f"t{j}", "is_trap": True, "qtype": "trap",
                     "recall@10": None, "mrr": None, "fact_recall": None, "faithfulness": None,
                     "cost_usd": 0.001, "query_lang": "de"})
    return rows


def write_run(root: Path, run_id: str, base: float, *, answers: bool = False,
              judge: str = "none", split: str = "test", config_hash: str | None = None,
              completed: bool = True, offset: float = 0.0) -> str:
    folder = root / "eval" / "results" / run_id
    folder.mkdir(parents=True)
    h = config_hash or run_id[-6:]
    system = {"kind": "pipeline", "retriever": "hybrid/all_lang/article",
              "answerer": "answer/claude-haiku-5-5" if answers else None,
              "reranker": "llm_rerank/claude-haiku-5-5"}
    (folder / "config.json").write_text(
        json.dumps({"config_hash": h, "split": split, "judge_model": judge, "system": system}),
        encoding="utf-8")
    (folder / "metrics.json").write_text(json.dumps(metrics_json(base, answers)), encoding="utf-8")
    rows = per_item_rows(offset, answers)
    (folder / "per_item.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    if split == "test":
        log = root / "eval" / "test_runs.log"
        lines = [{"config_hash": h, "run_id": run_id, "status": "started"}]
        if completed:
            lines.append({"config_hash": h, "run_id": run_id, "status": "completed"})
        with log.open("a", encoding="utf-8") as fh:
            fh.writelines(json.dumps(x) + "\n" for x in lines)
    return run_id


def make_root(tmp_path: Path, p1_id: str | None = None) -> tuple[Path, Path]:
    """Fake project with runs for every manifest key except P1..P3; returns (root, manifest)."""
    root = tmp_path / "proj"
    (root / "eval").mkdir(parents=True)
    (root / "site").mkdir()
    (root / "config.toml").write_text(
        "[budget]\nmonthly_usd = 25.0\nconfirm_usd = 5.0\n", encoding="utf-8")
    (root / "eval" / "prices.json").write_text(json.dumps({"retrieved": "2026-10-17"}))
    (root / "eval" / "frozen_configs.json").write_text(
        json.dumps({"T1": {"hash": "aaaaaa", "judge": None, "system": {}}}))
    ids = {
        "A2": ("20_test_a2aaaa", 0.40, False, "none"),
        "B2": ("20_test_b2bbbb", 0.60, True, "judge-x"),
        "C2": ("20_test_c2cccc", 0.77, True, "judge-x"),
        "D2": ("20_test_d2dddd", 0.91, True, "judge-x"),
        "A3": ("30_test_a3aaaa", 0.55, False, "none"),
        "Bp": ("30_test_bpbbbb", 0.71, False, "none"),
        "T1": ("30_test_aaaaaa", 0.8123, False, "none"),
        "T2": ("30_test_t2tttt", 0.89, False, "none"),
        "Cp": ("30_test_cpcccc", 0.87, False, "none"),
    }
    runs = {}
    for n, (key, (rid, base, ans, judge)) in enumerate(ids.items()):
        write_run(root, rid, base, answers=ans, judge=judge, offset=n * 0.1,
                  config_hash=rid[-6:])
        runs[key] = {"id": rid, "label": key, "name": f"System {key}"}
    for key in ("P1", "P2", "P3"):
        runs[key] = {"id": None, "label": key, "name": f"Phase four {key}"}
    if p1_id:
        runs["P1"]["id"] = p1_id
    manifest = {
        "schema": 1, "title": "Fake results", "links": LINKS, "runs": runs,
        "sections": {
            "phase3": {"runs": ["A3", "Bp", "T1", "T2", "Cp"],
                       "comparisons": [["T1", "A3"], ["T2", "Cp"]]},
            "phase2": {"runs": ["A2", "B2", "C2", "D2"],
                       "comparisons": [["C2", "B2"], ["D2", "C2"], ["B2", "A2"]]},
            "errata": {"runs": ["A2", "A3", "B2", "Bp", "C2", "Cp"], "comparisons": [["A3", "A2"]]},
            "phase4": {"runs": ["P1", "P2", "P3"], "comparisons": [["P1", "T1"]]},
        },
        "reproduce": [
            {"table": "Table one", "tag": "FREE",
             "commands": ["uv run python -m fedlex_bench.eval report {runs:phase3}",
                          "{compares:phase3}"],
             "needs": "Committed files. Cap {cap_usd}, confirm {confirm_usd}."},
            {"table": "Phase four", "tag": "FREE",
             "commands": ["uv run python -m fedlex_bench.eval report {runs:phase4}"],
             "needs": "x"},
        ],
    }
    path = root / "site" / "results_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return root, path


class Cells(HTMLParser):
    """Collect table cells that carry data-run/data-metric (or data-cmp) as normalised text."""

    def __init__(self) -> None:
        super().__init__()
        self.cells: dict[tuple, str] = {}
        self._key: tuple | None = None
        self._buf: list[str] = []
        self.pending_rows: list[str] = []
        self._in_pending_row = False
        self._row_buf: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "td" and ("data-metric" in a):
            self._key = (a.get("data-run") or a.get("data-cmp"), a["data-metric"],
                         a.get("data-lang"), a.get("data-errata"))
            self._buf = []
        if tag == "tr" and "pending" in (a.get("class") or ""):
            self._in_pending_row = True
            self._row_buf = []

    def handle_endtag(self, tag):
        if tag == "td" and self._key is not None:
            text = " ".join("".join(self._buf).split())
            self.cells[self._key] = text
            self._key = None
        if tag == "tr" and self._in_pending_row:
            self.pending_rows.append(" ".join("".join(self._row_buf).split()))
            self._in_pending_row = False

    def handle_data(self, data):
        if self._key is not None:
            self._buf.append(data)
        if self._in_pending_row:
            self._row_buf.append(data)


def parse_cells(html_text: str) -> Cells:
    p = Cells()
    p.feed(html_text)
    return p


def fmt2(s: dict) -> str:
    return f"{s['mean']:.2f} [{s['lo']:.2f}, {s['hi']:.2f}]"
