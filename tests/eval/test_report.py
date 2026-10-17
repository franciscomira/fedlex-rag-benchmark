import json

from fedlex_bench.eval.report import render_table


def _run(tmp_path, name, recall, lo, hi):
    d = tmp_path / name
    d.mkdir()
    stat = {"n": 40, "mean": recall, "lo": lo, "hi": hi}
    block = {"recall@1": stat, "recall@5": stat, "recall@10": stat, "mrr": stat,
             "fact_recall": {"n": 0}, "cost_usd": {"n": 40, "mean": 0.01, "lo": 0, "hi": 0.02},
             "latency_p50": 0.5, "latency_p95": 1.5}
    metrics = {"n_items": 63, "overall": block, "traps": {"correct": 0, "n": 0}}
    (d / "metrics.json").write_text(json.dumps(metrics))
    (d / "config.json").write_text(json.dumps({"system": {"retriever": f"sys-{name}"}}))
    return d


def test_table_has_one_row_per_run_with_cis(tmp_path):
    table = render_table([_run(tmp_path, "a", 0.5, 0.35, 0.65), _run(tmp_path, "b", 0.25, 0.1, 0.4)])
    lines = table.splitlines()
    assert len(lines) == 4  # header, separator, two rows
    assert "sys-a" in lines[2] and "0.50 [0.35, 0.65]" in lines[2]
    assert "n/a" in lines[2]  # fact recall absent for retrieval-only runs
