"""The ``python -m fedlex_bench.benchmark`` commands, and the whole chain on a synthetic corpus."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import pytest

from benchmark_fixtures import make_corpus, words
from fedlex_bench.benchmark import cli
from fedlex_bench.benchmark.common import read_jsonl


@pytest.fixture(scope="module")
def parquet(tmp_path_factory):
    path = tmp_path_factory.mktemp("corpus") / "articles.parquet"
    make_corpus().to_parquet(path, index=False)
    return path


@pytest.fixture(scope="module")
def planned(tmp_path_factory, parquet):
    out = tmp_path_factory.mktemp("plan") / "slots.jsonl"
    assert cli.main(["plan", "--parquet", str(parquet), "--out", str(out)]) == 0
    return out


def test_plan_command_writes_slots_and_prints_the_hash(tmp_path, parquet, capsys):
    out = tmp_path / "plan" / "slots.jsonl"
    assert cli.main(["plan", "--parquet", str(parquet), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    assert f"slots.jsonl sha256 {sha}" in text
    assert "PLAN: 258 slots" in text and "cross_ref allocation" in text
    assert len(read_jsonl(out)) == 258


def test_plan_command_is_idempotent(tmp_path, parquet, planned):
    again = tmp_path / "slots.jsonl"
    cli.main(["plan", "--parquet", str(parquet), "--out", str(again)])
    assert again.read_bytes() == planned.read_bytes()


def test_plan_command_reports_a_stop_condition(tmp_path, capsys):
    path = tmp_path / "tiny.parquet"
    make_corpus(sizes={"220": 400, "210": 200, "241": 10}).to_parquet(path, index=False)
    code = cli.main(["plan", "--parquet", str(path), "--out", str(tmp_path / "s.jsonl")])
    assert code == 2 and not (tmp_path / "s.jsonl").exists()
    assert "STOP" in capsys.readouterr().out


def test_brief_command_prints_a_hash_per_file(tmp_path, parquet, planned, capsys):
    out_dir = tmp_path / "briefs"
    code = cli.main(
        [
            "brief",
            "--slots",
            str(planned),
            "--pilot",
            "--out-dir",
            str(out_dir),
            "--parquet",
            str(parquet),
        ]
    )
    assert code == 0
    text = capsys.readouterr().out
    sha = hashlib.sha256((out_dir / "pilot.md").read_bytes()).hexdigest()
    assert f"pilot.md sha256 {sha}" in text
    code = cli.main(
        ["brief", "--slots", str(planned), "--out-dir", str(out_dir), "--parquet", str(parquet)]
    )
    assert code == 0
    text = capsys.readouterr().out
    assert "220-01.md sha256" in text and "traps-01.md sha256" in text


def test_unknown_command_and_help(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["nonsense"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2


def test_python_dash_m_entry_point_works():
    proc = subprocess.run(
        [sys.executable, "-m", "fedlex_bench.benchmark", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    for name in ("plan", "brief", "assemble", "select", "validate"):
        assert name in proc.stdout


# ------------------------------------------------------------------ the whole chain on the pilot


def test_pilot_chain_plan_to_validate(tmp_path, parquet, planned, capsys):
    slots = read_jsonl(planned)
    pilot = [s for s in slots if s["pilot"]]
    drafts = tmp_path / "drafts"
    drafts.mkdir()
    lines = []
    for s in pilot:
        if s["qtype"] == "trap":
            rec = {
                "slot_id": s["slot_id"],
                "skip": False,
                "skip_reason": None,
                "question": words(s["slot_id"]) + "?",
                "trap_reason": "Not regulated by the eight acts.",
            }
        else:
            labels = ["A", "B"] if s["qtype"] == "cross_ref" else ["A"]
            rec = {
                "slot_id": s["slot_id"],
                "skip": False,
                "skip_reason": None,
                "question": words(s["slot_id"]) + "?",
                "gold_facts": [{"text": f"Fact {x}.", "article": x} for x in labels],
                "difficulty": "medium",
                "competing_candidates": [],
                "note": "",
            }
        lines.append(json.dumps(rec))
    (drafts / "pilot.jsonl").write_bytes(("\n".join(lines) + "\n").encode())
    agent = tmp_path / "agent.md"
    agent.write_text("role", encoding="utf-8")

    cand = tmp_path / "assembled" / "candidates.jsonl"
    code = cli.main(
        [
            "assemble",
            "--slots",
            str(planned),
            "--drafts",
            str(drafts),
            "--out",
            str(cand),
            "--agent-file",
            str(agent),
            "--drafted-on",
            "2026-10-16",
            "--parquet",
            str(parquet),
        ]
    )
    assert code == 0 and len(read_jsonl(cand)) == 10

    critic = tmp_path / "critic"
    critic.mkdir()
    verdicts = [
        {"slot_id": c["slot_id"], "verdict": "keep", "notes": "ok"} for c in read_jsonl(cand)
    ]
    (critic / "pilot.jsonl").write_bytes(
        ("\n".join(json.dumps(v) for v in verdicts) + "\n").encode()
    )

    selected = tmp_path / "selected"
    code = cli.main(
        [
            "select",
            "--candidates",
            str(cand),
            "--critic",
            str(critic),
            "--out-dir",
            str(selected),
            "--reserve-dir",
            str(tmp_path / "reserve"),
        ]
    )
    assert code == 0
    items = read_jsonl(selected / "dev.jsonl") + read_jsonl(selected / "test.jsonl")
    assert len(items) == 10 and items[0]["id"] == "v1-0001"
    assert all(i["review_status"] == "unreviewed" for i in items)
    assert all(
        i["provenance"]["agent_file_sha256"] == hashlib.sha256(b"role").hexdigest() for i in items
    )

    capsys.readouterr()
    code = cli.main(["validate", "--dir", str(selected), "--partial", "--parquet", str(parquet)])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "0 error(s)" in out
    code = cli.main(
        ["validate", "--dir", str(selected), "--partial", "--freeze", "--parquet", str(parquet)]
    )
    assert code == 1 and "freeze_status" in capsys.readouterr().out
