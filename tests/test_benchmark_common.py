"""Shared helpers: the split rule, JSONL reading and the hashes that tie files together."""

from __future__ import annotations

import hashlib
import json

import pytest

from fedlex_bench.benchmark import common


def test_bucket_and_split_follow_the_spec_formula():
    for aid in ("220:1", "210:480", "235.1:5"):
        expected = int(hashlib.sha256(("fedlex-bench-v1:" + aid).encode()).hexdigest(), 16) % 100
        assert common.bucket(aid) == expected
        assert common.split_of(aid) == ("dev" if expected < 27 else "test")


def test_read_jsonl_splits_on_newline_only(tmp_path):
    # U+2028 and U+0085 are line breaks for str.splitlines() but legal inside a JSON string
    records = [{"q": "a b", "n": 1}, {"q": "c\u0085d\x0ce", "n": 2}]
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    path = tmp_path / "x.jsonl"
    path.write_bytes(text.encode("utf-8"))
    assert common.read_jsonl(path) == records
    assert common.split_lines(text) == [json.dumps(r, ensure_ascii=False) for r in records]


def test_split_lines_ignores_blank_lines_and_a_trailing_carriage_return():
    assert common.split_lines('{"a": 1}\r\n\n{"b": 2}\n') == ['{"a": 1}\r', '{"b": 2}']
    assert common.read_jsonl_text('{"a": 1}\r\n\n{"b": 2}\n', "t") == [{"a": 1}, {"b": 2}]


def test_read_jsonl_reports_the_line_of_a_bad_record(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_bytes(b'{"a": 1}\n{bad\n')
    with pytest.raises(ValueError, match="x.jsonl:2"):
        common.read_jsonl(path)


def test_slot_hash_covers_gold_ids_and_ignores_the_pilot_flag():
    slot = {"slot_id": "s-0001", "gold_article_ids": ["220:1"], "pilot": False, "act": "220"}
    base = common.slot_hash(slot)
    assert common.slot_hash({**slot, "pilot": True}) == base
    assert common.slot_hash({**slot, "gold_article_ids": ["220:2"]}) != base
    assert common.slot_hash({**slot, "act": "210"}) != base
    assert len(base) == 64


def test_slots_sha256_equals_the_hash_of_the_written_file(tmp_path):
    slots = [{"slot_id": "s-0001", "b": 1}, {"slot_id": "s-0002", "b": 2}]
    path = tmp_path / "s.jsonl"
    sha = common.write_jsonl(path, slots)
    assert common.slots_sha256(slots) == sha == hashlib.sha256(path.read_bytes()).hexdigest()


def test_canonical_sha256_ignores_key_order():
    a = {"question": "q", "gold_facts": [{"text": "t", "article_id": "220:1"}]}
    b = {"gold_facts": [{"article_id": "220:1", "text": "t"}], "question": "q"}
    assert common.canonical_sha256(a) == common.canonical_sha256(b)
    assert common.canonical_sha256(a) != common.canonical_sha256({**a, "question": "q2"})
