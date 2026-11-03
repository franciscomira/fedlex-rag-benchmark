"""--check-numbers: hand-typed numbers are flagged, placeholder output and named things are not."""

from __future__ import annotations

import pytest
from writeup_fixtures import make_project

from fedlex_bench.writeup.__main__ import main
from fedlex_bench.writeup.build import ALLOWLIST, TEMPLATE
from fedlex_bench.writeup.numbers import check_numbers, load_allowlist, numbers_in


@pytest.mark.parametrize("line, expected", [
    ("The model has 118 million parameters.", ["118"]),
    ("Recall rose to 0.81 or 81% of 4,073 articles.", ["0.81", "81%", "4,073"]),
    ("Article 257d and 3 epochs.", ["257d", "3"]),
    ("SR 220 is the Code of Obligations.", ["220"]),
    # ignored by rule
    ("Written in 2026 and updated 2026-11-03 or 3.11.2026 or 3 November or November 3.", []),
    ("Phase 4 follows Phase 3; see Section 2, Figure 1 and Table 2.", []),
    ("## 3. Results", []),
    ("### 2.1 Details", []),
    ("1. first item", []),
    ("2) second item", []),
    ("Models T1, T2, P3 and A3 with e5-small, int8 and recall@10, MRR@5.", []),
    ("Judged by gpt-6.1-sol and claude-haiku-5-5.", []),
    ("Run `uv run python 3.11 --seed 20261017` first.", []),
    ("See [the page](https://example.com/page/12) for more.", []),
    ("A year like 1999 is ignored, but 1999 items are not: 12 items.", ["12"]),
])
def test_rules(line, expected):
    assert numbers_in(line) == expected


def test_placeholder_tokens_are_not_numbers():
    from fedlex_bench.writeup.placeholders import CLOSE, OPEN

    assert numbers_in(f"Value {OPEN}12{CLOSE} here and {OPEN}3{CLOSE}%.") == []


def test_fenced_code_is_skipped_and_lines_are_reported():
    lines = ["intro 5", "```", "code 6", "```", "after 7"]
    assert [(o.line, o.token) for o in check_numbers(lines, [])] == [(1, "5"), (5, "7")]


def test_allowlist_entries_and_context(tmp_path):
    path = tmp_path / "allow.txt"
    path.write_text("# comment\n\n220\n15 @ traps\n 81%  # trailing comment\n", encoding="utf-8")
    allow = load_allowlist(path)
    assert allow == [("220", None), ("15", "traps"), ("81%", None)]
    lines = ["SR 220 and 81%.", "15 traps were set.", "15 acts were read.", "4 left."]
    assert [(o.line, o.token) for o in check_numbers(lines, allow)] == [(3, "15"), (4, "4")]


def test_missing_allowlist_allows_nothing(tmp_path):
    assert load_allowlist(tmp_path / "none.txt") == []
    assert load_allowlist(None) == []


def test_cli_check_numbers_prints_offending_lines_and_fails(tmp_path, capsys):
    template = "# T\n\nOne {{v T1 recall@10}} and 77 typed by hand.\n\nAnd 2026 and Phase 4.\n"
    root, _ = make_project(tmp_path, template=template)
    args = ["build", "--root", str(root), "--manifest", str(root / "site/results_manifest.json"),
            "--check-numbers"]
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "post.template.md:3: number '77' is typed by hand" in err
    assert "One {{...}} and 77 typed by hand." in err
    assert ":5:" not in err and "1 number(s) not allowed" in err
    # listing it in the allowlist makes the check pass
    (root / ALLOWLIST).write_text("77\n", encoding="utf-8")
    main(args)
    assert "not allowed" not in capsys.readouterr().err


def test_without_the_option_numbers_are_not_checked(tmp_path):
    root, _ = make_project(tmp_path, template="# T\n\nTyped 77 here.\n")
    main(["build", "--root", str(root), "--manifest", str(root / "site/results_manifest.json")])
    assert (root / TEMPLATE).exists()
