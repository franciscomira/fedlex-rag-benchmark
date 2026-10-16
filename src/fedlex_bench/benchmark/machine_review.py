"""Machine review of the selected items: sidecar verdicts, calibration against the human.

Judges write ``benchmark/machine_review/<judge>-<n>.jsonl`` (the judge name is the file name
before the last ``-<n>``), one record per item::

    {"id": "v1-0001", "verdict": "approve" | "reject", "issues": [...], "reason": "...",
     "confidence": 0.9, "content_sha256": "<hash>"}

``content_sha256`` binds a verdict to the text the judge saw: the sha256 of the canonical JSON
(sorted keys, no ASCII escaping) of ``{"question": ..., "gold_facts": ...}``, see
:func:`content_sha256`. A verdict whose hash is missing or differs from the item's current text
does not count for the freeze. ``machine-review-bind`` fills the hash from the judge input files.

Machine review never touches an item's ``review_status``: that stays human-only. The verdicts are
only (a) shown in the review app, (b) compared with the human on the calibration sample and
(c) an optional way into the freeze (``freeze --accept-machine-approved``).

Commands (``python -m fedlex_bench.benchmark <command>``):

- ``calibration-sample``  pick the 30 calibration items (deterministic, :func:`calibration_sample`)
- ``calibration``         compare the human-reviewed sample items with each judge
- ``machine-status``      per judge: approve / reject counts and items without a verdict
- ``machine-review-bind`` one-off: write ``content_sha256`` into the sidecar records

All of them read ``benchmark/selected`` and never write there; only ``machine-review-bind``
writes (the sidecar files, atomically).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from fedlex_bench.benchmark.common import (
    BENCHMARK_DIR,
    SELECTED_DIR,
    SPLITS,
    canonical_sha256,
    dumps,
    read_jsonl,
)
from fedlex_bench.ingest import PROJECT_ROOT

MACHINE_DIR = BENCHMARK_DIR / "machine_review"
AIDS_DIR = PROJECT_ROOT / "data" / "processed" / "review_aids"
SAMPLE_NAME = "calibration_sample.json"
REPORT_NAME = "calibration_report.json"
SAMPLE_SEED = 20261016
SAMPLE_SIZE = 30
SAMPLE_TRAPS = 4
VERDICTS = ("approve", "reject")
SAMPLE_RULE = (
    "random.Random(seed) over the sorted ids; 4 traps and 26 answerable items. Per group: the "
    "number per query language is the language's share of the group (largest remainder, ties in "
    "language order); the number of dev items is the dev share of the group (largest remainder), "
    "spread over the languages by their dev counts, the rest comes from test; inside each "
    "(language, split) the items are drawn uniformly from the sorted ids"
)

_JUDGE_FILE = re.compile(r"^(?P<judge>.+)-(?P<n>\d+)\.jsonl$")

Verdicts = dict[str, dict[str, dict]]  # {item_id: {judge: record}}


# ------------------------------------------------------------------ sidecar files


def judge_of(name: str | Path) -> str | None:
    """The judge of a sidecar file name (``claude-judge-1.jsonl`` -> ``claude-judge``), or
    ``None`` if the name is not ``<judge>-<n>.jsonl``."""
    m = _JUDGE_FILE.match(Path(name).name)
    return m.group("judge") if m else None


def _record_problem(rec: object) -> str | None:
    if not isinstance(rec, dict):
        return "record must be a JSON object"
    if not isinstance(rec.get("id"), str) or not rec["id"]:
        return "id must be a non-empty string"
    if rec.get("verdict") not in VERDICTS:
        return f"verdict {rec.get('verdict')!r} is not one of {', '.join(VERDICTS)}"
    issues = rec.get("issues", [])
    if not isinstance(issues, list) or not all(isinstance(x, str) for x in issues):
        return "issues must be a list of strings"
    if not isinstance(rec.get("reason", ""), str):
        return "reason must be a string"
    if rec.get("content_sha256") is not None and not isinstance(rec["content_sha256"], str):
        return "content_sha256 must be a string"
    return None


def load_verdicts(
    directory: Path, known_ids: Iterable[str] | None = None
) -> tuple[Verdicts, list[str]]:
    """``({item_id: {judge: record}}, errors)`` from every ``<judge>-<n>.jsonl`` in ``directory``
    (sorted file order). Strict: invalid JSON, a malformed record, an id not in ``known_ids``
    (when given) and a second record for the same ``(judge, id)`` are errors; the offending record
    is not loaded (a duplicate keeps the first). A missing folder gives no verdicts, no error."""
    verdicts: Verdicts = {}
    errors: list[str] = []
    if not directory.is_dir():
        return verdicts, errors
    known = None if known_ids is None else set(known_ids)
    for path in sorted(directory.glob("*.jsonl")):
        judge = judge_of(path)
        if judge is None:
            errors.append(f"{path.name}: file name must be <judge>-<n>.jsonl")
            continue
        text = path.read_bytes().decode("utf-8")
        for n, line in enumerate(text.split("\n"), 1):
            if not line.strip():
                continue
            where = f"{path.name}:{n}"
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{where}: invalid JSON ({exc.msg})")
                continue
            problem = _record_problem(rec)
            if problem:
                errors.append(f"{where}: {problem}")
                continue
            item_id = rec["id"]
            if known is not None and item_id not in known:
                errors.append(f"{where}: unknown id {item_id}")
                continue
            if judge in verdicts.get(item_id, {}):
                errors.append(f"{where}: duplicate verdict of {judge} for {item_id}")
                continue
            clean = {
                "id": item_id,
                "verdict": rec["verdict"],
                "issues": list(rec.get("issues", [])),
                "reason": rec.get("reason", ""),
                "confidence": rec.get("confidence"),
                "content_sha256": rec.get("content_sha256"),
            }
            verdicts.setdefault(item_id, {})[judge] = clean
    return verdicts, errors


def judges_of(verdicts: Mapping[str, Mapping[str, dict]]) -> list[str]:
    return sorted({j for by_judge in verdicts.values() for j in by_judge})


def judges_covering(verdicts: Mapping[str, Mapping[str, dict]], ids: Iterable[str]) -> list[str]:
    """Judges with a verdict for every id in ``ids`` (sorted)."""
    ids = list(ids)
    return [j for j in judges_of(verdicts) if all(j in verdicts.get(i, {}) for i in ids)]


def content_sha256(item: Mapping) -> str:
    """Hash of what a judge judges: the canonical JSON of the item's question and gold facts."""
    return canonical_sha256(
        {"question": item.get("question"), "gold_facts": item.get("gold_facts")}
    )


def _approved_by_all(
    item_id: str, judges_required: Iterable[str], verdicts: Mapping[str, Mapping[str, dict]]
) -> bool:
    required = list(judges_required)
    by_judge = verdicts.get(item_id, {})
    return bool(required) and all(by_judge.get(j, {}).get("verdict") == "approve" for j in required)


def machine_approved(
    item_id: str,
    judges_required: Iterable[str],
    verdicts: Mapping[str, Mapping[str, dict]],
    current_sha256: str | None,
) -> bool:
    """True only if every required judge has an ``approve`` verdict for the item and each verdict
    carries ``content_sha256`` equal to ``current_sha256`` (the hash of the item's text now). A
    missing or different hash, or no hash to compare with: False. No required judge: False."""
    required = list(judges_required)
    if not current_sha256 or not _approved_by_all(item_id, required, verdicts):
        return False
    by_judge = verdicts[item_id]
    return all(by_judge[j].get("content_sha256") == current_sha256 for j in required)


# ------------------------------------------------------------------ item helpers


def human_verdict(item: Mapping) -> str | None:
    """accepted / edited -> approve, rejected -> reject, unreviewed -> ``None``."""
    return {"accepted": "approve", "edited": "approve", "rejected": "reject"}.get(
        item.get("review_status")
    )


def is_expert_flagged(item: Mapping) -> bool:
    return bool((item.get("provenance") or {}).get("expert_flag"))


def load_items(directory: Path) -> list[dict]:
    """Items of dev then test (read only)."""
    return [i for split in SPLITS for i in read_jsonl(directory / f"{split}.jsonl")]


# ------------------------------------------------------------------ the freeze gate


@dataclass
class MachineAcceptance:
    """The machine route into a freeze. ``judges`` are the judges that cover every item (all of
    them must approve); ``ids`` the items that are not human-rejected, not expert-flagged and
    approved by every such judge (this includes human-reviewed items: the freeze counts those as
    human). ``unbound`` (sorted) are the not-human-reviewed items that every judge approved but
    whose verdict has no ``content_sha256`` or one that differs from the item's current text;
    they are also reported in ``errors``. ``errors`` is non-empty when the route cannot be used
    at all."""

    judges: list[str] = field(default_factory=list)
    ids: set[str] = field(default_factory=set)
    errors: list[str] = field(default_factory=list)
    unbound: list[str] = field(default_factory=list)


def machine_acceptance(files: Mapping[str, list[dict]], directory: Path) -> MachineAcceptance:
    items = [i for split in SPLITS for i in files.get(split, [])]
    ids = [i["id"] for i in items]
    verdicts, errors = load_verdicts(directory, known_ids=ids)
    acc = MachineAcceptance(errors=list(errors))
    acc.judges = judges_covering(verdicts, ids)
    if not acc.judges:
        found = judges_of(verdicts)
        acc.errors.append(
            f"no judge in {directory} covers all {len(ids)} items"
            + (f" (found: {', '.join(found)})" if found else " (no verdict files found)")
        )
        return acc
    candidates = [
        i for i in items if i.get("review_status") != "rejected" and not is_expert_flagged(i)
    ]
    acc.ids = {
        i["id"]
        for i in candidates
        if machine_approved(i["id"], acc.judges, verdicts, content_sha256(i))
    }
    acc.unbound = sorted(
        i["id"]
        for i in candidates
        if i["id"] not in acc.ids
        and i.get("review_status") not in HUMAN_STATUSES
        and _approved_by_all(i["id"], acc.judges, verdicts)
    )
    if acc.unbound:
        acc.errors.append(
            f"{len(acc.unbound)} machine verdict(s) are not bound to the current item text "
            f"(content_sha256 missing or different; the item changed after judging or "
            f"machine-review-bind has not run): {', '.join(acc.unbound)}"
        )
    return acc


HUMAN_STATUSES = ("accepted", "edited")


def review_coverage(files: Mapping[str, list[dict]], acceptance: MachineAcceptance) -> dict:
    """Who vouches for the frozen items. ``human_reviewed``: accepted or edited by the human (an
    item that a judge also approved counts here, not as machine-only). ``machine_approved_only``:
    not human-reviewed, approved by every judge in ``acceptance``. Sorted, no timestamps."""
    by_split: dict[str, dict] = {}
    machine_only: dict[str, list[str]] = {}
    for split in SPLITS:
        items = files.get(split, [])
        human = [i for i in items if i["review_status"] in HUMAN_STATUSES]
        only = sorted(
            i["id"]
            for i in items
            if i["review_status"] not in HUMAN_STATUSES and i["id"] in acceptance.ids
        )
        machine_only[split] = only
        by_split[split] = {
            "human_reviewed": len(human),
            "machine_approved_only": len(only),
            "total": len(human) + len(only),
        }
    h = sum(v["human_reviewed"] for v in by_split.values())
    m = sum(v["machine_approved_only"] for v in by_split.values())
    return {
        "accept_machine_approved": True,
        "by_split": by_split,
        "human_reviewed": h,
        "judges": list(acceptance.judges),
        "machine_approved_only": m,
        "machine_only_ids": machine_only,
        "total": h + m,
    }


def coverage_line(coverage: Mapping) -> str:
    return (
        f"review coverage: human-reviewed {coverage['human_reviewed']}, "
        f"machine-approved only {coverage['machine_approved_only']} "
        f"(judges: {', '.join(coverage['judges']) or 'none'})"
    )


# ------------------------------------------------------------------ calibration sample


def apportion(total: int, sizes: Mapping[str, int]) -> dict[str, int]:
    """Split ``total`` over the keys in proportion to ``sizes`` (largest remainder; a tie goes to
    the key that comes first in sorted order). A key never gets more than its size."""
    keys = sorted(sizes)
    pool = sum(sizes.values())
    if pool == 0 or total <= 0:
        return {k: 0 for k in keys}
    total = min(total, pool)
    exact = {k: total * sizes[k] / pool for k in keys}
    out = {k: int(exact[k]) for k in keys}
    spare = total - sum(out.values())
    by_remainder = sorted(keys, key=lambda k: (-(exact[k] - out[k]), k))
    for k in by_remainder:
        if spare == 0:
            break
        if out[k] < sizes[k]:
            out[k] += 1
            spare -= 1
    return out


def calibration_sample(items: Iterable[Mapping], seed: int = SAMPLE_SEED) -> list[str]:
    """The sorted ids of the calibration sample (see :data:`SAMPLE_RULE`). Independent of the
    order of ``items`` and of their review state."""
    rng = random.Random(seed)
    chosen: list[str] = []
    for is_trap, size in ((True, SAMPLE_TRAPS), (False, SAMPLE_SIZE - SAMPLE_TRAPS)):
        cells: dict[str, dict[str, list[str]]] = {}  # lang -> split -> sorted ids
        for it in items:
            if bool(it["is_trap"]) == is_trap:
                cells.setdefault(it["query_lang"], {}).setdefault(it["split"], []).append(it["id"])
        for by_split in cells.values():
            for ids in by_split.values():
                ids.sort()
        n_split = {
            sp: sum(len(by_split.get(sp, [])) for by_split in cells.values()) for sp in SPLITS
        }
        quota = apportion(size, {lang: sum(map(len, c.values())) for lang, c in cells.items()})
        dev_total = apportion(size, n_split)["dev"]
        dev_quota = apportion(dev_total, {lang: len(c.get("dev", [])) for lang, c in cells.items()})
        for lang in sorted(cells):
            dev_ids, test_ids = cells[lang].get("dev", []), cells[lang].get("test", [])
            n_dev = min(dev_quota[lang], quota[lang], len(dev_ids))
            n_test = min(quota[lang] - n_dev, len(test_ids))
            n_dev = min(quota[lang] - n_test, len(dev_ids))  # a short test pool is made up by dev
            chosen += rng.sample(dev_ids, n_dev) + rng.sample(test_ids, n_test)
    return sorted(chosen)


def sample_document(ids: list[str], seed: int = SAMPLE_SEED) -> dict:
    return {"ids": sorted(ids), "n": len(ids), "rule": SAMPLE_RULE, "seed": seed}


def load_sample(directory: Path) -> list[str]:
    """The sample ids (sorted), or ``[]`` when there is no sample file."""
    path = directory / SAMPLE_NAME
    if not path.is_file():
        return []
    return sorted(json.loads(path.read_bytes().decode("utf-8"))["ids"])


# ------------------------------------------------------------------ reports


def calibration_report(
    items: Iterable[Mapping], verdicts: Mapping[str, Mapping[str, dict]], sample: Iterable[str]
) -> dict:
    """Human vs judge on the sample items the human has reviewed (accepted/edited = approve,
    rejected = reject). ``agreement`` is ``None`` when a judge was compared on nothing."""
    by_id = {i["id"]: i for i in items}
    sample = sorted(set(sample))
    present = [i for i in sample if i in by_id]
    done = [i for i in present if human_verdict(by_id[i]) is not None]
    judges: dict[str, dict] = {}
    for judge in judges_of(verdicts):
        confusion = {
            "human_approve": {"judge_approve": 0, "judge_reject": 0},
            "human_reject": {"judge_approve": 0, "judge_reject": 0},
        }
        disagreements, no_verdict = [], []
        for item_id in done:
            rec = verdicts.get(item_id, {}).get(judge)
            if rec is None:
                no_verdict.append(item_id)
                continue
            human = human_verdict(by_id[item_id])
            confusion[f"human_{human}"][f"judge_{rec['verdict']}"] += 1
            if rec["verdict"] != human:
                disagreements.append(item_id)
        compared = len(done) - len(no_verdict)
        judges[judge] = {
            "n_compared": compared,
            "agreement": (compared - len(disagreements)) / compared if compared else None,
            "confusion": confusion,
            "disagreements": disagreements,
            "no_verdict": no_verdict,
        }
    return {
        "sample_size": len(sample),
        "n_reviewed": len(done),
        "missing_from_selection": [i for i in sample if i not in by_id],
        "judges": judges,
    }


def machine_status(items: Iterable[Mapping], verdicts: Mapping[str, Mapping[str, dict]]) -> dict:
    ids = [i["id"] for i in items]
    judges: dict[str, dict] = {}
    for judge in judges_of(verdicts):
        counts = {v: 0 for v in VERDICTS}
        missing = []
        for item_id in ids:
            rec = verdicts.get(item_id, {}).get(judge)
            if rec is None:
                missing.append(item_id)
            else:
                counts[rec["verdict"]] += 1
        judges[judge] = {**counts, "without_verdict": len(missing), "missing_ids": missing}
    return {"n_items": len(ids), "judges": judges}


# ------------------------------------------------------------------ binding (one-off)

_JUDGE_INPUT = re.compile(r"^judge-in-(?:(?P<tag>[a-z0-9]+)-)?(?P<n>\d+)\.jsonl$")
_TAG_ORDER = ("r2", "r3", "bf", "tr")  # the later rounds, in the order they were judged


def judge_input_files(directory: Path) -> list[Path]:
    """The ``judge-in-*.jsonl`` files, oldest round first: ``judge-in-<n>`` (numeric), then the
    tagged rounds ``r2``, ``r3``, ``bf``, ``tr`` (an unknown tag after those, by name), each by
    ``<n>``. A later file overrides an earlier one for the same id."""
    keyed = []
    for path in directory.glob("judge-in-*.jsonl") if directory.is_dir() else []:
        m = _JUDGE_INPUT.match(path.name)
        if not m:
            continue
        tag = m.group("tag")
        if tag is None:
            rank: tuple = (0, "")
        elif tag in _TAG_ORDER:
            rank = (1 + _TAG_ORDER.index(tag), "")
        else:
            rank = (1 + len(_TAG_ORDER), tag)
        keyed.append(((*rank, int(m.group("n"))), path))
    return [p for _, p in sorted(keyed)]


def judged_hashes(directory: Path) -> dict[str, tuple[str, str]]:
    """``{item_id: (content_sha256, file name)}`` of the text the judges were shown (the latest
    judge input file wins). Gold facts are normalised to ``{"text", "article_id"}`` as items
    store them; a trap's input has no ``gold_facts`` and counts as ``[]`` (as stored)."""
    out: dict[str, tuple[str, str]] = {}
    for path in judge_input_files(directory):
        for row in read_jsonl(path):
            facts = [
                {"text": f["text"], "article_id": f["article_id"]}
                for f in row.get("gold_facts") or []
            ]
            out[row["id"]] = (
                content_sha256({"question": row["question"], "gold_facts": facts}),
                path.name,
            )
    return out


@dataclass
class BindReport:
    bound: list[str] = field(default_factory=list)  # ids that now carry the right hash
    changed: dict[str, str] = field(default_factory=dict)  # id -> judge input file (text differs)
    no_input: list[str] = field(default_factory=list)  # ids no judge input file shows
    files_written: list[str] = field(default_factory=list)

    @property
    def unbound(self) -> list[str]:
        return sorted([*self.changed, *self.no_input])


def _replace_atomic(path: Path, data: bytes) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def bind_sidecars(
    items: Iterable[Mapping], machine_dir: Path, aids_dir: Path, dry_run: bool = False
) -> BindReport:
    """Write ``content_sha256`` into every sidecar record whose judged text (judge input files)
    equals the item's current text. A record that was judged on other text, or that no judge
    input shows, is reported and left without a hash (so it fails the freeze). A hash that is
    already right is kept; a record is only rewritten when it gains a hash."""
    current = {i["id"]: content_sha256(i) for i in items}
    judged = judged_hashes(aids_dir)
    report = BindReport()
    for path in sorted(machine_dir.glob("*.jsonl")) if machine_dir.is_dir() else []:
        if judge_of(path) is None:
            continue
        raw = path.read_bytes()
        lines, dirty = [], False
        for line in raw.decode("utf-8").split("\n"):
            if not line.strip():
                lines.append(line)
                continue
            rec = json.loads(line)
            item_id = rec.get("id")
            if item_id not in current:
                lines.append(line)
                continue
            if item_id not in judged:
                report.no_input.append(item_id)
            elif judged[item_id][0] != current[item_id]:
                report.changed[item_id] = judged[item_id][1]
            else:
                report.bound.append(item_id)
                if rec.get("content_sha256") != current[item_id]:
                    rec["content_sha256"] = current[item_id]
                    line = json.dumps(rec)
                    dirty = True
            lines.append(line)
        if dirty:
            report.files_written.append(path.name)
            if not dry_run:
                _replace_atomic(path, "\n".join(lines).encode("utf-8"))
    report.bound.sort()
    report.no_input.sort()
    return report


# ------------------------------------------------------------------ commands


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def _parser(prog: str, doc_line: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=f"benchmark {prog}", description=doc_line)
    ap.add_argument("--dir", type=Path, default=SELECTED_DIR, help="selected items (read only)")
    ap.add_argument("--machine-dir", type=Path, default=MACHINE_DIR, help="machine review folder")
    return ap


def run_sample(argv: list[str] | None = None) -> int:
    ap = _parser("calibration-sample", "write the deterministic calibration sample")
    ap.add_argument("--force", action="store_true", help="overwrite a different existing sample")
    args = ap.parse_args(argv)
    ids = calibration_sample(load_items(args.dir))
    text = dumps(sample_document(ids)) + "\n"
    path = args.machine_dir / SAMPLE_NAME
    if path.is_file() and path.read_bytes().decode("utf-8") != text and not args.force:
        print(f"REFUSED: {path} holds a different sample (use --force to overwrite)")
        return 1
    _write_text(path, text)
    items = {i["id"]: i for i in load_items(args.dir)}
    traps = sum(items[i]["is_trap"] for i in ids)
    dev = sum(items[i]["split"] == "dev" for i in ids)
    print(
        f"calibration sample: {len(ids)} items ({traps} traps, {len(ids) - traps} answerable; "
        f"dev {dev}, test {len(ids) - dev}) -> {path}"
    )
    return 0


def run_calibration(argv: list[str] | None = None) -> int:
    ap = _parser("calibration", "compare human reviews with each judge on the sample")
    args = ap.parse_args(argv)
    sample = load_sample(args.machine_dir)
    if not sample:
        print(f"no sample in {args.machine_dir}: run calibration-sample first")
        return 1
    items = load_items(args.dir)
    verdicts, errors = load_verdicts(args.machine_dir, known_ids=[i["id"] for i in items])
    for e in errors:
        print(f"ERROR {e}")
    report = calibration_report(items, verdicts, sample)
    _write_text(args.machine_dir / REPORT_NAME, dumps(report) + "\n")
    print(f"calibration: reviewed {report['n_reviewed']} of {report['sample_size']} sample items")
    for judge, row in report["judges"].items():
        agree = "n/a" if row["agreement"] is None else f"{row['agreement']:.3f}"
        c = row["confusion"]
        print(
            f"  {judge}: compared {row['n_compared']}, agreement {agree}; human approve -> "
            f"judge approve {c['human_approve']['judge_approve']} / reject "
            f"{c['human_approve']['judge_reject']}; human reject -> judge approve "
            f"{c['human_reject']['judge_approve']} / reject {c['human_reject']['judge_reject']}"
        )
        if row["disagreements"]:
            print(f"    disagreements: {', '.join(row['disagreements'])}")
        if row["no_verdict"]:
            print(f"    reviewed but no verdict: {', '.join(row['no_verdict'])}")
    for i in report["missing_from_selection"]:
        print(f"WARNING sample id {i} is not in the selection")
    return 1 if errors else 0


def run_bind(argv: list[str] | None = None) -> int:
    ap = _parser(
        "machine-review-bind", "one-off: write content_sha256 into the machine review sidecars"
    )
    ap.add_argument(
        "--aids-dir", type=Path, default=AIDS_DIR, help="folder with the judge-in-*.jsonl files"
    )
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args(argv)
    items = load_items(args.dir)
    report = bind_sidecars(items, args.machine_dir, args.aids_dir, dry_run=args.dry_run)
    for item_id, name in sorted(report.changed.items()):
        print(f"UNBOUND {item_id}: the text the judge saw ({name}) differs from the item now")
    for item_id in report.no_input:
        print(f"UNBOUND {item_id}: no judge input file shows this item")
    verb = "would write" if args.dry_run else "wrote"
    print(
        f"bound {len(report.bound)} verdict(s), unbound {len(report.unbound)}; "
        f"{verb} {len(report.files_written)} sidecar file(s)"
    )
    return 0


def run_status(argv: list[str] | None = None) -> int:
    ap = _parser("machine-status", "per judge: verdict counts and items without a verdict")
    args = ap.parse_args(argv)
    items = load_items(args.dir)
    verdicts, errors = load_verdicts(args.machine_dir, known_ids=[i["id"] for i in items])
    for e in errors:
        print(f"ERROR {e}")
    status = machine_status(items, verdicts)
    print(f"machine review: {status['n_items']} items, {len(status['judges'])} judge(s)")
    for judge, row in status["judges"].items():
        print(
            f"  {judge}: approve {row['approve']}, reject {row['reject']}, "
            f"without verdict {row['without_verdict']}"
        )
    return 1 if errors else 0
