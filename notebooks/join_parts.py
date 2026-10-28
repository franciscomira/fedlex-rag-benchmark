"""Join checkpoint parts downloaded from Kaggle, verify them, and unpack into ``models/<run>/``.

    uv run python notebooks/join_parts.py models/incoming r1_inbatch

``models/incoming`` holds ``manifest.json`` and the ``*.partNN`` files (and, optionally,
``<run>_log.jsonl`` and ``<run>_config.json``). The joined zip must match the SHA-256 in the
manifest, otherwise nothing is unpacked. Result: ``models/<run>/best/`` (the checkpoint),
``models/<run>/log.jsonl`` and ``models/<run>/config.json``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(incoming: Path, run: str) -> None:
    manifest = json.loads((incoming / "manifest.json").read_text(encoding="utf-8"))
    missing = [p for p in manifest["parts"] if not (incoming / p).exists()]
    if missing:
        raise SystemExit(f"missing parts: {missing}")
    data = b"".join((incoming / p).read_bytes() for p in manifest["parts"])
    digest = hashlib.sha256(data).hexdigest()
    if digest != manifest["sha256"]:
        raise SystemExit(
            f"checksum mismatch: got {digest[:12]}, expected {manifest['sha256'][:12]}"
        )
    out = ROOT / "models" / run
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / manifest["name"]
    zip_path.write_bytes(data)
    best = out / "best"
    if best.exists():
        shutil.rmtree(best)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(best)
    zip_path.unlink()
    for suffix in ("log.jsonl", "config.json"):
        src = incoming / f"{run}_{suffix}"
        if src.exists():
            shutil.copy(src, out / suffix)
    print(f"checksum ok ({digest[:12]}); unpacked to {best.relative_to(ROOT)}:")
    for f in sorted(best.iterdir()):
        print(f"  {f.name}  {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]), sys.argv[2])
