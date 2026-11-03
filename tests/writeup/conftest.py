"""The write-up tests reuse the fake project of the results page tests (tests/site)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "site"))
