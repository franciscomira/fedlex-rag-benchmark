"""Group bootstrap confidence intervals (docs/eval_spec.md section 7)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

N_BOOT = 10_000
SEED = 20261017


def _group_means(values: np.ndarray, groups: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per-group sum and count, with groups in sorted order."""
    names = sorted(set(groups))
    index = {g: i for i, g in enumerate(names)}
    sums = np.zeros(len(names))
    counts = np.zeros(len(names))
    for v, g in zip(values, groups, strict=True):
        sums[index[g]] += v
        counts[index[g]] += 1
    return sums, counts


def _boot(sums: np.ndarray, counts: np.ndarray, seed: int, n_boot: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    return sums[idx].sum(axis=1) / counts[idx].sum(axis=1)


def _summary(point: float, dist: np.ndarray) -> dict[str, float]:
    lo, hi = np.percentile(dist, [2.5, 97.5])
    return {"mean": float(point), "lo": float(lo), "hi": float(hi)}


def bootstrap_ci(
    values: Sequence[float], groups: Sequence[str], seed: int = SEED, n_boot: int = N_BOOT
) -> dict[str, float]:
    """Mean with a 95% percentile interval, resampling whole groups."""
    arr = np.asarray(values, dtype=float)
    sums, counts = _group_means(arr, groups)
    return _summary(arr.mean(), _boot(sums, counts, seed, n_boot))


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    groups: Sequence[str],
    seed: int = SEED,
    n_boot: int = N_BOOT,
) -> dict[str, float]:
    """Mean of (a - b) over the same items, with a 95% interval, resampling groups."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    sums, counts = _group_means(diff, groups)
    return _summary(diff.mean(), _boot(sums, counts, seed, n_boot))
