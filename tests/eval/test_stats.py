import numpy as np

from fedlex_bench.eval.stats import bootstrap_ci, paired_bootstrap


def test_bootstrap_ci_is_deterministic_and_brackets_mean():
    vals = [1.0] * 30 + [0.0] * 10
    groups = [f"g{i}" for i in range(40)]
    a = bootstrap_ci(vals, groups, seed=1)
    b = bootstrap_ci(vals, groups, seed=1)
    assert a == b
    assert a["lo"] < a["mean"] == 0.75 < a["hi"]


def test_groups_resampled_together():
    # two groups, one all-1, one all-0: every resample mean is in {0, .5, 1}
    vals = [1, 1, 0, 0]
    groups = ["a", "a", "b", "b"]
    ci = bootstrap_ci(vals, groups, seed=3, n_boot=500)
    assert ci["lo"] == 0.0 and ci["hi"] == 1.0


def test_paired_difference_excludes_zero_when_clearly_better():
    n = 60
    a = np.ones(n)
    b = np.zeros(n)
    groups = [f"g{i}" for i in range(n)]
    d = paired_bootstrap(a, b, groups, seed=1)
    assert d["mean"] == 1.0 and d["lo"] > 0


def test_paired_difference_includes_zero_when_equal():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, 50).astype(float)
    groups = [f"g{i}" for i in range(50)]
    d = paired_bootstrap(a, a.copy(), groups, seed=1)
    assert d["lo"] <= 0 <= d["hi"]
