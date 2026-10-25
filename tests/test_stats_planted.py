import numpy as np
import pandas as pd

from src import stats_tools as st
from src.seeds import derive_seed

LEVEL = 0.95


def seed(*labels):
    return derive_seed("test_stats_planted", *labels)


def test_event_means_average_repeated_events_and_keep_single_ones():
    frame = pd.DataFrame({"dataset": ["D3", "D3", "D3", "D1"], "event_id": ["e1", "e1", "e2", "e9"],
                          "damage": [2.0, 4.0, 5.0, 7.0]})
    out = st.event_means(frame, ["dataset", "event_id"], "damage")
    assert out.loc[("D3", "e1")] == 3.0 and out.loc[("D3", "e2")] == 5.0 and out.loc[("D1", "e9")] == 7.0 and len(out) == 3


def test_bca_interval_is_deterministic_and_covers_the_planted_mean():
    rng = np.random.default_rng(seed("sample"))
    x = rng.normal(5.0, 1.0, 200)
    a = st.bca_interval(x, st.mean_stat, 999, LEVEL, seed("a"))
    b = st.bca_interval(x, st.mean_stat, 999, LEVEL, seed("a"))
    assert a == b and a[0] < 5.0 < a[1] and a[1] - a[0] < 0.6


def test_bca_interval_coverage_of_a_planted_mean_is_near_nominal():
    hits = 0
    runs = 100
    for i in range(runs):
        x = np.random.default_rng(seed("cover", i)).normal(5.0, 1.0, 40)
        lo, hi = st.bca_interval(x, st.mean_stat, 499, LEVEL, seed("cover_boot", i))
        hits += lo <= 5.0 <= hi
    assert 0.85 <= hits / runs <= 1.0


def test_bca_interval_degenerate_cases():
    assert st.bca_interval(np.full(10, 3.0), st.mean_stat, 99, LEVEL, 1) == (3.0, 3.0)
    lo, hi = st.bca_interval(np.array([1.0]), st.mean_stat, 99, LEVEL, 1)
    assert np.isnan(lo) and np.isnan(hi)


def test_bca_median_interval_brackets_a_planted_median():
    x = np.random.default_rng(seed("median")).exponential(2.0, 150)
    lo, hi = st.bca_interval(x, st.median_stat, 999, LEVEL, seed("m"))
    assert lo < np.median(x) < hi


def test_bca_paired_spearman_recovers_a_planted_correlation():
    rng = np.random.default_rng(seed("spear"))
    x = rng.normal(0, 1, 60)
    y = x + rng.normal(0, 0.5, 60)
    lo, hi = st.bca_paired(x, y, lambda a, b: st.spearman(a, b), 499, LEVEL, seed("sp"))
    assert 0.5 < lo < st.spearman(x, y) < hi <= 1.0


def test_spearman_and_kendall_known_values():
    x = np.arange(10.0)
    assert abs(st.spearman(x, x ** 3) - 1) < 1e-12 and abs(st.spearman(x, -x) + 1) < 1e-12
    assert abs(st.kendall_tau([1, 2, 3, 4], [1, 2, 3, 4]) - 1) < 1e-12 and abs(st.kendall_tau([1, 2, 3, 4], [4, 3, 2, 1]) + 1) < 1e-12


def test_paired_wilcoxon_planted_shift_zeros_and_direction():
    d = np.random.default_rng(seed("w")).normal(1.0, 0.5, 40)
    _, p_greater = st.paired_wilcoxon(d, "greater")
    _, p_less = st.paired_wilcoxon(d, "less")
    assert p_greater < 1e-6 and p_less > 0.99
    assert st.paired_wilcoxon(np.zeros(12), "greater")[1] == 1
    with_zeros = np.concatenate([d, np.zeros(20)])
    assert st.paired_wilcoxon(with_zeros, "greater") == st.paired_wilcoxon(d, "greater")


def test_mann_whitney_planted_separation():
    a = np.arange(20.0) + 100
    b = np.arange(20.0)
    u, p = st.mann_whitney(a, b, "greater")
    assert u == 400.0 and p < 1e-6
    assert st.mann_whitney(a, b, "less")[1] > 0.99


def test_cliffs_delta_known_values():
    assert abs(st.cliffs_delta([3, 4, 5], [1, 2, 4]) - 6 / 9) < 1e-12
    assert st.cliffs_delta([5, 6, 7], [1, 2, 3]) == 1.0 and st.cliffs_delta([1, 2, 3], [5, 6, 7]) == -1.0
    assert st.cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0


def test_holm_known_values_and_order():
    adj = st.holm_adjust([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(adj, [0.03, 0.06, 0.06, 0.02])
    assert np.allclose(st.holm_adjust([0.5, 0.9]), [1.0, 1.0])
    assert st.holm_adjust([0.2]).tolist() == [0.2]


def test_recovery_ratio_planted_values_and_the_minimum_tax_rule():
    reactive = np.array([10.0, 10.0, 10.0, 0.0, 10.0])
    oracle = np.array([4.0, 9.8, 4.0, 0.0, 4.0])
    x = np.array([7.0, 9.0, 12.0, 0.0, 4.0])
    tax, r, defined = st.recovery_ratio(reactive, x, oracle, 0.05)
    assert np.allclose(tax, [6.0, 0.2, 6.0, 0.0, 6.0])
    assert defined.tolist() == [True, False, True, False, True]
    assert abs(r[0] - 0.5) < 1e-12 and abs(r[2] + 1 / 3) < 1e-12 and abs(r[4] - 1) < 1e-12
    assert np.isnan(r[1]) and np.isnan(r[3])


def test_support_label_truth_table():
    assert st.support_label(True, False, 30, 30) == st.SUPPORTED
    assert st.support_label(False, True, 30, 30) == st.NOT_SUPPORTED
    assert st.support_label(False, False, 30, 30) == st.INCONCLUSIVE
    assert st.support_label(True, False, 29, 30) == st.INCONCLUSIVE
