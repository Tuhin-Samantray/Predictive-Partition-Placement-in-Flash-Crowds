import numpy as np

from src.metrics import compute_metrics, weighted_quantile
from src.simulator import simulate
from stub_policy import ScriptedPolicy

KW = dict(slo=1, wasted_window=10, wait_quantile=0.99)


def test_damage_closed_form():
    n_steps = 40
    load = np.tile(np.array([15.0, 3.0]), (n_steps, 1))
    mu = np.array([10.0, 10.0])
    res = simulate(load, np.array([0, 1]), mu, m=1, kappa=0.3, budget=0, d=0, k=1)
    met = compute_metrics(res, load, **KW)
    dmg, vio = 0.0, 0.0
    for t in range(n_steps):
        w = (5 * t + 7.5) / 10
        dmg += max(0.0, w - 1) * 15
        vio += 15 if w > 1 else 0
    total = n_steps * 18
    np.testing.assert_allclose(met["damage"], dmg / total)
    np.testing.assert_allclose(met["slo_violation_frac"], vio / total)
    assert met["overload_steps"] == n_steps
    np.testing.assert_allclose(met["peak_backlog_steps"], n_steps * 5 / 10)
    np.testing.assert_allclose(met["imbalance"], 1.5 / ((1.5 + 0.3) / 2))
    assert met["moves"] == 0 and met["migrated_load"] == 0.0
    assert "migrated_bytes" not in met


def test_weighted_quantile():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    ones = np.ones(4)
    assert weighted_quantile(v, ones, 0.25) == 1.0
    assert weighted_quantile(v, ones, 0.5) == 2.0
    assert weighted_quantile(v, ones, 1.0) == 4.0
    assert weighted_quantile(v, np.array([0.0, 0.0, 0.0, 5.0]), 0.5) == 4.0
    assert weighted_quantile(v, np.zeros(4), 0.5) == 0.0


def _move_run(p0_load):
    load = np.ones((30, 4))
    load[:, 0] = p0_load
    node0 = np.array([0, 0, 1, 1])
    mu = np.array([10.0, 10.0])
    pol = ScriptedPolicy({3: [(0, 0, 1)]})
    res = simulate(load, node0, mu, m=2, kappa=0.0, budget=1, d=0, k=1, policy=pol)
    return res, load


def test_wasted_move_classification():
    res, load = _move_run(1.0)
    assert compute_metrics(res, load, **KW)["wasted_moves"] == 1
    res, load = _move_run(12.0)
    assert compute_metrics(res, load, **KW)["wasted_moves"] == 0


def test_migrated_bytes_only_with_sizes():
    res, load = _move_run(1.0)
    met = compute_metrics(res, load, part_size=np.array([5.0, 6.0, 7.0, 8.0]), **KW)
    assert met["migrated_bytes"] == 5.0
    assert met["migrated_load"] == 1.0 and met["moves"] == 1