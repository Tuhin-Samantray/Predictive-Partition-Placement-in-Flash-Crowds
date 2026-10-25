import numpy as np

from src.simulator import simulate
from stub_policy import ScriptedPolicy


def _const(vals, n_steps=40):
    return np.tile(np.array(vals, dtype=float), (n_steps, 1))


def test_below_capacity_zero_queue():
    load = _const([6.0, 3.0])
    mu = np.array([10.0, 10.0])
    res = simulate(load, np.array([0, 1]), mu, m=1, kappa=0.3, budget=0, d=0, k=1)
    assert res.queue.max() == 0.0
    np.testing.assert_allclose(res.wait, load / (2 * mu))
    np.testing.assert_allclose(res.served, load)


def test_above_capacity_closed_form():
    n_steps = 40
    load = _const([15.0, 3.0], n_steps)
    mu = np.array([10.0, 10.0])
    res = simulate(load, np.array([0, 1]), mu, m=1, kappa=0.3, budget=0, d=0, k=1)
    t = np.arange(n_steps)
    np.testing.assert_allclose(res.queue[:, 0], (t + 1) * 5.0)
    np.testing.assert_allclose(res.queue[-1, 0], n_steps * (15.0 - 10.0))
    np.testing.assert_allclose(res.wait[:, 0], (t * 5.0 + 15.0 / 2) / 10.0)
    np.testing.assert_allclose(res.served[:, 0], 10.0)
    assert res.queue[:, 1].max() == 0.0


def _base():
    return np.ones((20, 4)), np.array([0, 0, 1, 1]), np.array([100.0, 100.0])


def test_move_timing_and_overhead():
    load, node0, mu = _base()
    pol = ScriptedPolicy({3: [(0, 0, 1)]})
    res = simulate(load, node0, mu, m=4, kappa=0.5, budget=1, d=0, k=1, policy=pol)
    assert res.moves == [(3, 0, 0, 1, 7)]
    assert (res.node_hist[:7, 0] == 0).all() and (res.node_hist[7:, 0] == 1).all()
    exp_src = np.full(20, 2.0)
    exp_src[3:7] = 2.5
    exp_src[7:] = 1.0
    exp_dst = np.full(20, 2.0)
    exp_dst[7:] = 3.0
    np.testing.assert_allclose(res.arrivals[:, 0], exp_src)
    np.testing.assert_allclose(res.arrivals[:, 1], exp_dst)


def test_zero_duration_move_is_instant():
    load, node0, mu = _base()
    pol = ScriptedPolicy({3: [(0, 0, 1)]})
    res = simulate(load, node0, mu, m=0, kappa=0.5, budget=1, d=0, k=1, policy=pol)
    assert res.moves == [(3, 0, 0, 1, 3)]
    assert res.node_hist[3, 0] == 1 and res.node_hist[2, 0] == 0
    assert res.arrivals[3, 0] == 1.0 and res.arrivals[3, 1] == 3.0
    np.testing.assert_allclose(res.arrivals, res.fg)


def test_budget_limit_and_unlimited():
    load, node0, mu = _base()
    script = {3: [(0, 0, 1), (1, 0, 1)]}
    res = simulate(load, node0, mu, m=4, kappa=0.0, budget=1, d=0, k=1, policy=ScriptedPolicy(script))
    assert len(res.moves) == 1 and res.dropped_moves == 1
    res = simulate(load, node0, mu, m=4, kappa=0.0, budget=None, d=0, k=1, policy=ScriptedPolicy(script))
    assert len(res.moves) == 2 and res.dropped_moves == 0


def test_cooldown():
    load, node0, mu = _base()
    script = {3: [(0, 0, 1)], 8: [(0, 1, 0)], 12: [(0, 1, 0)]}
    res = simulate(load, node0, mu, m=4, kappa=0.0, budget=1, d=0, k=1,
                   cooldown=5, policy=ScriptedPolicy(script))
    assert res.moves == [(3, 0, 0, 1, 7), (12, 0, 1, 0, 16)]
    assert res.dropped_moves == 1


def test_invalid_moves_dropped():
    load, node0, mu = _base()
    script = {3: [(0, 1, 0), (0, 0, 0), (0, 0, 9)]}
    res = simulate(load, node0, mu, m=2, kappa=0.0, budget=3, d=0, k=1, policy=ScriptedPolicy(script))
    assert res.moves == [] and res.dropped_moves == 3