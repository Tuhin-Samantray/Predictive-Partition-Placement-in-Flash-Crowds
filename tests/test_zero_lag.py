import numpy as np

from src.simulator import simulate
from stub_policy import GreedyStub, ScriptedPolicy


def _smooth(n_steps=200, n_part=24):
    t = np.arange(n_steps)[:, None]
    phase = (np.arange(n_part) % 4)[None, :] * (np.pi / 2)
    return 5 + 3 * np.sin(2 * np.pi * t / 80 + phase)


def test_zero_lag_reactive_equals_oracle():
    load = _smooth()
    n_nodes = 4
    node0 = np.arange(load.shape[1]) % n_nodes
    mu = np.full(n_nodes, 1.2 * load.sum(axis=1).mean() / n_nodes)
    kw = dict(m=0, kappa=0.3, budget=4, d=0, k=1)
    r_re = simulate(load, node0, mu, policy=GreedyStub(load, "reactive", max_moves=2), **kw)
    r_or = simulate(load, node0, mu, policy=GreedyStub(load, "oracle", m=0, max_moves=2), **kw)
    assert len(r_re.moves) > 0
    assert r_re.moves == r_or.moves
    np.testing.assert_array_equal(r_re.wait, r_or.wait)


def test_observed_window_boundary_and_read_only():
    load = _smooth(n_steps=20)
    node0 = np.arange(load.shape[1]) % 4
    mu = np.full(4, 100.0)
    pol = ScriptedPolicy({})
    simulate(load, node0, mu, m=1, kappa=0.0, budget=1, d=2, k=3, policy=pol)
    assert [t for t, _n, _r in pol.seen] == list(range(0, 20, 3))
    for t, n, last in pol.seen:
        assert n == max(t - 2 + 1, 0)
        if n:
            np.testing.assert_array_equal(last, load[t - 2])

    class Poke:
        raised = False

        def decide(self, t, observed, state):
            if observed.shape[0] > 0:
                try:
                    observed[0, 0] = -1.0
                except ValueError:
                    Poke.raised = True
            return []

    before = load.copy()
    simulate(load, node0, mu, m=1, kappa=0.0, budget=1, d=0, k=1, policy=Poke())
    assert Poke.raised
    np.testing.assert_array_equal(load, before)