import json

import numpy as np

from src.metrics import compute_metrics
from src.simulator import node_capacity, simulate
from stub_policy import GreedyStub, make_case


def _run():
    load, node0, mu = make_case()
    pol = GreedyStub(load, "reactive", max_moves=2)
    res = simulate(load, node0, mu, m=3, kappa=0.3, budget=4, d=1, k=1, policy=pol)
    return res, compute_metrics(res, load, slo=1, wasted_window=10, wait_quantile=0.99)


def test_determinism():
    r1, m1 = _run()
    r2, m2 = _run()
    assert json.dumps(m1, sort_keys=True) == json.dumps(m2, sort_keys=True)
    assert r1.wait.tobytes() == r2.wait.tobytes()
    assert r1.node_hist.tobytes() == r2.node_hist.tobytes()
    assert r1.moves == r2.moves
    train = np.random.default_rng(3).gamma(2.0, 3.0, (50, 8))
    a = node_capacity(train, 1.6, 4, jitter_frac=0.1, seed=5)
    b = node_capacity(train, 1.6, 4, jitter_frac=0.1, seed=5)
    c = node_capacity(train, 1.6, 4, jitter_frac=0.1, seed=6)
    assert a.tobytes() == b.tobytes()
    assert a.tobytes() != c.tobytes()
    flat = node_capacity(train, 1.6, 4)
    np.testing.assert_allclose(flat, 1.6 * train.sum(axis=1).mean() / 4)