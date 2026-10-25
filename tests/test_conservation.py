import numpy as np

from src.metrics import compute_metrics
from src.simulator import simulate
from stub_policy import GreedyStub, make_case


def test_conservation():
    load, node0, mu = make_case()
    pol = GreedyStub(load, "reactive", max_moves=2)
    res = simulate(load, node0, mu, m=3, kappa=0.3, budget=4, d=1, k=1, policy=pol)
    assert len(res.moves) > 0
    prev = np.vstack([np.zeros((1, mu.shape[0])), res.queue[:-1]])
    np.testing.assert_allclose(res.arrivals + prev, res.served + res.queue, rtol=1e-12, atol=1e-9)
    assert (res.served <= mu + 1e-9).all()
    assert (res.queue >= 0).all()
    assert (res.arrivals >= res.fg - 1e-12).all()
    np.testing.assert_allclose(res.fg.sum(axis=1), load.sum(axis=1), rtol=1e-12)
    np.testing.assert_allclose(res.arrivals.sum(), res.served.sum() + res.queue[-1].sum(), rtol=1e-12)
    met = compute_metrics(res, load, slo=1, wasted_window=10, wait_quantile=0.99)
    assert met["conservation_ok"]