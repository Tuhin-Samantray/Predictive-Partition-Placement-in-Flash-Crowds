import numpy as np

from src.simulator import simulate
from stub_policy import GreedyStub, make_case


def test_placement_invariant():
    load, node0, mu = make_case()
    m, budget = 3, 2
    pol = GreedyStub(load, "reactive", max_moves=2)
    res = simulate(load, node0, mu, m=m, kappa=0.3, budget=budget, d=1, k=1, policy=pol)
    n_steps, n_part = load.shape
    n_nodes = mu.shape[0]
    assert len(res.moves) > 0
    assert res.node_hist.shape == (n_steps, n_part)
    assert res.node_hist.min() >= 0 and res.node_hist.max() < n_nodes
    # every partition is served by exactly one node: foreground over nodes equals total load
    np.testing.assert_allclose(res.fg.sum(axis=1), load.sum(axis=1), rtol=1e-12)
    in_flight = np.zeros(n_steps, dtype=int)
    for t0, p, src, dst, tf in res.moves:
        assert tf == t0 + m
        assert (res.node_hist[t0:min(tf, n_steps), p] == src).all()
        if tf < n_steps:
            assert res.node_hist[tf, p] == dst
        in_flight[t0:min(tf, n_steps)] += 1
    assert in_flight.max() <= budget
    for p in {mv[1] for mv in res.moves}:
        spans = sorted((mv[0], mv[4]) for mv in res.moves if mv[1] == p)
        for (_a0, a1), (b0, _b1) in zip(spans, spans[1:]):
            assert b0 >= a1