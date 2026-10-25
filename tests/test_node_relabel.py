import numpy as np

from src.metrics import compute_metrics
from src.simulator import simulate
from stub_policy import GreedyStub, make_case


def _run(load, node0, mu):
    pol = GreedyStub(load, "reactive", max_moves=2)
    res = simulate(load, node0, mu, m=3, kappa=0.3, budget=4, d=1, k=1, policy=pol)
    return res, compute_metrics(res, load, slo=1, wasted_window=10, wait_quantile=0.99)


def test_node_relabel():
    load, node0, mu = make_case(headroom=1.0)
    n_nodes = mu.shape[0]
    mu_h = mu * np.array([0.8, 0.9, 1.1, 1.2])
    perm = np.random.default_rng(11).permutation(n_nodes)
    mu_p = np.empty(n_nodes)
    mu_p[perm] = mu_h
    r1, m1 = _run(load, node0, mu_h)
    r2, m2 = _run(load, perm[node0], mu_p)
    assert m1["moves"] > 0
    for key, v in m1.items():
        if isinstance(v, bool):
            assert v == m2[key], key
        else:
            np.testing.assert_allclose(v, m2[key], rtol=1e-9, atol=1e-12, err_msg=key)
    np.testing.assert_allclose(r1.wait, r2.wait[:, perm], rtol=1e-9, atol=1e-12)
    assert [(a, b) for a, b, *_ in r1.moves] == [(a, b) for a, b, *_ in r2.moves]