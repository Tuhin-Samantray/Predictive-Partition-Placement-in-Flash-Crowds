import numpy as np

from src.metrics import compute_metrics
from src.simulator import simulate
from stub_policy import GreedyStub, make_case


def _run(load, node0, mu):
    pol = GreedyStub(load, "reactive", max_moves=2)
    res = simulate(load, node0, mu, m=3, kappa=0.3, budget=4, d=1, k=1, policy=pol)
    return res, compute_metrics(res, load, slo=1, wasted_window=10, wait_quantile=0.99)


def test_metamorphic_scaling():
    load, node0, mu = make_case(headroom=1.0)
    scale = 7.0
    r1, m1 = _run(load, node0, mu)
    r2, m2 = _run(load * scale, node0, mu * scale)
    assert m1["moves"] > 0 and m1["moves"] == m2["moves"]
    for key, v in m1.items():
        if key == "migrated_load":
            continue
        if isinstance(v, bool):
            assert v == m2[key], key
            continue
        np.testing.assert_allclose(v, m2[key], rtol=1e-9, atol=1e-12, err_msg=key)
    np.testing.assert_allclose(r1.wait, r2.wait, rtol=1e-9, atol=1e-12)