import numpy as np
from layers_world import PLANS, run, world

from src.layers import CPU, NET, attribution


def test_attribution_shares_sum_to_one():
    res = run(world(5), PLANS, mode="coupled")
    out = attribution(res, 0.5)
    assert out["n"] > 0
    assert abs(out["cpu"] + out["net"] - 1) < 1e-12
    assert 0 <= out["cpu"] <= 1


def test_attribution_without_violations_is_empty_and_binding_is_valid():
    res = run(world(5), PLANS, mode="coupled")
    assert attribution(res, 1e12) == {"n": 0, "cpu": 0, "net": 0}
    assert set(np.unique(res.binding)) <= {CPU, NET}
