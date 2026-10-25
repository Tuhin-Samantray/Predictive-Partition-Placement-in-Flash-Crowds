import math

import numpy as np
import pytest

from small_world import real_cfg
from src.policies import Ewma, P0Static, P1Threshold, P2Greedy, TUNABLE, build_policy
from src.simulator import simulate
from stub_policy import make_case


def _state(node_of, free=4, n_nodes=2):
    node_of = np.array(node_of)
    return dict(node_of=node_of, moving=np.zeros(node_of.size, bool),
                movable=np.ones(node_of.size, bool), free_slots=free, n_nodes=n_nodes)


def test_p0_never_moves():
    load, node0, mu = make_case()
    kw = dict(m=3, kappa=0.3, budget=4, d=1, k=1)
    r0 = simulate(load, node0, mu, policy=P0Static(), **kw)
    rn = simulate(load, node0, mu, policy=None, **kw)
    assert r0.moves == [] and r0.dropped_moves == 0
    np.testing.assert_array_equal(r0.wait, rn.wait)


def test_ewma_recursion_and_incremental_update():
    rows = np.array([[2.0], [4.0], [8.0]])
    ew = Ewma(0.5)
    ew.update(rows[:2])
    assert ew.value[0] == 3.0
    ew.update(rows)
    assert ew.value[0] == 5.5
    batch = Ewma(0.5)
    batch.update(rows)
    assert batch.value[0] == 5.5
    with pytest.raises(ValueError):
        ew.update(rows[:1])


def test_p1_moves_hottest_fitting_partition():
    pol = P1Threshold(np.array([10.0, 10.0]), ewma_alpha=0.5, theta_hi=0.8)
    plan = pol.decide(5, np.array([[9.0, 1.0, 1.0, 1.0]]), _state([0, 0, 1, 1]))
    assert plan == [(1, 0, 1)]


def test_p1_idle_below_threshold_and_without_slots():
    mu = np.array([10.0, 10.0])
    quiet = P1Threshold(mu, 0.5, 0.8).decide(5, np.array([[1.0, 1.0, 1.0, 1.0]]), _state([0, 0, 1, 1]))
    assert quiet == []
    full = P1Threshold(mu, 0.5, 0.8).decide(5, np.array([[9.0, 1.0, 1.0, 1.0]]), _state([0, 0, 1, 1], free=0))
    assert full == []
    assert P1Threshold(mu, 0.5, 0.8).decide(0, np.zeros((0, 4)), _state([0, 0, 1, 1])) == []


def test_p2_hysteresis_levels():
    obs = np.array([[5.0, 5.0, 1.0, 1.0]])
    n_moves = {}
    for hyst in (0.0, 0.4, 0.6):
        n_moves[hyst] = len(P2Greedy(0.5, hyst).decide(5, obs, _state([0, 0, 1, 1])))
    assert n_moves == {0.0: 2, 0.4: 1, 0.6: 0}


def test_p2_respects_free_slots():
    obs = np.array([[5.0, 5.0, 1.0, 1.0]])
    assert len(P2Greedy(0.5, 0.0).decide(5, obs, _state([0, 0, 1, 1], free=1))) == 1
    assert P2Greedy(0.5, 0.0).decide(5, obs, _state([0, 0, 1, 1], free=0)) == []


def test_build_policy_and_grids():
    cfg = real_cfg()
    for grid in TUNABLE.values():
        for key in grid.values():
            assert key in cfg and len(cfg[key]) > 0
    sizes = [math.prod(len(cfg[k]) for k in g.values()) for g in TUNABLE.values()]
    assert all(s >= cfg["tuning_budget"] for s in sizes)
    mu = np.ones(4)
    assert isinstance(build_policy("P0", {}, mu), P0Static)
    assert isinstance(build_policy("P1", {"ewma_alpha": 0.5, "theta_hi": 0.9, "cooldown": 0}, mu), P1Threshold)
    assert isinstance(build_policy("P2", {"ewma_alpha": 0.5, "hysteresis": 0.1, "cooldown": 0}, mu), P2Greedy)
    with pytest.raises(ValueError):
        build_policy("P9", {}, mu)