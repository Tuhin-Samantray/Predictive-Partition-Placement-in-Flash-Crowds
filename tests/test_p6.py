from pathlib import Path

import numpy as np
import pytest

from src.config import load_config
from src.layers import simulate_layers
from src.policies import LAYER_TUNABLE, P6CrossLayer, build_policy, bytes_per_load, cross_plan, greedy_plan
from src.simulator import simulate

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = [1, 2, 5, 10]


def two_nodes():
    node_of = np.array([0, 0, 1, 1])
    cpu_w = np.ones(4)
    byt_w = np.array([8.0, 1.0, 0.0, 1.0])
    return node_of, cpu_w, byt_w, np.full(2, 10.0), np.ones(4, dtype=bool)


def test_without_bytes_and_with_uniform_capacity_cross_plan_equals_the_p3_repack():
    for seed in range(20):
        rng = np.random.default_rng(seed)
        n_part, n_nodes = 12, 4
        pred = rng.uniform(0.5, 5, n_part)
        node_of = rng.integers(0, n_nodes, n_part)
        movable = rng.random(n_part) < 0.8
        nl = np.bincount(node_of, weights=pred, minlength=n_nodes)
        want = greedy_plan(pred, nl, node_of, movable, 5, 0.1 * nl.mean())
        got = cross_plan(pred, np.zeros(n_part), np.zeros(n_part), node_of, movable, 5,
                         np.ones(n_nodes), 1.0, 0.1, 0.0)
        assert got == want, seed


def test_bytes_can_trigger_a_move_the_cpu_only_repack_does_not_see():
    node_of, cpu_w, byt_w, mu, movable = two_nodes()
    nl = np.bincount(node_of, weights=cpu_w, minlength=2)
    assert greedy_plan(cpu_w, nl, node_of, movable, 2, 0.0) == []
    assert cross_plan(cpu_w, byt_w, np.zeros(4), node_of, movable, 2, mu, 10.0, 0.0, 0.0) == [(1, 0, 1)]


def test_psi_delays_a_move_when_the_free_link_share_is_too_low():
    node_of, cpu_w, byt_w, mu, movable = two_nodes()
    fg_last = np.array([9.5, 0.0, 0.0, 0.0])
    for psi, n_moves in ((0.04, 1), (0.1, 0), (0.0, 1)):
        plan = cross_plan(cpu_w, byt_w, fg_last, node_of, movable, 2, mu, 10.0, 0.0, psi)
        assert len(plan) == n_moves, psi


def test_bytes_per_load_uses_the_window_and_the_mean_ratio_for_idle_partitions():
    observed = np.array([[1.0, 2.0, 0.0], [1.0, 2.0, 0.0], [2.0, 1.0, 0.0]])
    nbytes = np.array([[0.0, 0.0, 0.0], [6.0, 3.0, 5.0], [3.0, 9.0, 7.0]])
    assert np.allclose(bytes_per_load(observed, nbytes, 2), [3.0, 4.0, 3.5])
    assert (bytes_per_load(np.zeros((3, 3)), nbytes, 2) == 0).all()
    with pytest.raises(ValueError):
        bytes_per_load(observed, nbytes[:2], 2)


def test_p6_grid_equals_the_tuning_budget_and_the_policy_is_built_by_name():
    cfg = load_config(str(ROOT / "configs" / "default.yaml"))
    grid = LAYER_TUNABLE["P6"]
    assert int(np.prod([len(cfg[k]) for k in grid.values()])) == cfg["tuning_budget"]
    ctx = {"m": 3, "d": 1, "horizons": HORIZONS, "residual_window": 8}
    pol = build_policy("P6", {"hysteresis": 0.1, "cooldown": 5, "psi": 0.2}, np.ones(4), ctx)
    assert isinstance(pol, P6CrossLayer)


def test_p6_refuses_to_run_without_the_cross_layer_state():
    rng = np.random.default_rng(0)
    load = rng.uniform(0.5, 1.5, (20, 6))
    fc = np.repeat(load[:, None, :], len(HORIZONS), axis=1)
    pol = P6CrossLayer(0.0, 0.0, 2, 1, HORIZONS, 8)
    with pytest.raises(ValueError, match="cross-layer"):
        simulate(load, np.arange(6) % 3, np.full(3, 30.0), m=2, kappa=0.3, budget=4, d=1, k=1,
                 policy=pol, cooldown=0, forecast=fc)


def test_p6_decisions_do_not_depend_on_data_after_the_visible_prefix():
    rng = np.random.default_rng(5)
    n_steps, n_part, n_nodes, cut, d = 60, 8, 4, 30, 1
    load = rng.uniform(0.5, 1.5, (n_steps, n_part))
    load[:, 0] = 20.0
    nbytes = load * 10.0
    fc = np.repeat(load[:, None, :], len(HORIZONS), axis=1) + rng.normal(0, 0.1, (n_steps, len(HORIZONS), n_part))
    load2, nb2, fc2 = load.copy(), nbytes.copy(), fc.copy()
    load2[cut + 1:] = rng.uniform(0, 100, load2[cut + 1:].shape)
    nb2[cut + 1:] = rng.uniform(0, 1000, nb2[cut + 1:].shape)
    fc2[cut + 1:] = rng.uniform(0, 100, fc2[cut + 1:].shape)
    mu = np.full(n_nodes, 30.0)
    link = 2 * nbytes.sum(axis=1).mean() / n_nodes
    node0 = np.arange(n_part) % n_nodes
    sizes, rel = np.full(n_part, 5.0), np.ones(n_part)
    started = []
    for lo, nb, fo in ((load, nbytes, fc), (load2, nb2, fc2)):
        pol = P6CrossLayer(0.0, 0.0, 2, d, HORIZONS, 8)
        res = simulate_layers(lo, nb, rel, node0, mu, link, sizes, m=2, kappa=0.3, budget=4, d=d, k=1,
                              rho=0.25, phi=1.0, mode="coupled", policy=pol, cooldown=0, forecast=fo)
        started.append([mv[:4] for mv in res.moves if mv[0] <= cut])
    assert len(started[0]) > 0 and started[0] == started[1]
