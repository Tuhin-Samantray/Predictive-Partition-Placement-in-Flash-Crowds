import numpy as np

from src.layers import (CPU, NET, binding_layer, demand_scale, queue_step, relative_demand)
from src.simulator import simulate


def test_foreground_bytes_are_conserved_across_nodes():
    row = np.array([10.0, 20.0, 30.0, 40.0])
    node_of = np.array([0, 1, 1, 0])
    per_node = np.bincount(node_of, weights=row, minlength=2)
    assert per_node.tolist() == [50.0, 50.0]
    assert per_node.sum() == row.sum()


def test_queue_step_conserves_and_never_serves_above_capacity():
    rng = np.random.default_rng(1)
    q = rng.uniform(0, 5, 6)
    inflow = rng.uniform(0, 20, 6)
    cap = np.full(6, 8.0)
    wait, q_new, served = queue_step(q, inflow, cap)
    assert np.allclose(q + inflow, served + q_new)
    assert (served <= cap + 1e-12).all()
    assert (q_new >= 0).all() and (wait >= 0).all()


def test_queue_step_reproduces_the_l1_simulator():
    rng = np.random.default_rng(3)
    load = rng.uniform(0, 10, (30, 8))
    node0 = np.arange(8) % 4
    mu = np.full(4, 12.0)
    res = simulate(load, node0, mu, m=0, kappa=0.0, budget=0, d=0, k=1)
    q = np.zeros(4)
    for t in range(30):
        inflow = np.bincount(node0, weights=load[t], minlength=4)
        wait, q, served = queue_step(q, inflow, mu)
        assert np.allclose(wait, res.wait[t])
        assert np.allclose(q, res.queue[t])
        assert np.allclose(served, res.served[t])


def test_demand_scale_is_the_load_weighted_mean():
    train = np.array([[1.0, 3.0], [1.0, 3.0], [1.0, 3.0]])
    demand = np.array([2.0, 4.0])
    assert abs(demand_scale(train, demand) - 3.5) < 1e-12
    rel = relative_demand(train, demand)
    weights = train.sum(axis=0)
    assert abs((weights * rel).sum() / weights.sum() - 1) < 1e-12


def test_constant_demand_gives_relative_demand_one():
    train = np.random.default_rng(5).uniform(1, 9, (20, 7))
    assert np.allclose(relative_demand(train, np.full(7, 1.0)), 1)
    assert np.allclose(relative_demand(train, np.full(7, 37.5)), 1)


def test_binding_layer_picks_the_larger_wait():
    cpu = np.array([2.0, 1.0, 3.0])
    net = np.array([1.0, 2.0, 3.0])
    assert binding_layer(cpu, net).tolist() == [CPU, NET, CPU]
