import numpy as np
from layers_world import N_PART, run, same, world


def test_phi_zero_gives_the_one_step_floor():
    w = world(3)
    plans = {2: [(0, 0, 1)], 5: [(1, 1, 2)], 12: [(0, 1, 0)]}
    coupled = run(w, plans, mode="coupled", phi=0)
    control = run(w, plans, mode="const_m", m=1)
    assert len(coupled.moves) > 0
    assert all(tf == t0 + 1 for t0, _p, _s, _d, tf in coupled.moves)
    same(coupled, control)


def test_phi_one_reproduces_the_full_partition_transfer():
    w = world(3)
    w = dict(w, nbytes=np.zeros_like(w["nbytes"]), link=100.0, sizes=np.full(N_PART, 5 * 0.25 * 100.0))
    plans = {3: [(0, 0, 1)]}
    coupled = run(w, plans, mode="coupled", phi=1.0, rho=0.25)
    control = run(w, plans, mode="const_m", m=5)
    assert coupled.moves == control.moves == [(3, 0, 0, 1, 8)]
    assert np.array_equal(coupled.node_hist, control.node_hist)
    assert coupled.sent.sum() == 5 * 0.25 * 100.0
