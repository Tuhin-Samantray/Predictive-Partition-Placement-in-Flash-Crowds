import numpy as np
from layers_world import PLANS, run, same, world


def test_no_moves_makes_coupled_and_control_identical():
    w = world(2)
    same(run(w, None, mode="coupled"), run(w, None, mode="const_m"))
    same(run(w, {}, mode="coupled"), run(w, {}, mode="const_m"))


def test_const_m_control_puts_no_migration_bytes_on_links():
    w = world(2)
    res = run(w, PLANS, mode="const_m")
    assert len(res.moves) > 0
    assert not res.mig_charge.any() and not res.sent.any()
    assert np.array_equal(res.net_in, np.stack([np.bincount(h, weights=b, minlength=4)
                                                for h, b in zip(res.node_hist, w["nbytes"])]))
