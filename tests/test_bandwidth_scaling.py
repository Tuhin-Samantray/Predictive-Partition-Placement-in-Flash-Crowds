import numpy as np
from layers_world import PLANS, run, world


def test_scaling_all_bytes_and_bandwidth_leaves_normalised_results_unchanged():
    w = world(4)
    c = 7.3
    w2 = dict(w, nbytes=w["nbytes"] * c, link=w["link"] * c, sizes=w["sizes"] * c)
    a = run(w, PLANS, mode="coupled")
    b = run(w2, PLANS, mode="coupled")
    assert len(a.moves) > 0 and a.moves == b.moves
    assert np.array_equal(a.node_hist, b.node_hist)
    assert np.allclose(a.wait, b.wait, rtol=1e-9, atol=1e-12)
    assert np.allclose(a.net_queue / w["link"], b.net_queue / w2["link"], rtol=1e-9, atol=1e-12)
    assert np.allclose(a.net_served / w["link"], b.net_served / w2["link"], rtol=1e-9, atol=1e-12)
