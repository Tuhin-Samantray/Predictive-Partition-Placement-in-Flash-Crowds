import numpy as np
from layers_world import PLANS, run, world


def test_bytes_in_equal_bytes_out_plus_queued_over_a_whole_run():
    w = world(6)
    res = run(w, PLANS, mode="coupled")
    prev = np.vstack([np.zeros((1, 4)), res.net_queue[:-1]])
    assert np.allclose(res.net_in + prev, res.net_served + res.net_queue)
    assert (res.net_served <= w["link"] + 1e-9).all()
    assert (res.net_queue >= 0).all()


def test_each_migrated_byte_is_charged_on_both_links():
    res = run(world(6), PLANS, mode="coupled")
    assert res.sent.sum() > 0
    assert np.isclose(res.mig_charge.sum(), 2 * res.sent.sum())


def test_bytes_sent_never_exceed_the_bytes_of_the_started_moves():
    w = world(6)
    res = run(w, PLANS, mode="coupled", phi=0.5)
    cap = sum(0.5 * w["sizes"][p] for _t, p, _s, _d, _f in res.moves)
    assert res.sent.sum() <= cap + 1e-9
