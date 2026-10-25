import numpy as np
from layers_world import DEFAULTS, world

from src.layers import simulate_layers


class Spy:
    def __init__(self):
        self.rows = []

    def decide(self, t, observed, state):
        self.rows.append((t, observed.shape[0], state["nbytes"].shape[0], state["nbytes"].flags.writeable,
                          state["rel"].flags.writeable, state["mu"].flags.writeable, state["link"]))
        return []


def run_spy(w):
    spy = Spy()
    simulate_layers(w["load"], w["nbytes"], w["rel"], w["node0"], w["mu"], w["link"], w["sizes"],
                    policy=spy, mode="coupled", **DEFAULTS)
    return spy


def test_policy_sees_only_the_visible_bytes():
    w = world(7)
    spy = run_spy(w)
    assert len(spy.rows) == w["load"].shape[0]
    for t, n_obs, n_bytes, *_ in spy.rows:
        assert n_bytes == n_obs == max(t - DEFAULTS["d"] + 1, 0)


def test_state_entries_are_read_only_and_match_the_inputs():
    w = world(7)
    spy = run_spy(w)
    assert all(not wr_b and not wr_r and not wr_m for _t, _o, _n, wr_b, wr_r, wr_m, _l in spy.rows)
    assert all(link == w["link"] for *_head, link in spy.rows)
    before = w["nbytes"].copy()
    run_spy(w)
    assert np.array_equal(before, w["nbytes"])
