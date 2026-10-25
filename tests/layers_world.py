import numpy as np

from src.layers import partition_sizes, simulate_layers

N_NODES = 4
N_PART = 8
N_STEPS = 40
PLANS = {2: [(0, 0, 1)], 5: [(1, 1, 2), (2, 2, 3)], 12: [(0, 1, 0)]}
DEFAULTS = dict(m=3, kappa=0.3, budget=4, d=1, k=1, rho=0.25, phi=1.0, cooldown=2)
ARRAYS = ("arrivals", "fg", "wait", "queue", "served", "node_hist", "wait_cpu", "wait_net", "binding",
          "net_in", "net_queue", "net_served", "mig_charge", "sent")


class Scripted:
    def __init__(self, plans):
        self.plans = plans

    def decide(self, t, observed, state):
        return list(self.plans.get(t, []))


def world(seed=0):
    rng = np.random.default_rng(seed)
    load = rng.uniform(0, 10, (N_STEPS, N_PART))
    nbytes = rng.uniform(0, 50, (N_STEPS, N_PART))
    link = 1.1 * nbytes.sum(axis=1).mean() / N_NODES
    return {"load": load, "nbytes": nbytes, "rel": np.ones(N_PART), "node0": np.arange(N_PART) % N_NODES,
            "mu": np.full(N_NODES, 11.0), "link": link,
            "sizes": partition_sizes(N_PART, 5, 0.25, link, 0.5, seed + 1)}


def run(w, plans=None, **kw):
    args = dict(DEFAULTS)
    args.update(kw)
    policy = None if plans is None else Scripted(plans)
    return simulate_layers(w["load"], w["nbytes"], w["rel"], w["node0"], w["mu"], w["link"], w["sizes"],
                           policy=policy, **args)


def same(a, b):
    for name in ARRAYS:
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert a.moves == b.moves
    assert a.dropped_moves == b.dropped_moves
