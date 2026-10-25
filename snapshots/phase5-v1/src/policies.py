"""Reactive policies P0 to P2 (Section 4.2) and the shared greedy repack.

A policy sees only `observed` (load up to t - d, enforced by the simulator) and the
placement state. Every parameter is an argument; none is defined in this module.
Use one policy instance per replicate: the EWMA state is not reset between runs.
"""
import numpy as np

# tunable parameter -> config key holding its grid
TUNABLE = {
    "P1": {"ewma_alpha": "ewma_alpha_grid", "theta_hi": "theta_hi_grid", "cooldown": "cooldown_grid"},
    "P2": {"ewma_alpha": "ewma_alpha_grid", "hysteresis": "hysteresis_grid", "cooldown": "cooldown_grid"},
}


class Ewma:
    """Per-partition exponentially weighted moving average, updated incrementally."""

    def __init__(self, alpha):
        self.alpha = alpha
        self.value = None
        self.n_seen = 0

    def update(self, observed):
        if observed.shape[0] < self.n_seen:
            raise ValueError("policy instance reused across runs")
        for row in observed[self.n_seen:]:
            if self.value is None:
                self.value = np.array(row, dtype=np.float64)
            else:
                self.value = self.alpha * row + (1 - self.alpha) * self.value
        self.n_seen = observed.shape[0]
        return self.value


def greedy_plan(pred, node_load, node_of, movable, free, delta):
    """Greedy hot-to-cold repack (Section 4.5 sketch). `pred` is the per-partition load the
    policy believes in (EWMA for P2, forecast for P3, truth for P5)."""
    nl = np.array(node_load, dtype=np.float64)
    movable = np.array(movable, dtype=bool)
    plan = []
    for _ in range(free):
        hot, cold = int(nl.argmax()), int(nl.argmin())
        if hot == cold:
            break
        cand = np.flatnonzero((node_of == hot) & movable)
        if cand.size == 0:
            break
        others = np.delete(nl, [hot, cold])
        rest = others.max() if others.size else 0
        new_max = np.maximum.reduce(
            [nl[hot] - pred[cand], nl[cold] + pred[cand], np.full(cand.size, rest)]
        )
        j = int(new_max.argmin())
        if nl.max() - new_max[j] <= delta:
            break
        p = int(cand[j])
        plan.append((p, hot, cold))
        nl[hot] -= pred[p]
        nl[cold] += pred[p]
        movable[p] = False
    return plan


class P0Static:
    """Static hash placement: never moves."""

    def decide(self, t, observed, state):
        return []


class P1Threshold:
    """E-Store style: a node whose EWMA load exceeds theta_hi x capacity sheds its hottest
    movable partition that fits under the limit of the least-utilized node."""

    def __init__(self, mu, ewma_alpha, theta_hi):
        self.mu = np.asarray(mu, dtype=np.float64)
        self.limit = theta_hi * self.mu
        self.ewma = Ewma(ewma_alpha)

    def decide(self, t, observed, state):
        e = self.ewma.update(observed)
        if e is None:
            return []
        node_of, n_nodes = state["node_of"], state["n_nodes"]
        nl = np.bincount(node_of, weights=e, minlength=n_nodes)
        movable = np.array(state["movable"], dtype=bool)
        free = state["free_slots"]
        plan = []
        for hot in np.argsort(-(nl - self.limit), kind="stable"):
            hot = int(hot)
            while free > 0 and nl[hot] > self.limit[hot]:
                cold = int(np.argmin(nl / self.mu))
                if cold == hot:
                    break
                fits = (node_of == hot) & movable & (e + nl[cold] <= self.limit[cold])
                cand = np.flatnonzero(fits)
                if cand.size == 0:
                    break
                p = int(cand[np.argmax(e[cand])])
                plan.append((p, hot, cold))
                nl[hot] -= e[p]
                nl[cold] += e[p]
                movable[p] = False
                free -= 1
        return plan


class P2Greedy:
    """Reactive greedy repack on EWMA load; hysteresis is a fraction of mean node load;
    the number of moves is capped by the free migration budget."""

    def __init__(self, ewma_alpha, hysteresis):
        self.ewma = Ewma(ewma_alpha)
        self.hysteresis = hysteresis

    def decide(self, t, observed, state):
        e = self.ewma.update(observed)
        if e is None:
            return []
        node_of = state["node_of"]
        nl = np.bincount(node_of, weights=e, minlength=state["n_nodes"])
        return greedy_plan(e, nl, node_of, state["movable"], state["free_slots"],
                           self.hysteresis * nl.mean())


def build_policy(name, params, mu):
    if name == "P0":
        return P0Static()
    if name == "P1":
        return P1Threshold(mu, params["ewma_alpha"], params["theta_hi"])
    if name == "P2":
        return P2Greedy(params["ewma_alpha"], params["hysteresis"])
    raise ValueError("unknown policy: " + str(name))