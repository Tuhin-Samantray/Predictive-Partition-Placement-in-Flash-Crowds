import numpy as np


def make_case(seed=7, T=120, P=24, N=4, headroom=0.9):
    rng = np.random.default_rng(seed)
    load = rng.gamma(2.0, 3.0, size=(T, P))
    load[40:60, 3] *= 8
    mu = np.full(N, headroom * load.sum(axis=1).mean() / N)
    node0 = np.arange(P) % N
    return load, node0, mu


class GreedyStub:
    """Greedy hot-to-cold mover. mode 'reactive' reads the newest observed row;
    mode 'oracle' reads true load at t + m."""

    def __init__(self, load, mode, m=0, delta=0.0, max_moves=1):
        self.load, self.mode, self.m = load, mode, m
        self.delta, self.max_moves = delta, max_moves

    def decide(self, t, observed, state):
        if self.mode == "reactive":
            if observed.shape[0] == 0:
                return []
            row = observed[-1]
        else:
            row = self.load[min(t + self.m, self.load.shape[0] - 1)]
        node_of, n_nodes = state["node_of"], state["n_nodes"]
        nl = np.bincount(node_of, weights=row, minlength=n_nodes)
        movable = state["movable"].copy()
        plan = []
        for _ in range(min(self.max_moves, state["free_slots"])):
            hot, cold = int(nl.argmax()), int(nl.argmin())
            if hot == cold:
                break
            cand = np.flatnonzero((node_of == hot) & movable)
            if cand.size == 0:
                break
            others = np.delete(nl, [hot, cold])
            oth = others.max() if others.size else 0.0
            new_max = np.maximum.reduce(
                [nl[hot] - row[cand], nl[cold] + row[cand], np.full(cand.size, oth)]
            )
            j = int(new_max.argmin())
            if nl.max() - new_max[j] <= self.delta:
                break
            p = int(cand[j])
            plan.append((p, hot, cold))
            nl[hot] -= row[p]
            nl[cold] += row[p]
            movable[p] = False
        return plan


class ScriptedPolicy:
    def __init__(self, script):
        self.script = script
        self.seen = []

    def decide(self, t, observed, state):
        last = observed[-1].copy() if observed.shape[0] else None
        self.seen.append((t, observed.shape[0], last))
        return list(self.script.get(t, []))