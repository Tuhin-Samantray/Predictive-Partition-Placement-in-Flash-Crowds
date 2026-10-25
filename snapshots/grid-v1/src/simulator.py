"""Layer L1 fluid-queue simulator (Section 4.1).

Per step t: (a) moves whose finish step is <= t complete; (b) every k steps the
policy sees load up to t - d and may start moves; a move started at t finishes
at t + m (m = 0 is instantaneous); (c) the step is served, the source of an
in-flight move paying factor (1 + kappa).
A forecast table F (T, horizons, P) indexed by forecast origin may be supplied. The
policy at step t is handed only F[: t - d + 1] as a read-only view, so the newest
forecast it can read is F[t - d], the one made from the load it is allowed to see.
Every numeric parameter is an argument; none is defined in this module.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class SimResult:
    arrivals: np.ndarray      # (T, N) queue input incl. migration overhead
    fg: np.ndarray            # (T, N) foreground arrivals
    wait: np.ndarray          # (T, N) wait proxy (Q + A/2) / mu
    queue: np.ndarray         # (T, N) backlog after the step
    served: np.ndarray        # (T, N) work served
    node_hist: np.ndarray     # (T, P) serving node of each partition
    moves: list               # (t_start, p, src, dst, t_finish)
    mu: np.ndarray            # (N,) capacity per node per step
    dropped_moves: int


def initial_placement(n_partitions, n_nodes):
    """Static hash placement: partition ids are already hashed, so mod N."""
    return np.arange(n_partitions) % n_nodes


def node_capacity(train_load, headroom, n_nodes, jitter_frac=0, seed=None):
    """mu = h x mean total load per step on the training prefix / N, optional seeded jitter."""
    base = headroom * train_load.sum(axis=1).mean() / n_nodes
    mu = np.full(n_nodes, base, dtype=np.float64)
    if jitter_frac > 0:
        rng = np.random.default_rng(seed)
        mu = mu * rng.uniform(1 - jitter_frac, 1 + jitter_frac, n_nodes)
    return mu


def simulate(load, node0, mu, *, m, kappa, budget, d, k, policy=None, cooldown=0, forecast=None):
    """Run one replicate window. budget=None means unlimited moves in flight."""
    load = np.asarray(load, dtype=np.float64)
    if load.ndim != 2 or (load < 0).any():
        raise ValueError("load must be a non-negative 2-D array (T, P)")
    n_steps, n_part = load.shape
    mu = np.asarray(mu, dtype=np.float64)
    if mu.ndim != 1 or (mu <= 0).any():
        raise ValueError("mu must be a positive 1-D array (N,)")
    n_nodes = mu.shape[0]
    node_of = np.array(node0, dtype=np.int64)
    if node_of.shape != (n_part,) or node_of.min() < 0 or node_of.max() >= n_nodes:
        raise ValueError("node0 must have one valid node per partition")
    if m < 0 or d < 0 or k < 1 or cooldown < 0:
        raise ValueError("need m >= 0, d >= 0, k >= 1, cooldown >= 0")

    seen = load.view()
    seen.flags.writeable = False          # policies get a read-only view

    fview = None
    if forecast is not None:
        fview = np.asarray(forecast).view()
        if fview.ndim != 3 or fview.shape[0] != n_steps or fview.shape[2] != n_part:
            raise ValueError("forecast must have shape (T, horizons, P) matching the load window")
        fview.flags.writeable = False

    moving = np.zeros(n_part, dtype=bool)
    finish = np.zeros(n_part, dtype=np.int64)
    dest = np.zeros(n_part, dtype=np.int64)
    ready_at = np.zeros(n_part, dtype=np.int64)

    arrivals = np.zeros((n_steps, n_nodes))
    fg = np.zeros((n_steps, n_nodes))
    wait = np.zeros((n_steps, n_nodes))
    queue = np.zeros((n_steps, n_nodes))
    served = np.zeros((n_steps, n_nodes))
    node_hist = np.zeros((n_steps, n_part), dtype=np.int64)
    moves = []
    dropped = 0
    q = np.zeros(n_nodes)

    def complete(t):
        done = moving & (finish <= t)
        if done.any():
            node_of[done] = dest[done]
            ready_at[done] = finish[done] + cooldown
            moving[done] = False

    for t in range(n_steps):
        complete(t)
        if policy is not None and t % k == 0:
            free = n_part if budget is None else budget - int(moving.sum())
            movable = (~moving) & (ready_at <= t)
            visible = max(t - d + 1, 0)
            state = {
                "node_of": node_of.copy(),
                "moving": moving.copy(),
                "movable": movable.copy(),
                "free_slots": max(free, 0),
                "n_nodes": n_nodes,
                "forecast": None if fview is None else fview[:visible],
            }
            plan = policy.decide(t, seen[:visible], state)
            for p, src, dst in plan:
                p, src, dst = int(p), int(src), int(dst)
                ok = (
                    free > 0
                    and 0 <= p < n_part
                    and 0 <= src < n_nodes
                    and 0 <= dst < n_nodes
                    and src != dst
                    and movable[p]
                    and node_of[p] == src
                )
                if ok:
                    moving[p] = True
                    dest[p] = dst
                    finish[p] = t + m
                    movable[p] = False
                    free -= 1
                    moves.append((t, p, src, dst, t + m))
                else:
                    dropped += 1
            complete(t)

        node_hist[t] = node_of
        row = load[t]
        fg[t] = np.bincount(node_of, weights=row, minlength=n_nodes)
        a = np.bincount(node_of, weights=row * (1 + kappa * moving), minlength=n_nodes)
        wait[t] = (q + a / 2) / mu
        q_new = np.maximum(0, q + a - mu)
        served[t] = q + a - q_new
        arrivals[t] = a
        q = q_new
        queue[t] = q

    return SimResult(arrivals, fg, wait, queue, served, node_hist, moves, mu, dropped)
