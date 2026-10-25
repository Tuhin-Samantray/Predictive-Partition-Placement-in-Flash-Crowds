"""Cross-layer building blocks (Section 4.4, Phase 9, decisions of amendment A24).

Pure functions; every parameter is an argument. CPU layer: demand is divided by the training-prefix
load-weighted mean demand, so CPU work has L1 units and cpu capacity equals mu. NIC layer: link capacity
comes from the training prefix; a move is a flow charged on its source and destination links.
"""
from dataclasses import dataclass

import numpy as np

CPU = 0
NET = 1


def demand_scale(train_load, demand):
    """Load-weighted mean service demand over the training prefix."""
    demand = np.asarray(demand, dtype=np.float64)
    if (demand <= 0).any():
        raise ValueError("service demand must be positive")
    weights = np.asarray(train_load, dtype=np.float64).sum(axis=0)
    total = weights.sum()
    if total <= 0:
        raise ValueError("training load is zero")
    return float((weights * demand).sum() / total)


def relative_demand(train_load, demand):
    """Demand in units of the training mean, so that CPU work is comparable with L1 load."""
    return np.asarray(demand, dtype=np.float64) / demand_scale(train_load, demand)


def link_capacity(train_bytes, h_net, n_nodes):
    """h_net x mean foreground bytes per node-step on the training prefix (no jitter)."""
    per_step = np.asarray(train_bytes, dtype=np.float64).sum(axis=1).mean()
    return float(h_net * per_step / n_nodes)


def partition_sizes(n_part, m0, rho, link, sigma, seed):
    """S_p = m0 x rho x link x z_p, z_p lognormal normalised to sample mean 1 (seeded)."""
    z = np.exp(sigma * np.random.default_rng(seed).standard_normal(n_part))
    return m0 * rho * link * z / z.mean()


def migration_rates(src, dst, net_fg, link, rho):
    """Rate of each in-flight move: min(rho x link, free source link / moves touching the source,
    free destination link / moves touching the destination); free = max(0, link - foreground)."""
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    n_nodes = np.asarray(net_fg).shape[0]
    if src.size == 0:
        return np.zeros(0)
    count = np.bincount(src, minlength=n_nodes) + np.bincount(dst, minlength=n_nodes)
    share = np.maximum(0, link - np.asarray(net_fg, dtype=np.float64)) / np.maximum(count, 1)
    return np.minimum(rho * link, np.minimum(share[src], share[dst]))


def charge_links(src, dst, rates, n_nodes):
    """Migration bytes per node-step: each move is charged on its source and its destination link."""
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    rates = np.asarray(rates, dtype=np.float64)
    return np.bincount(src, weights=rates, minlength=n_nodes) + np.bincount(dst, weights=rates, minlength=n_nodes)


def queue_step(q, inflow, cap):
    """One fluid-queue step, the same formulas as layer L1: returns wait, new backlog, served."""
    wait = (q + inflow / 2) / cap
    q_new = np.maximum(0, q + inflow - cap)
    return wait, q_new, q + inflow - q_new


def binding_layer(wait_cpu, wait_net):
    """CPU where the CPU wait is at least the NIC wait, NET otherwise (CL3)."""
    return np.where(wait_cpu >= wait_net, CPU, NET)


MODES = ("coupled", "const_m")


@dataclass
class LayerResult:
    arrivals: np.ndarray      # (T, N) CPU input incl. kappa overhead, in L1 units
    fg: np.ndarray            # (T, N) foreground load
    wait: np.ndarray          # (T, N) max(CPU wait, NIC wait)
    queue: np.ndarray         # (T, N) CPU backlog after the step
    served: np.ndarray        # (T, N) CPU work served
    node_hist: np.ndarray     # (T, P) serving node of each partition
    moves: list               # (t_start, p, src, dst, t_finish)
    mu: np.ndarray            # (N,) CPU capacity per step
    dropped_moves: int
    wait_cpu: np.ndarray
    wait_net: np.ndarray
    binding: np.ndarray       # (T, N) CPU or NET
    net_in: np.ndarray        # (T, N) foreground plus migration bytes
    net_queue: np.ndarray
    net_served: np.ndarray
    mig_charge: np.ndarray    # (T, N) migration bytes charged on each node link
    sent: np.ndarray          # (T,) migration bytes sent in the step
    link: float


def attribution(res, slo):
    """CL3: shares of violating node-steps (wait above slo) by binding layer."""
    viol = res.wait > slo
    n = int(viol.sum())
    if n == 0:
        return {"n": 0, "cpu": 0, "net": 0}
    cpu = int((viol & (res.binding == CPU)).sum())
    return {"n": n, "cpu": cpu / n, "net": (n - cpu) / n}


def simulate_layers(load, nbytes, rel, node0, mu, link, sizes, *, m, kappa, budget, d, k, rho, phi, mode,
                    policy=None, cooldown=0, forecast=None, nic=True):
    """Cross-layer replicate (A24, A25). Same step loop as layer L1 plus a NIC layer and migration flows."""
    if mode not in MODES:
        raise ValueError("mode must be one of %s" % (MODES,))
    if mode == "coupled" and not nic:
        raise ValueError("coupled migration needs the NIC layer")
    load = np.asarray(load, dtype=np.float64)
    nbytes = np.asarray(nbytes, dtype=np.float64)
    rel = np.asarray(rel, dtype=np.float64)
    sizes = np.asarray(sizes, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    if load.ndim != 2 or (load < 0).any() or nbytes.shape != load.shape or (nbytes < 0).any():
        raise ValueError("load and nbytes must be non-negative 2-D arrays of one shape (T, P)")
    n_steps, n_part = load.shape
    if rel.shape != (n_part,) or (rel <= 0).any() or sizes.shape != (n_part,) or (sizes < 0).any():
        raise ValueError("rel must be positive and sizes non-negative, one value per partition")
    if mu.ndim != 1 or (mu <= 0).any() or link <= 0 or rho <= 0 or phi < 0:
        raise ValueError("need mu > 0, link > 0, rho > 0, phi >= 0")
    n_nodes = mu.shape[0]
    node_of = np.array(node0, dtype=np.int64)
    if node_of.shape != (n_part,) or node_of.min() < 0 or node_of.max() >= n_nodes:
        raise ValueError("node0 must have one valid node per partition")
    if m < 0 or d < 0 or k < 1 or cooldown < 0:
        raise ValueError("need m >= 0, d >= 0, k >= 1, cooldown >= 0")

    seen = load.view()
    seen.flags.writeable = False
    nseen = nbytes.view()
    nseen.flags.writeable = False
    rel_ro = rel.view()
    rel_ro.flags.writeable = False
    mu_ro = mu.view()
    mu_ro.flags.writeable = False
    fview = None
    if forecast is not None:
        fview = np.asarray(forecast).view()
        if fview.ndim != 3 or fview.shape[0] != n_steps or fview.shape[2] != n_part:
            raise ValueError("forecast must have shape (T, horizons, P) matching the load window")
        fview.flags.writeable = False

    coupled = mode == "coupled"
    moving = np.zeros(n_part, dtype=bool)
    finish = np.zeros(n_part, dtype=np.int64)
    started = np.zeros(n_part, dtype=np.int64)
    dest = np.zeros(n_part, dtype=np.int64)
    ready_at = np.zeros(n_part, dtype=np.int64)
    remaining = np.zeros(n_part)
    move_of = np.zeros(n_part, dtype=np.int64)

    arrivals = np.zeros((n_steps, n_nodes))
    fg = np.zeros((n_steps, n_nodes))
    wait = np.zeros((n_steps, n_nodes))
    queue = np.zeros((n_steps, n_nodes))
    served = np.zeros((n_steps, n_nodes))
    wait_cpu = np.zeros((n_steps, n_nodes))
    wait_net = np.zeros((n_steps, n_nodes))
    binding = np.full((n_steps, n_nodes), CPU, dtype=np.int64)
    net_in = np.zeros((n_steps, n_nodes))
    net_queue = np.zeros((n_steps, n_nodes))
    net_served = np.zeros((n_steps, n_nodes))
    mig_charge = np.zeros((n_steps, n_nodes))
    sent = np.zeros(n_steps)
    node_hist = np.zeros((n_steps, n_part), dtype=np.int64)
    moves = []
    dropped = 0
    q_cpu = np.zeros(n_nodes)
    q_net = np.zeros(n_nodes)

    def complete(t):
        if coupled:
            done = moving & (remaining <= 0) & (started < t)
        else:
            done = moving & (finish <= t)
        if done.any():
            if coupled:
                for p in np.flatnonzero(done):
                    moves[move_of[p]][4] = t
                finish[done] = t
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
                "nbytes": nseen[:visible],
                "rel": rel_ro,
                "mu": mu_ro,
                "link": float(link),
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
                    started[p] = t
                    if coupled:
                        remaining[p] = phi * sizes[p]
                        finish[p] = n_steps
                    else:
                        finish[p] = t + m
                    movable[p] = False
                    free -= 1
                    moves.append([t, p, src, dst, int(finish[p])])
                    move_of[p] = len(moves) - 1
                else:
                    dropped += 1
            complete(t)

        node_hist[t] = node_of
        row = load[t]
        fg[t] = np.bincount(node_of, weights=row, minlength=n_nodes)
        a = np.bincount(node_of, weights=row * rel * (1 + kappa * moving), minlength=n_nodes)
        wait_cpu[t], q_new, served[t] = queue_step(q_cpu, a, mu)
        arrivals[t] = a
        q_cpu = q_new
        queue[t] = q_cpu
        if nic:
            net_fg = np.bincount(node_of, weights=nbytes[t], minlength=n_nodes)
            charge = np.zeros(n_nodes)
            if coupled and moving.any():
                idx = np.flatnonzero(moving)
                rates = migration_rates(node_of[idx], dest[idx], net_fg, link, rho)
                rates = np.minimum(rates, remaining[idx])
                remaining[idx] -= rates
                charge = charge_links(node_of[idx], dest[idx], rates, n_nodes)
                sent[t] = rates.sum()
            mig_charge[t] = charge
            net_in[t] = net_fg + charge
            wait_net[t], qn_new, net_served[t] = queue_step(q_net, net_in[t], link)
            q_net = qn_new
            net_queue[t] = q_net
            binding[t] = binding_layer(wait_cpu[t], wait_net[t])
            wait[t] = np.maximum(wait_cpu[t], wait_net[t])
        else:
            wait[t] = wait_cpu[t]

    moves = [tuple(x) for x in moves]
    return LayerResult(arrivals, fg, wait, queue, served, node_hist, moves, mu, dropped, wait_cpu, wait_net,
                       binding, net_in, net_queue, net_served, mig_charge, sent, float(link))
