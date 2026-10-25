"""Per-replicate metrics (Section 4.1). All parameters are arguments."""
import numpy as np


def weighted_quantile(values, weights, q):
    order = np.argsort(values, kind="stable")
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    total = cw[-1] if cw.size else 0
    if total <= 0:
        return 0.0
    idx = int(np.searchsorted(cw, q * total, side="left"))
    return float(v[min(idx, v.size - 1)])


def wasted_flags(res, load, window):
    """True for a move whose source node, under true load with the partition kept on it,
    never exceeds mu within `window` steps from the move start."""
    n_steps = load.shape[0]
    flags = []
    for t0, p, src, _dst, _tf in res.moves:
        t1 = min(t0 + window, n_steps)
        hist = res.node_hist[t0:t1]
        cf = (load[t0:t1] * (hist == src)).sum(axis=1) + load[t0:t1, p] * (hist[:, p] != src)
        flags.append(bool((cf <= res.mu[src]).all()))
    return flags


def compute_metrics(res, load, *, slo, wasted_window, wait_quantile, part_size=None):
    load = np.asarray(load, dtype=np.float64)
    fg, wait, mu = res.fg, res.wait, res.mu
    total = float(fg.sum())
    if total > 0:
        damage = float((np.maximum(0, wait - slo) * fg).sum() / total)
        viol = float((fg * (wait > slo)).sum() / total)
    else:
        damage = 0.0
        viol = 0.0
    util = res.arrivals / mu
    mean_u = util.mean(axis=1)
    max_u = util.max(axis=1)
    live = mean_u > 0
    imbalance = float((max_u[live] / mean_u[live]).mean()) if live.any() else 0.0
    prev = np.vstack([np.zeros((1, mu.shape[0])), res.queue[:-1]])
    conservation = bool(np.allclose(res.arrivals + prev, res.served + res.queue))
    mig_load = float(sum(load[t0, p] for t0, p, _s, _d, _f in res.moves))
    out = {
        "damage": damage,
        "slo_violation_frac": viol,
        "wait_quantile_value": weighted_quantile(wait.ravel(), fg.ravel(), wait_quantile),
        "overload_steps": int((res.arrivals > mu).sum()),
        "peak_backlog_steps": float((res.queue / mu).max()),
        "imbalance": imbalance,
        "moves": len(res.moves),
        "migrated_load": mig_load,
        "migrated_load_share": mig_load / total if total > 0 else 0.0,
        "wasted_moves": int(sum(wasted_flags(res, load, wasted_window))),
        "dropped_moves": int(res.dropped_moves),
        "conservation_ok": conservation,
    }
    if part_size is not None:
        out["migrated_bytes"] = float(sum(part_size[p] for _t, p, _s, _d, _f in res.moves))
    return out