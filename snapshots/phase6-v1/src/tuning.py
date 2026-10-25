"""Seeded random-search tuner shared by every tunable policy (equal budget, paired replicates)."""
import math
import time

import numpy as np
from sklearn.model_selection import ParameterSampler

from src.policies import TUNABLE
from src.replicates import run_replicate
from src.seeds import derive_seed


def tuning_grid(cfg, name):
    return {param: list(cfg[key]) for param, key in TUNABLE[name].items()}


def plateau_gain(curve, frac):
    """Relative gain of the best-so-far curve over its last frac of iterations."""
    n = len(curve)
    n_last = max(1, int(frac * n))
    ref = curve[0] if n <= n_last else curve[n - n_last - 1]
    return (ref - curve[-1]) / ref if ref > 0 else 0.0


def evaluate_config(cfg, name, params, reps):
    mets = [run_replicate(cfg, rep, name, params) for rep in reps]
    if not all(m["conservation_ok"] for m in mets):
        raise RuntimeError("conservation violated while tuning " + name)
    damages = [float(m["damage"]) for m in mets]
    return {
        "mean_damage": float(np.mean(damages)),
        "mean_moves": float(np.mean([m["moves"] for m in mets])),
        "mean_wasted": float(np.mean([m["wasted_moves"] for m in mets])),
        "mean_slo_violation": float(np.mean([m["slo_violation_frac"] for m in mets])),
        "damages": damages,
    }


def tune_policy(cfg, name, dataset, reps):
    grid = tuning_grid(cfg, name)
    size = math.prod(len(v) for v in grid.values())
    n_iter = cfg["tuning_budget"]
    if size < n_iter:
        raise ValueError("grid of %s has %d configs, fewer than tuning_budget %d" % (name, size, n_iter))
    seed = derive_seed("tune", cfg["master_seed"], dataset, name) % (1 << cfg["seed_bits"])
    sampler = ParameterSampler(grid, n_iter=n_iter, random_state=np.random.RandomState(seed))
    rows, timing, curve = [], [], []
    best_key, best_params, best_iter = None, None, None
    for it, params in enumerate(sampler):
        params = dict(params)
        t0 = time.perf_counter()
        stats = evaluate_config(cfg, name, params, reps)
        timing.append(time.perf_counter() - t0)
        key = (stats["mean_damage"], stats["mean_moves"])
        if best_key is None or key < best_key:
            best_key, best_params, best_iter = key, params, it
        curve.append(best_key[0])
        rows.append({"type": "iter", "dataset": dataset, "policy": name, "iter": it,
                     "params": params, "best_so_far": best_key[0], "best_iter": best_iter, **stats})
    gain = plateau_gain(curve, cfg["tune_plateau_frac"])
    summary = {
        "type": "summary", "dataset": dataset, "policy": name, "best_params": best_params,
        "best_iter": best_iter, "best_mean_damage": best_key[0], "best_mean_moves": best_key[1],
        "plateau_gain": gain, "plateau_tol": cfg["tune_plateau_tol"],
        "converged": bool(gain <= cfg["tune_plateau_tol"]), "n_configs": n_iter, "grid_size": size,
        "n_replicates": len(reps), "n_positive_damage": int(sum(r["mean_damage"] > 0 for r in rows)),
        "event_ids": [r.event_id for r in reps],
    }
    return rows, summary, timing