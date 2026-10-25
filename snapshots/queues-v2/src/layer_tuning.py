"""Layers-on tuner (Phase 9, A32).

Same selection rule as src/tuning.py (lowest mean damage, then lowest mean moves), same seeded sampling for P1,
exhaustive search for the small grids, but every replicate runs through run_replicate_layers in the tuning
conditions of A30 item 4: coupled mode, config h_net, config phi, default m0 and headroom, tuning split.
"""
import math
import time

import numpy as np
from sklearn.model_selection import ParameterGrid, ParameterSampler

from src.layer_runs import run_replicate_layers
from src.policies import FORECAST_TUNABLE, LAYER_TUNABLE, TUNABLE
from src.replicates import forecast_window
from src.seeds import derive_seed
from src.tuning import plateau_gain

LAYER_POLICIES = ("P1", "P3", "P5", "P6")
SPACES = {"P1": TUNABLE["P1"], "P3": FORECAST_TUNABLE["P3"], "P5": FORECAST_TUNABLE["P5"], "P6": LAYER_TUNABLE["P6"]}
TUNING_MODE = "coupled"


def layer_grid(cfg, name):
    return {param: list(cfg[key]) for param, key in SPACES[name].items()}


def evaluate_config_layers(cfg, name, params, reps, cross, table=None):
    mets = [run_replicate_layers(cfg, rep, cross, name, params,
                                 forecast=None if table is None else forecast_window(table, rep),
                                 mode=TUNING_MODE)
            for rep in reps]
    if not all(m["conservation_ok"] for m in mets):
        raise RuntimeError("conservation violated while tuning " + name + " with layers on")
    damages = [float(m["damage"]) for m in mets]
    return {
        "mean_damage": float(np.mean(damages)),
        "mean_moves": float(np.mean([m["moves"] for m in mets])),
        "mean_wasted": float(np.mean([m["wasted_moves"] for m in mets])),
        "mean_slo_violation": float(np.mean([m["slo_violation_frac"] for m in mets])),
        "damages": damages,
    }


def tune_policy_layers(cfg, name, dataset, reps, cross, table=None, exhaustive=False, forecaster=None):
    grid = layer_grid(cfg, name)
    size = math.prod(len(v) for v in grid.values())
    if exhaustive:
        configs = list(ParameterGrid(grid))
        n_iter = size
    else:
        n_iter = cfg["tuning_budget"]
        if size < n_iter:
            raise ValueError("grid of %s has %d configs, fewer than tuning_budget %d" % (name, size, n_iter))
        seed = derive_seed("tune", cfg["master_seed"], dataset, name) % (1 << cfg["seed_bits"])
        configs = ParameterSampler(grid, n_iter=n_iter, random_state=np.random.RandomState(seed))
    tag = {"layers": True, "mode": TUNING_MODE, "h_net": cross.h_net, "phi": cfg["phi"]}
    if forecaster is not None:
        tag["forecaster"] = forecaster
    rows, timing, curve = [], [], []
    best_key, best_params, best_iter = None, None, None
    for it, params in enumerate(configs):
        params = dict(params)
        t0 = time.perf_counter()
        stats = evaluate_config_layers(cfg, name, params, reps, cross, table)
        timing.append(time.perf_counter() - t0)
        key = (stats["mean_damage"], stats["mean_moves"])
        if best_key is None or key < best_key:
            best_key, best_params, best_iter = key, params, it
        curve.append(best_key[0])
        rows.append({"type": "iter", "dataset": dataset, "policy": name, "iter": it,
                     "params": params, "best_so_far": best_key[0], "best_iter": best_iter,
                     **tag, **stats})
    gain = plateau_gain(curve, cfg["tune_plateau_frac"])
    summary = {
        "type": "summary", "dataset": dataset, "policy": name, "best_params": best_params,
        "best_iter": best_iter, "best_mean_damage": best_key[0], "best_mean_moves": best_key[1],
        "plateau_gain": gain, "plateau_tol": cfg["tune_plateau_tol"],
        "converged": bool(exhaustive or gain <= cfg["tune_plateau_tol"]), "n_configs": n_iter,
        "grid_size": size, "n_replicates": len(reps),
        "n_positive_damage": int(sum(r["mean_damage"] > 0 for r in rows)),
        "event_ids": [r.event_id for r in reps], **tag,
    }
    if exhaustive:
        summary["exhaustive"] = True
    return rows, summary, timing
