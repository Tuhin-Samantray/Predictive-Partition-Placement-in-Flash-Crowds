"""Datasets, replicate windows and the single-replicate runner (Section 2, Phase 5).

A replicate is a (window, hash seed, capacity-jitter seed) triple. The window is anchored on
one detected event, lies inside one split span and contains the whole event. Capacity comes
from the training prefix only. All draws are seeded from config via derive_seed.
Forecast tables are indexed by origin, so a replicate's forecast window is the table rows
start .. start + window_steps; local row i is the forecast made at local step i.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import split_points
from src.metrics import compute_metrics
from src.policies import build_policy
from src.seeds import derive_seed
from src.simulator import node_capacity, simulate


@dataclass
class DataSet:
    name: str
    load: np.ndarray
    events: pd.DataFrame
    train_end: int
    tune_end: int
    n_steps: int


@dataclass
class Replicate:
    event_id: str
    dataset: str
    split: str
    onset_class: str
    j: int
    onset: int
    end: int
    start: int
    load: np.ndarray
    node0: np.ndarray
    mu: np.ndarray


def load_dataset(cfg, name):
    base = Path(cfg["data_root"]) / cfg["processed_dirname"]
    load = np.load(base / (name + "_L.npy"), mmap_mode="r")
    events = pd.read_csv(base / (name + "_events.csv"))
    train_end, tune_end = split_points(cfg, name, load.shape[0])
    expected = np.where(events["onset"] < train_end, "train",
                        np.where(events["onset"] < tune_end, "tune", "eval"))
    if not (expected == events["split"].to_numpy()).all():
        raise ValueError("event split labels disagree with split_points for " + name)
    return DataSet(name, load, events, train_end, tune_end, load.shape[0])


def load_forecast(cfg, dataset, forecaster):
    """Memory-mapped forecast table F[origin, horizon, partition] for one (dataset, forecaster)."""
    base = Path(cfg["data_root"]) / cfg["processed_dirname"] / cfg["forecast_dirname"]
    return np.load(base / ("%s_%s.npy" % (dataset, forecaster)), mmap_mode="r")


def forecast_window(table, rep):
    """Forecast rows for a replicate window; row i is the forecast origin at local step i."""
    return table[rep.start: rep.start + rep.load.shape[0]]


def split_span(data, split):
    if split == "tune":
        return data.train_end, data.tune_end
    if split == "eval":
        return data.tune_end, data.n_steps
    raise ValueError("split must be tune or eval")


def window_bounds(onset, end, lo, hi, w):
    """Feasible window starts: inside [lo, hi) and containing steps onset..end."""
    a_lo = np.maximum(lo, np.asarray(end) - w + 1)
    a_hi = np.minimum(np.asarray(onset), hi - w)
    return a_lo, a_hi


def eligible_events(cfg, data, split):
    lo, hi = split_span(data, split)
    ev = data.events[data.events["split"] == split]
    a_lo, a_hi = window_bounds(ev["onset"].to_numpy(), ev["end"].to_numpy(), lo, hi, cfg["window_steps"])
    return ev[a_lo <= a_hi].reset_index(drop=True)


def select_events(cfg, data, split, n):
    """n distinct eligible events, chosen with a seed that does not depend on policy or config."""
    elig = eligible_events(cfg, data, split)
    if len(elig) < n:
        raise ValueError("only %d eligible %s events in %s, need %d" % (len(elig), split, data.name, n))
    rng = np.random.default_rng(derive_seed("select", cfg["master_seed"], data.name, split))
    return [elig.iloc[int(i)] for i in rng.permutation(len(elig))[:n]]


def make_replicate(cfg, data, ev, j, headroom):
    w = cfg["window_steps"]
    split = ev["split"]
    lo, hi = split_span(data, split)
    a_lo, a_hi = window_bounds(ev["onset"], ev["end"], lo, hi, w)
    if a_lo > a_hi:
        raise ValueError("event " + str(ev["event_id"]) + " has no feasible window")
    label = (cfg["master_seed"], data.name, ev["event_id"], j)
    start = int(np.random.default_rng(derive_seed("offset", *label)).integers(a_lo, a_hi + 1))
    n_part, n_nodes = cfg["n_partitions"], cfg["n_nodes"]
    hash_rng = np.random.default_rng(derive_seed("hash", cfg["hash_seed"], *label))
    node0 = hash_rng.permutation(np.arange(n_part) % n_nodes)
    mu = node_capacity(data.load[: data.train_end], headroom, n_nodes,
                       jitter_frac=cfg["capacity_jitter_frac"], seed=derive_seed("jitter", *label))
    window = np.array(data.load[start:start + w], dtype=np.float64)
    return Replicate(str(ev["event_id"]), data.name, split, str(ev["onset_class"]), j,
                     int(ev["onset"]), int(ev["end"]), start, window, node0, mu)


def tune_replicates(cfg, data):
    return [make_replicate(cfg, data, ev, 0, cfg["headroom"])
            for ev in select_events(cfg, data, "tune", cfg["tune_replicates"])]


def run_replicate(cfg, rep, name, params, m=None, forecast=None):
    """Run one policy on one replicate. `forecast` is the replicate's forecast window
    (see forecast_window); P3 and P4 require it, every other policy ignores it."""
    m = cfg["migration_steps_m0"] if m is None else m
    ctx = {"m": m, "d": cfg["monitor_delay_d"], "horizons": cfg["horizons"], "load": rep.load,
           "margin_quantile": cfg["margin_quantile"], "residual_window": cfg["residual_window"]}
    policy = build_policy(name, params, rep.mu, ctx)
    res = simulate(rep.load, rep.node0, rep.mu, m=m, kappa=cfg["migration_overhead_kappa"],
                   budget=cfg["migration_budget_B"], d=cfg["monitor_delay_d"],
                   k=cfg["decision_interval_k"], policy=policy, cooldown=params.get("cooldown", 0),
                   forecast=forecast)
    return compute_metrics(res, rep.load, slo=cfg["slo_threshold"],
                           wasted_window=cfg["wasted_move_window"], wait_quantile=cfg["wait_quantile"])
