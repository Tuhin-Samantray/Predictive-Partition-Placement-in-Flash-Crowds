"""Datasets, replicate windows and the single-replicate runner (Section 2, Phase 5).

A replicate is a (window, hash seed, capacity-jitter seed) triple. The window is anchored on
one detected event, lies inside one split span and contains the whole event. Capacity comes
from the training prefix only. All draws are seeded from config via derive_seed.
Forecast tables are indexed by origin, so a replicate's forecast window is the table rows
start .. start + window_steps; local row i is the forecast made at local step i.
"""
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import split_points
from src.metrics import compute_metrics
from src.policies import build_policy
from src.rv4 import LoggedPolicy, rv4_fields
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


def skewed_placement(cfg, data, label):
    """RV2 skewed initial placement (A38, A45). Partitions sorted by training-prefix mean load (stable,
    heaviest first); the top skew_hot_share of them are spread evenly over the first skew_hot_nodes nodes;
    the remaining partitions, in index order, get the seeded hash placement over the other nodes."""
    n_part, n_nodes = cfg["n_partitions"], cfg["n_nodes"]
    n_hot_nodes = cfg["skew_hot_nodes"]
    if n_hot_nodes >= n_nodes:
        raise ValueError("skew_hot_nodes must be smaller than n_nodes")
    n_hot = int(cfg["skew_hot_share"] * n_part)
    mean_load = np.asarray(data.load[: data.train_end], dtype=np.float64).mean(axis=0)
    order = np.argsort(-mean_load, kind="stable")
    node0 = np.zeros(n_part, dtype=np.int64)
    node0[order[:n_hot]] = np.arange(n_hot) % n_hot_nodes
    rest = np.sort(order[n_hot:])
    hash_rng = np.random.default_rng(derive_seed("hash", cfg["hash_seed"], *label))
    node0[rest] = n_hot_nodes + hash_rng.permutation(np.arange(rest.size) % (n_nodes - n_hot_nodes))
    return node0


def make_replicate_skewed(cfg, data, ev, j, headroom):
    """The replicate of make_replicate with the skewed initial placement; window, capacity and seeds unchanged."""
    rep = make_replicate(cfg, data, ev, j, headroom)
    label = (cfg["master_seed"], data.name, ev["event_id"], j)
    return replace(rep, node0=skewed_placement(cfg, data, label))


def tune_replicates(cfg, data):
    return [make_replicate(cfg, data, ev, 0, cfg["headroom"])
            for ev in select_events(cfg, data, "tune", cfg["tune_replicates"])]


def run_replicate(cfg, rep, name, params, m=None, forecast=None, budget=None, lead=None, margin_quantile=None,
                  kappa=None, log=False, d=None, k=None):
    """Run one policy on one replicate. `forecast` is the replicate's forecast window
    (see forecast_window); P3 and P4 require it, every other policy ignores it."""
    m = cfg["migration_steps_m0"] if m is None else m
    budget = cfg["migration_budget_B"] if budget is None else budget
    margin_quantile = cfg["margin_quantile"] if margin_quantile is None else margin_quantile
    d = cfg["monitor_delay_d"] if d is None else d
    k = cfg["decision_interval_k"] if k is None else k
    ctx = {"m": m if lead is None else lead, "d": d, "horizons": cfg["horizons"], "load": rep.load,
           "margin_quantile": margin_quantile, "residual_window": cfg["residual_window"],
           "gamma": cfg["gamma"], "a_min": cfg["a_min"], "a_max": cfg["a_max"]}
    policy = build_policy(name, params, rep.mu, ctx)
    if log:
        policy = LoggedPolicy(policy)
    res = simulate(rep.load, rep.node0, rep.mu, m=m, kappa=cfg["migration_overhead_kappa"] if kappa is None else kappa,
                   budget=budget, d=d,
                   k=k, policy=policy, cooldown=params.get("cooldown", 0),
                   forecast=forecast)
    out = compute_metrics(res, rep.load, slo=cfg["slo_threshold"],
                           wasted_window=cfg["wasted_move_window"], wait_quantile=cfg["wait_quantile"])
    if not log:
        return out
    out = dict(out, **policy.fields())
    if name in ("P4", "P4a"):
        out = dict(out, **rv4_fields(rep.load, forecast, ctx, margin_quantile))
    return out
