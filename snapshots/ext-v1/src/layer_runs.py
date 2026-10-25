"""Cross-layer replicate runner (Phase 9, A30; CL fields and CL4 replay, A39).

Builds the per-dataset cross-layer inputs of amendment A24 and runs one policy on one replicate
with simulate_layers. Every parameter comes from the config; none is defined here.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.layers import attribution, link_capacity, partition_sizes, relative_demand, simulate_layers
from src.metrics import compute_metrics
from src.policies import build_policy
from src.rv4 import LoggedPolicy, rv4_fields
from src.seeds import derive_seed


@dataclass
class CrossData:
    nbytes: np.ndarray
    rel: np.ndarray
    link: float
    sizes: np.ndarray
    h_net: float


def load_cross(cfg, data, h_net=None):
    """Foreground bytes, relative demand, link capacity and partition sizes of one dataset."""
    h_net = cfg["h_net"] if h_net is None else h_net
    base = Path(cfg["data_root"]) / cfg["processed_dirname"]
    nbytes = np.load(base / (data.name + "_Bytes.npy"), mmap_mode="r")
    demand = np.load(base / (data.name + "_demand.npy"))
    if nbytes.shape != data.load.shape:
        raise ValueError("Bytes and load shapes differ for " + data.name)
    link = link_capacity(nbytes[: data.train_end], h_net, cfg["n_nodes"])
    rel = relative_demand(data.load[: data.train_end], demand)
    seed = derive_seed(cfg["size_label"], cfg["master_seed"], data.name)
    sizes = partition_sizes(cfg["n_partitions"], cfg["migration_steps_m0"], cfg["rho"], link,
                            cfg["sigma_size"], seed)
    return CrossData(nbytes, rel, link, sizes, h_net)


class ReplayPolicy:
    """Re-issues the moves of a finished run at their start steps, in their original order (A39, CL4)."""

    def __init__(self, moves):
        self.by_step = {}
        for t0, p, src, dst, _tf in moves:
            self.by_step.setdefault(int(t0), []).append((int(p), int(src), int(dst)))

    def decide(self, t, observed, state):
        return list(self.by_step.get(t, []))


def weighted_mean(values, weights):
    total = float(weights.sum())
    return float((values * weights).sum() / total) if total > 0 else 0.0


def layer_metrics(res, slo):
    """Per-run raw fields for CL1, CL2, CL5, CL6 and CL9 (A39); ratios and correlations are formed in analysis."""
    over_cpu = res.arrivals > res.mu
    over_net = res.net_in > res.link
    viol_cpu = res.wait_cpu > slo
    viol_net = res.wait_net > slo
    total_net = float(res.net_in.sum())
    return {
        "cl_util_cpu": float((res.arrivals / res.mu).mean()),
        "cl_util_net": float((res.net_in / res.link).mean()),
        "cl_viol_cpu": int(viol_cpu.sum()),
        "cl_viol_net": int(viol_net.sum()),
        "cl_viol_both": int((viol_cpu & viol_net).sum()),
        "cl_over_cpu": int(over_cpu.sum()),
        "cl_over_net": int(over_net.sum()),
        "cl_over_both": int((over_cpu & over_net).sum()),
        "cl_mig_share": float(res.mig_charge.sum()) / total_net if total_net > 0 else 0.0,
        "cl_bytes_sent": float(res.sent.sum()),
        "cl_wait_mean": weighted_mean(res.wait, res.fg),
        "cl_wait_cpu_mean": weighted_mean(res.wait_cpu, res.fg),
        "cl_wait_net_mean": weighted_mean(res.wait_net, res.fg),
    }


def run_replicate_layers(cfg, rep, cross, name, params, m=None, forecast=None, mode="coupled", phi=None,
                         nic=True, return_result=False, cl=False, replay=False, log=False, kappa=None):
    """One policy on one replicate under the cross-layer model; metrics as run_replicate, plus the
    CL3 attribution shares when the NIC layer is on. cl adds the layer_metrics fields; replay (coupled
    mode only) adds the CL4 counterfactual: the same moves with the same start and finish steps and the
    migration bytes removed from the links (A39)."""
    if cl and not nic:
        raise ValueError("the CL fields need the NIC layer")
    if replay and (mode != "coupled" or not nic):
        raise ValueError("a replay needs coupled mode with the NIC layer")
    m = cfg["migration_steps_m0"] if m is None else m
    phi = cfg["phi"] if phi is None else phi
    ctx = {"m": m, "d": cfg["monitor_delay_d"], "horizons": cfg["horizons"], "load": rep.load,
           "margin_quantile": cfg["margin_quantile"], "residual_window": cfg["residual_window"],
           "gamma": cfg["gamma"], "a_min": cfg["a_min"], "a_max": cfg["a_max"]}
    policy = build_policy(name, params, rep.mu, ctx)
    if log:
        policy = LoggedPolicy(policy)
    window = np.array(cross.nbytes[rep.start: rep.start + rep.load.shape[0]], dtype=np.float64)
    shared = dict(m=m, kappa=cfg["migration_overhead_kappa"] if kappa is None else kappa, budget=cfg["migration_budget_B"],
                  d=cfg["monitor_delay_d"], k=cfg["decision_interval_k"], rho=cfg["rho"], phi=phi,
                  cooldown=params.get("cooldown", 0), nic=nic)
    res = simulate_layers(rep.load, window, cross.rel, rep.node0, rep.mu, cross.link, cross.sizes,
                          mode=mode, policy=policy, forecast=forecast, **shared)
    met = compute_metrics(res, rep.load, slo=cfg["slo_threshold"], wasted_window=cfg["wasted_move_window"],
                          wait_quantile=cfg["wait_quantile"])
    if nic:
        attr = attribution(res, cfg["slo_threshold"])
        met = dict(met, attr_violating_steps=attr["n"], attr_cpu_share=attr["cpu"], attr_net_share=attr["net"])
    if cl:
        met = dict(met, **layer_metrics(res, cfg["slo_threshold"]))
    if replay:
        finish = {(t0, p): tf for t0, p, _s, _d, tf in res.moves}
        res2 = simulate_layers(rep.load, window, cross.rel, rep.node0, rep.mu, cross.link, cross.sizes,
                               mode="const_m", policy=ReplayPolicy(res.moves), forecast=None,
                               replay_finish=finish, **shared)
        met2 = compute_metrics(res2, rep.load, slo=cfg["slo_threshold"], wasted_window=cfg["wasted_move_window"],
                               wait_quantile=cfg["wait_quantile"])
        same = res2.moves == res.moves and bool(np.array_equal(res2.node_hist, res.node_hist))
        met = dict(met, cl_replay_ok=same, cl_damage_replay=met2["damage"],
                   cl_wait_mean_replay=weighted_mean(res2.wait, res2.fg))
    if log:
        met = dict(met, **policy.fields())
        if name in ("P4", "P4a"):
            met = dict(met, **rv4_fields(rep.load, forecast, ctx, cfg["margin_quantile"]))
    return (met, res) if return_result else met
