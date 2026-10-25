"""Cross-layer replicate runner (Phase 9, A30).

Builds the per-dataset cross-layer inputs of amendment A24 and runs one policy on one replicate
with simulate_layers. Every parameter comes from the config; none is defined here.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.layers import attribution, link_capacity, partition_sizes, relative_demand, simulate_layers
from src.metrics import compute_metrics
from src.policies import build_policy
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


def run_replicate_layers(cfg, rep, cross, name, params, m=None, forecast=None, mode="coupled", phi=None,
                         nic=True, return_result=False):
    """One policy on one replicate under the cross-layer model; metrics as run_replicate, plus the
    CL3 attribution shares when the NIC layer is on."""
    m = cfg["migration_steps_m0"] if m is None else m
    phi = cfg["phi"] if phi is None else phi
    ctx = {"m": m, "d": cfg["monitor_delay_d"], "horizons": cfg["horizons"], "load": rep.load,
           "margin_quantile": cfg["margin_quantile"], "residual_window": cfg["residual_window"],
           "gamma": cfg["gamma"], "a_min": cfg["a_min"], "a_max": cfg["a_max"]}
    policy = build_policy(name, params, rep.mu, ctx)
    window = np.array(cross.nbytes[rep.start: rep.start + rep.load.shape[0]], dtype=np.float64)
    res = simulate_layers(rep.load, window, cross.rel, rep.node0, rep.mu, cross.link, cross.sizes,
                          m=m, kappa=cfg["migration_overhead_kappa"], budget=cfg["migration_budget_B"],
                          d=cfg["monitor_delay_d"], k=cfg["decision_interval_k"], rho=cfg["rho"], phi=phi,
                          mode=mode, policy=policy, cooldown=params.get("cooldown", 0), forecast=forecast,
                          nic=nic)
    met = compute_metrics(res, rep.load, slo=cfg["slo_threshold"], wasted_window=cfg["wasted_move_window"],
                          wait_quantile=cfg["wait_quantile"])
    if nic:
        attr = attribution(res, cfg["slo_threshold"])
        met = dict(met, attr_violating_steps=attr["n"], attr_cpu_share=attr["cpu"], attr_net_share=attr["net"])
    return (met, res) if return_result else met
