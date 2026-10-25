"""Exploratory diagnostic (tuning split only, never a result): why is the oracle not the best?

Usage: python scripts/diag_oracle.py --config configs/default.yaml --tuned configs/tuned.yaml --windows 1 5 10
Reruns the tuning replicates of both datasets with the tuned parameters for P0, the best
reactive policies, P3 with the lowest-damage forecaster, P5, and P5 variants whose target is the
mean true load over `w` steps starting at t + m (w from --windows). Prints per-replicate damage,
moves, wasted moves and the share of damage inside the event window.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402
from src.policies import build_policy, greedy_plan  # noqa: E402
from src.replicates import forecast_window, load_dataset, load_forecast, tune_replicates  # noqa: E402
from src.simulator import simulate  # noqa: E402

DATASETS = ("D1", "D2")


class WindowOracle:
    """Diagnostic oracle: target is the mean true load over w steps starting at t + m."""

    def __init__(self, hysteresis, m, load, w):
        self.hysteresis, self.m, self.w = hysteresis, m, w
        self.load = np.asarray(load, dtype=np.float64)

    def decide(self, t, observed, state):
        lo = min(t + self.m, self.load.shape[0] - 1)
        pred = self.load[lo:lo + self.w].mean(axis=0)
        node_of = state["node_of"]
        nl = np.bincount(node_of, weights=pred, minlength=state["n_nodes"])
        return greedy_plan(pred, nl, node_of, state["movable"], state["free_slots"],
                           self.hysteresis * nl.mean())


def run(cfg, rep, policy, cooldown, forecast=None):
    res = simulate(rep.load, rep.node0, rep.mu, m=cfg["migration_steps_m0"],
                   kappa=cfg["migration_overhead_kappa"], budget=cfg["migration_budget_B"],
                   d=cfg["monitor_delay_d"], k=cfg["decision_interval_k"], policy=policy,
                   cooldown=cooldown, forecast=forecast)
    met = compute_metrics(res, rep.load, slo=cfg["slo_threshold"],
                          wasted_window=cfg["wasted_move_window"], wait_quantile=cfg["wait_quantile"])
    per_step = (np.maximum(0, res.wait - cfg["slo_threshold"]) * res.fg).sum(axis=1)
    lo, hi = rep.onset - rep.start, rep.end - rep.start
    inside = float(per_step[lo:hi + 1].sum() / per_step.sum()) if per_step.sum() > 0 else 0.0
    return met, inside


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tuned", required=True)
    parser.add_argument("--windows", nargs="*", type=int, default=[])
    args = parser.parse_args()
    cfg = load_config(args.config)
    tuned = yaml.safe_load(Path(args.tuned).read_text(encoding="utf-8"))
    m, d = cfg["migration_steps_m0"], cfg["monitor_delay_d"]
    for dataset in DATASETS:
        data = load_dataset(cfg, dataset)
        reps = tune_replicates(cfg, data)
        t = tuned["tuned"][dataset]
        ctx = {"m": m, "d": d, "horizons": cfg["horizons"], "margin_quantile": cfg["margin_quantile"],
               "residual_window": cfg["residual_window"]}
        react = tuned["best_reactive"][dataset]
        variants = [("P0", "P0", {}, None), ("P1", "P1", t["P1"], None), ("P2", "P2", t["P2"], None),
                    ("P3_ewma", "P3", t["P3_ewma"], "ewma"), ("P5", "P5", t["P5"], None)]
        for w in args.windows:
            variants.append(("P5_w%d" % w, "W", dict(t["P5"], w=w), None))
        print("\n== %s  (best reactive %s, m %d, d %d, %d replicates) ==" % (dataset, react, m, d, len(reps)))
        print("%-9s %12s %8s %8s %12s   per-replicate damage" % ("policy", "mean damage", "moves", "wasted", "in-event share"))
        for label, name, params, fc in variants:
            table = None if fc is None else load_forecast(cfg, dataset, fc)
            dam, mov, was, ins = [], [], [], []
            for rep in reps:
                c = dict(ctx, load=rep.load)
                pol = (WindowOracle(params["hysteresis"], m, rep.load, params["w"]) if name == "W"
                       else build_policy(name, params, rep.mu, c))
                met, inside = run(cfg, rep, pol, params.get("cooldown", 0),
                                  None if table is None else forecast_window(table, rep))
                dam.append(met["damage"]); mov.append(met["moves"]); was.append(met["wasted_moves"]); ins.append(inside)
            print("%-9s %12.6g %8.1f %8.1f %12.3f   %s" % (
                label, np.mean(dam), np.mean(mov), np.mean(was), np.mean(ins),
                " ".join("%.3g" % x for x in dam)))


if __name__ == "__main__":
    main()
