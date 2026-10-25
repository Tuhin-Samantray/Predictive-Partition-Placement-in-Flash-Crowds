"""Exploratory diagnostic (tuning split only, never a result): where in time does the damage fall?

Usage: python scripts/diag_oracle_trace.py --config configs/default.yaml --tuned configs/tuned.yaml --dataset D2 --events D2-72 D2-60 --window W
For each named tune-split event, runs P2, P5 and a window-mean P5 (window W) and prints the share of
damage before the event onset, inside onset..end and after the end, the peak per-step damage and the
number of steps with damage.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.policies import build_policy  # noqa: E402
from src.replicates import eligible_events, load_dataset, make_replicate, select_events  # noqa: E402
from src.simulator import simulate  # noqa: E402
from diag_oracle import WindowOracle  # noqa: E402


def per_step_damage(cfg, rep, policy, cooldown):
    res = simulate(rep.load, rep.node0, rep.mu, m=cfg["migration_steps_m0"],
                   kappa=cfg["migration_overhead_kappa"], budget=cfg["migration_budget_B"],
                   d=cfg["monitor_delay_d"], k=cfg["decision_interval_k"], policy=policy, cooldown=cooldown)
    return (np.maximum(0, res.wait - cfg["slo_threshold"]) * res.fg).sum(axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tuned", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--events", nargs="+", required=True)
    parser.add_argument("--window", required=True, type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    tuned = yaml.safe_load(Path(args.tuned).read_text(encoding="utf-8"))
    m, d = cfg["migration_steps_m0"], cfg["monitor_delay_d"]
    data = load_dataset(cfg, args.dataset)
    t = tuned["tuned"][args.dataset]
    chosen = {str(ev["event_id"]): ev
              for ev in select_events(cfg, data, "tune", len(eligible_events(cfg, data, "tune")))}
    for event_id in args.events:
        rep = make_replicate(cfg, data, chosen[event_id], 0, cfg["headroom"])
        lo, hi = rep.onset - rep.start, rep.end - rep.start
        print("\n== %s (%s), onset step %d, end step %d of %d ==" % (
            event_id, rep.onset_class, lo, hi, rep.load.shape[0]))
        print("%-8s %12s %8s %8s %8s %12s %12s %12s" % (
            "policy", "damage", "before", "inside", "after", "peak/step", "damage steps", "peak step"))
        ctx = {"m": m, "d": d, "horizons": cfg["horizons"], "margin_quantile": cfg["margin_quantile"],
               "residual_window": cfg["residual_window"], "load": rep.load}
        for label, kind, params in (("P2", "P2", t["P2"]), ("P5", "P5", t["P5"]),
                                    ("P5_w%d" % args.window, "W", dict(t["P5"], w=args.window))):
            pol = (WindowOracle(params["hysteresis"], m, rep.load, params["w"]) if kind == "W"
                   else build_policy(kind, params, rep.mu, ctx))
            ps = per_step_damage(cfg, rep, pol, params.get("cooldown", 0))
            tot = ps.sum()
            share = (lambda x: x / tot if tot > 0 else 0)
            print("%-8s %12.6g %8.3f %8.3f %8.3f %12.6g %12d %12d" % (
                label, tot, share(ps[:lo].sum()), share(ps[lo:hi + 1].sum()), share(ps[hi + 1:].sum()),
                ps.max(), int((ps > 0).sum()), int(ps.argmax()) - lo))


if __name__ == "__main__":
    main()
