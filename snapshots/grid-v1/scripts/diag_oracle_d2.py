"""Exploratory diagnostic (tuning split only, never a result): why is the oracle worse on D2?

Usage: python scripts/diag_oracle_d2.py --config configs/default.yaml --tuned configs/tuned.yaml --dataset D2 --hysteresis H1 H2 ... --windows W1 W2 ... --out logs/diag_oracle_d2.csv
Runs P1, P2 and P5 at tuned parameters, P5 with other hysteresis values (tuned cooldown) and P5 with a
window-mean target (tuned hysteresis and cooldown) on every eligible tune-split event of one dataset.
Reports means, medians, moves, wasted moves and the in-event damage share, for all events and for the
events not used in tuning (the tuning events are the first tune_replicates of the seeded selection),
plus paired comparisons of every P5 variant against P2 and P1.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.policies import build_policy  # noqa: E402
from src.replicates import eligible_events, load_dataset, make_replicate, select_events  # noqa: E402
from src.seeds import derive_seed  # noqa: E402
from diag_oracle import WindowOracle, run  # noqa: E402
from diag_oracle_paired import paired_summary  # noqa: E402

REFERENCES = ("P2", "P1")


def report(df, labels, name, seed):
    print("\n-- %s (n=%d) --" % (name, len(df)))
    print("%-9s %12s %12s %8s %8s %10s" % ("policy", "mean damage", "median", "moves", "wasted", "in-event"))
    for lab in labels:
        print("%-9s %12.6g %12.6g %8.1f %8.1f %10.3f" % (
            lab, df[lab].mean(), df[lab].median(), df[lab + "_moves"].mean(),
            df[lab + "_wasted"].mean(), df[lab + "_inside"].mean()))
    for ref in REFERENCES:
        for lab in labels:
            if lab in REFERENCES:
                continue
            s = paired_summary(df[lab], df[ref], seed)
            print("%s - %s: wins %d losses %d | mean diff %.4g median diff %.4g | Wilcoxon p %.4g | CI [%.4g, %.4g]" % (
                lab, ref, s["wins"], s["losses"], s["mean_diff"], s["median_diff"],
                s["wilcoxon_p"], s["ci_low"], s["ci_high"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tuned", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--hysteresis", nargs="*", type=float, default=[])
    parser.add_argument("--windows", nargs="*", type=int, default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    tuned = yaml.safe_load(Path(args.tuned).read_text(encoding="utf-8"))
    m, d = cfg["migration_steps_m0"], cfg["monitor_delay_d"]
    data = load_dataset(cfg, args.dataset)
    t = tuned["tuned"][args.dataset]
    variants = [("P1", "P1", t["P1"]), ("P2", "P2", t["P2"]), ("P5", "P5", t["P5"])]
    for h in args.hysteresis:
        variants.append(("P5_h%g" % h, "P5", dict(t["P5"], hysteresis=h)))
    for w in args.windows:
        variants.append(("P5_w%d" % w, "W", dict(t["P5"], w=w)))
    labels = [v[0] for v in variants]
    n_events = len(eligible_events(cfg, data, "tune"))
    events = select_events(cfg, data, "tune", n_events)
    print("%s: %d eligible tune events; first %d were the tuning events" % (
        args.dataset, n_events, cfg["tune_replicates"]), flush=True)
    rows = []
    for idx, ev in enumerate(events):
        rep = make_replicate(cfg, data, ev, 0, cfg["headroom"])
        row = {"event_id": rep.event_id, "onset_class": rep.onset_class,
               "tuning_event": idx < cfg["tune_replicates"]}
        for label, kind, params in variants:
            ctx = {"m": m, "d": d, "horizons": cfg["horizons"], "margin_quantile": cfg["margin_quantile"],
                   "residual_window": cfg["residual_window"], "load": rep.load}
            pol = (WindowOracle(params["hysteresis"], m, rep.load, params["w"]) if kind == "W"
                   else build_policy(kind, params, rep.mu, ctx))
            met, inside = run(cfg, rep, pol, params.get("cooldown", 0))
            row[label] = float(met["damage"])
            row[label + "_moves"] = float(met["moves"])
            row[label + "_wasted"] = float(met["wasted_moves"])
            row[label + "_inside"] = inside
        rows.append(row)
        print("  done %d / %d" % (len(rows), n_events), flush=True)
    df = pd.DataFrame(rows)
    seed = derive_seed("diag_d2", cfg["master_seed"], args.dataset)
    report(df, labels, "all events", seed)
    report(df[~df["tuning_event"]], labels, "events not used in tuning", seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print("\nper-event table written to " + args.out)


if __name__ == "__main__":
    main()
