"""Exploratory diagnostic (tuning split only, never a result): is the oracle's D2 shortfall real?

Usage: python scripts/diag_oracle_paired.py --config configs/default.yaml --tuned configs/tuned.yaml --n-events N --out logs/diag_oracle_paired.csv
Runs P1, P2, P3 (ewma) and P5 at their tuned parameters on up to N eligible tune-split events per
dataset (one replicate each, same seeded selection as tuning) and compares P5 with each other policy
event by event: wins, losses, ties, mean and median damage difference, Wilcoxon signed-rank p-value,
bootstrap confidence interval of the mean difference, and the mean difference by onset class.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.replicates import (eligible_events, forecast_window, load_dataset, load_forecast,  # noqa: E402
                            make_replicate, run_replicate, select_events)
from src.seeds import derive_seed  # noqa: E402

DATASETS = ("D1", "D2")
ORACLE = "P5"


def paired_summary(oracle, other, seed):
    diff = np.asarray(oracle, dtype=np.float64) - np.asarray(other, dtype=np.float64)
    out = {"wins": int((diff < 0).sum()), "losses": int((diff > 0).sum()), "ties": int((diff == 0).sum()),
           "mean_diff": float(diff.mean()), "median_diff": float(np.median(diff))}
    try:
        out["wilcoxon_p"] = float(stats.wilcoxon(diff).pvalue)
    except ValueError:
        out["wilcoxon_p"] = float("nan")
    try:
        ci = stats.bootstrap((diff,), np.mean, random_state=np.random.default_rng(seed)).confidence_interval
        out["ci_low"], out["ci_high"] = float(ci.low), float(ci.high)
    except ValueError:
        out["ci_low"] = out["ci_high"] = float("nan")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tuned", required=True)
    parser.add_argument("--n-events", required=True, type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    tuned = yaml.safe_load(Path(args.tuned).read_text(encoding="utf-8"))
    frames = []
    for dataset in DATASETS:
        data = load_dataset(cfg, dataset)
        t = tuned["tuned"][dataset]
        variants = [("P1", "P1", t["P1"], None), ("P2", "P2", t["P2"], None),
                    ("P3_ewma", "P3", t["P3_ewma"], "ewma"), (ORACLE, ORACLE, t[ORACLE], None)]
        tables = {label: (None if fc is None else load_forecast(cfg, dataset, fc)) for label, _, _, fc in variants}
        n_eligible = len(eligible_events(cfg, data, "tune"))
        n = min(args.n_events, n_eligible)
        events = select_events(cfg, data, "tune", n)
        print("\n== %s: %d eligible tune events, using %d ==" % (dataset, n_eligible, n), flush=True)
        rows = []
        for ev in events:
            rep = make_replicate(cfg, data, ev, 0, cfg["headroom"])
            row = {"dataset": dataset, "event_id": rep.event_id, "onset_class": rep.onset_class,
                   "event_frac": (rep.end - rep.onset + 1) / rep.load.shape[0]}
            for label, name, params, _ in variants:
                table = tables[label]
                met = run_replicate(cfg, rep, name, params,
                                    forecast=None if table is None else forecast_window(table, rep))
                row[label] = float(met["damage"])
                row[label + "_moves"] = float(met["moves"])
            rows.append(row)
            print("  done %d / %d" % (len(rows), n), flush=True)
        df = pd.DataFrame(rows)
        frames.append(df)
        print("mean event share of window: %.4f" % df["event_frac"].mean())
        print("mean damage: " + "  ".join("%s %.6g" % (label, df[label].mean()) for label, _, _, _ in variants))
        seed = derive_seed("diag_paired", cfg["master_seed"], dataset)
        for label, _, _, _ in variants:
            if label == ORACLE:
                continue
            s = paired_summary(df[ORACLE], df[label], seed)
            print("%s - %s: wins %d losses %d ties %d | mean diff %.4g median diff %.4g | Wilcoxon p %.4g | bootstrap CI [%.4g, %.4g]" % (
                ORACLE, label, s["wins"], s["losses"], s["ties"], s["mean_diff"], s["median_diff"],
                s["wilcoxon_p"], s["ci_low"], s["ci_high"]))
            for cls, grp in df.groupby("onset_class"):
                print("    class %-10s n %3d mean diff %.4g" % (cls, len(grp), (grp[ORACLE] - grp[label]).mean()))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(args.out, index=False)
    print("\nper-event table written to " + args.out)


if __name__ == "__main__":
    main()
