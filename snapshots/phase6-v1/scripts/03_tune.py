"""Phase 5 tuner: seeded random search over the config grids on the tuning split.

Usage: python scripts/03_tune.py --config configs/default.yaml --policy P1 --split tune --out results/tune/P1.jsonl
Every number comes from the config. Timing goes to logs/timing.csv, never into result rows.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.policies import TUNABLE  # noqa: E402
from src.replicates import load_dataset, tune_replicates  # noqa: E402
from src.tuning import tune_policy  # noqa: E402

DATASETS = ("D1", "D2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--policy", required=True, choices=sorted(TUNABLE))
    parser.add_argument("--split", required=True, choices=["tune"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = Path(cfg["logs_dirname"]) / "timing.csv"
    log.parent.mkdir(parents=True, exist_ok=True)

    lines, summaries, timing_rows = [], [], []
    for dataset in DATASETS:
        data = load_dataset(cfg, dataset)
        reps = tune_replicates(cfg, data)
        rows, summary, timing = tune_policy(cfg, args.policy, dataset, reps)
        lines += [json.dumps(r, sort_keys=True) for r in rows] + [json.dumps(summary, sort_keys=True)]
        summaries.append(summary)
        timing_rows += [("tune", args.policy, dataset, r["iter"], len(reps), sec)
                        for r, sec in zip(rows, timing)]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    new_log = not log.exists()
    with open(log, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new_log:
            writer.writerow(["stage", "policy", "dataset", "iter", "runs", "seconds"])
        writer.writerows(timing_rows)

    for s in summaries:
        status = "PLATEAU OK" if s["converged"] else "PLATEAU FAILED"
        print(f'{s["dataset"]} {s["policy"]}: best {json.dumps(s["best_params"], sort_keys=True)} '
              f'mean damage {s["best_mean_damage"]:.6g} gain {s["plateau_gain"]:.4g} '
              f'tol {s["plateau_tol"]} {status}; configs with damage > 0: '
              f'{s["n_positive_damage"]} of {s["n_configs"]}; replicates {s["n_replicates"]}')
        if s["n_positive_damage"] == 0:
            print(f'{s["dataset"]} {s["policy"]}: DEGENERATE, every configuration has zero damage')


if __name__ == "__main__":
    main()