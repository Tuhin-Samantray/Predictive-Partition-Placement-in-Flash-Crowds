"""Layers-on tuner for P1, P3, P5 and P6 on the tuning split (A32).

python scripts/03_tune_layers.py --config configs/default.yaml --policy all --split tune --results-dir results/tune_layers --choice results/tune/forecaster_choice.json
P3 and P6 use the forecaster chosen for P4 (A13). Conditions: coupled mode, config h_net and phi, default m0 and headroom (A30).
Every number comes from the config. Timing goes to logs/timing.csv, never into result rows.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.layer_runs import load_cross  # noqa: E402
from src.layer_tuning import LAYER_POLICIES, tune_policy_layers  # noqa: E402
from src.replicates import load_dataset, load_forecast, tune_replicates  # noqa: E402

DATASETS = ("D1", "D2")


def jobs(name, choice, dataset):
    """(forecaster, exhaustive) of one policy on one dataset."""
    if name == "P1":
        return (None, False)
    if name in ("P3", "P6"):
        return (choice[dataset]["choice"], True)
    return (None, True)


def report(summaries):
    for s in summaries:
        label = s["policy"] + ("" if s.get("forecaster") is None else ":" + s["forecaster"])
        status = "EXHAUSTIVE" if s.get("exhaustive") else ("PLATEAU OK" if s["converged"] else "PLATEAU FAILED")
        print(f'{s["dataset"]} {label}: best {json.dumps(s["best_params"], sort_keys=True)} '
              f'mean damage {s["best_mean_damage"]:.6g} moves {s["best_mean_moves"]:.6g} {status}; '
              f'configs with damage > 0: {s["n_positive_damage"]} of {s["n_configs"]}; replicates {s["n_replicates"]}')
        if s["n_positive_damage"] == 0:
            print(f'{s["dataset"]} {label}: DEGENERATE, every configuration has zero damage')


def append_timing(cfg, timing_rows):
    log = Path(cfg["logs_dirname"]) / "timing.csv"
    log.parent.mkdir(parents=True, exist_ok=True)
    new_log = not log.exists()
    with open(log, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new_log:
            writer.writerow(["stage", "policy", "dataset", "iter", "runs", "seconds"])
        writer.writerows(timing_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--policy", required=True, choices=list(LAYER_POLICIES) + ["all"])
    parser.add_argument("--split", required=True, choices=["tune"])
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--choice", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    choice = json.loads(Path(args.choice).read_text(encoding="utf-8"))
    names = LAYER_POLICIES if args.policy == "all" else (args.policy,)
    results = Path(args.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    for name in names:
        if (results / (name + ".jsonl")).exists():
            parser.error("refusing to overwrite " + str(results / (name + ".jsonl")))
    cache = {}
    for dataset in DATASETS:
        data = load_dataset(cfg, dataset)
        cache[dataset] = (tune_replicates(cfg, data), load_cross(cfg, data))
    for name in names:
        lines, summaries, timing_rows = [], [], []
        for dataset in DATASETS:
            reps, cross = cache[dataset]
            fc, exhaustive = jobs(name, choice, dataset)
            table = load_forecast(cfg, dataset, fc) if fc is not None else None
            rows, summary, timing = tune_policy_layers(cfg, name, dataset, reps, cross, table=table,
                                                       exhaustive=exhaustive, forecaster=fc)
            lines += [json.dumps(r, sort_keys=True) for r in rows] + [json.dumps(summary, sort_keys=True)]
            summaries.append(summary)
            label = name if fc is None else name + ":" + fc
            timing_rows += [("tune_layers", label, dataset, r["iter"], len(reps), sec) for r, sec in zip(rows, timing)]
        (results / (name + ".jsonl")).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        append_timing(cfg, timing_rows)
        report(summaries)


if __name__ == "__main__":
    main()
