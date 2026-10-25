"""Tuner for P1 to P5 on the tuning split (seeded random search, exhaustive for small grids).

Single policy:  python scripts/03_tune.py --config configs/default.yaml --policy P1 --split tune --out results/tune/P1.jsonl
All policies:   python scripts/03_tune.py --config configs/default.yaml --policy all --split tune --out configs/tuned.yaml --results-dir results/tune --choice results/tune/forecaster_choice.json
`all` reuses the files of the policies named by --reuse (default the Phase 5 P1 and P2) when present and writes one jsonl per policy plus the
generated tuned.yaml. P3 is tuned for every forecaster, P4 for the forecaster chosen in A13.
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
from src.replicates import load_dataset, load_forecast, tune_replicates  # noqa: E402
from src.tuned_config import render_tuned  # noqa: E402
from src.tuning import ALL_TUNABLE, tune_policy  # noqa: E402

DATASETS = ("D1", "D2")
ORDER = ("P1", "P2", "P3", "P4", "P5")


def jobs(cfg, name, choice, dataset):
    """(forecaster, exhaustive) jobs of one policy on one dataset."""
    if name in TUNABLE:
        return [(None, False)]
    if name == "P3":
        return [(fc, True) for fc in cfg["forecaster_names"]]
    if name in ("P4", "P4a"):
        return [(choice[dataset]["choice"], True)]
    return [(None, True)]


def read_summaries(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    return [r for r in rows if r["type"] == "summary"]


def tune_name(cfg, name, choice, cache):
    lines, summaries, timing_rows = [], [], []
    for dataset in DATASETS:
        if dataset not in cache:
            data = load_dataset(cfg, dataset)
            cache[dataset] = tune_replicates(cfg, data)
        reps = cache[dataset]
        for fc, exhaustive in jobs(cfg, name, choice, dataset):
            table = load_forecast(cfg, dataset, fc) if fc is not None else None
            rows, summary, timing = tune_policy(cfg, name, dataset, reps, table=table,
                                                exhaustive=exhaustive, forecaster=fc)
            lines += [json.dumps(r, sort_keys=True) for r in rows] + [json.dumps(summary, sort_keys=True)]
            summaries.append(summary)
            label = name if fc is None else name + ":" + fc
            timing_rows += [("tune", label, dataset, r["iter"], len(reps), sec) for r, sec in zip(rows, timing)]
    return lines, summaries, timing_rows


def report(summaries):
    for s in summaries:
        label = s["policy"] + ("" if s.get("forecaster") is None else ":" + s["forecaster"])
        status = "EXHAUSTIVE" if s.get("exhaustive") else ("PLATEAU OK" if s["converged"] else "PLATEAU FAILED")
        print(f'{s["dataset"]} {label}: best {json.dumps(s["best_params"], sort_keys=True)} '
              f'mean damage {s["best_mean_damage"]:.6g} {status}; configs with damage > 0: '
              f'{s["n_positive_damage"]} of {s["n_configs"]}; replicates {s["n_replicates"]}')
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
    parser.add_argument("--policy", required=True, choices=sorted(ALL_TUNABLE) + ["all"])
    parser.add_argument("--split", required=True, choices=["tune"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--results-dir")
    parser.add_argument("--choice")
    parser.add_argument("--reuse", nargs="*", default=sorted(TUNABLE),
                        help="policies whose existing jsonl is reused in all mode (default P1 P2)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    choice = json.loads(Path(args.choice).read_text(encoding="utf-8")) if args.choice else None
    if args.policy in ("P4", "P4a", "all") and choice is None:
        parser.error("--choice is required for P4 and all")
    cache = {}

    if args.policy != "all":
        lines, summaries, timing_rows = tune_name(cfg, args.policy, choice, cache)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        append_timing(cfg, timing_rows)
        report(summaries)
        return

    if not args.results_dir:
        parser.error("--results-dir is required for all")
    results = Path(args.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    for name in ORDER:
        path = results / (name + ".jsonl")
        if name in args.reuse and path.exists():
            summaries = read_summaries(path)
            print("reused", path)
        else:
            lines, summaries, timing_rows = tune_name(cfg, name, choice, cache)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
            append_timing(cfg, timing_rows)
        report(summaries)
        all_summaries += summaries
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_tuned(all_summaries, DATASETS), encoding="utf-8", newline="\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
