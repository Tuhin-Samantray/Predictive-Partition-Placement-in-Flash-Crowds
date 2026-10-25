"""Phase 6: forecast tables (with forecaster tuning) and the accuracy / early-warning tables.

Usage:
  python scripts/02_forecast.py --config configs/default.yaml --dataset D1 --forecaster all
  python scripts/02_forecast.py --config configs/default.yaml --stage accuracy --out tables/forecast_accuracy.csv
--forecaster takes all, one name or a comma-separated list. Timing goes to logs/timing.csv.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.accuracy import accuracy_rows, early_warning_rows, tune_forecaster  # noqa: E402
from src.config import load_config  # noqa: E402
from src.forecasters import LEARNED, build_table, day_steps  # noqa: E402
from src.replicates import load_dataset  # noqa: E402

DATASETS = ("D1", "D2")


def table_path(cfg, dataset, kind, suffix=".npy"):
    base = Path(cfg["data_root"]) / cfg["processed_dirname"] / cfg["forecast_dirname"]
    return base / (dataset + "_" + kind + suffix)


def log_timing(cfg, rows):
    log = Path(cfg["logs_dirname"]) / "timing.csv"
    log.parent.mkdir(parents=True, exist_ok=True)
    new_log = not log.exists()
    with open(log, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new_log:
            writer.writerow(["stage", "policy", "dataset", "iter", "runs", "seconds"])
        writer.writerows(rows)


def save_atomic(path, array):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.save(handle, array)
    os.replace(tmp, path)


def make_tables(cfg, dataset, kinds, resume):
    data = load_dataset(cfg, dataset)
    day = day_steps(cfg, dataset)
    expected = (data.n_steps, len(cfg["horizons"]), data.load.shape[1])
    table_path(cfg, dataset, "x").parent.mkdir(parents=True, exist_ok=True)
    for kind in kinds:
        path = table_path(cfg, dataset, kind)
        if resume and path.exists():
            print(dataset, kind, "exists, skipped")
            continue
        start = time.perf_counter()
        hp = None
        if kind in LEARNED:
            results, best = tune_forecaster(cfg, dataset, kind, data.load, data.train_end,
                                            data.tune_end, day, cfg["workers"])
            hp = results[best]["hp"]
            report = {"dataset": dataset, "forecaster": kind, "best_hp": hp, "best_index": best,
                      "n_configs": len(results), "grid": results}
            table_path(cfg, dataset, kind, "_tuning.json").write_text(
                json.dumps(report, sort_keys=True, indent=1) + "\n", encoding="utf-8", newline="\n")
            print(dataset, kind, "tuned: best", hp, "objective", results[best]["objective"], flush=True)
        table = build_table(cfg, dataset, kind, data.load, data.train_end, day, hp=hp, n_jobs=cfg["workers"])
        if table.shape != expected or table.dtype != np.float32:
            raise SystemExit("bad table %s %s: %s %s, expected %s" % (dataset, kind, table.shape, table.dtype, expected))
        save_atomic(path, table)
        seconds = time.perf_counter() - start
        log_timing(cfg, [("forecast", kind, dataset, 0, 1, seconds)])
        print(dataset, kind, "table", table.shape, "seconds %.1f" % seconds, flush=True)


def run_accuracy(cfg, datasets, kinds, out):
    acc, warn = [], []
    for dataset in datasets:
        data = load_dataset(cfg, dataset)
        expected = (data.n_steps, len(cfg["horizons"]), data.load.shape[1])
        for kind in kinds:
            path = table_path(cfg, dataset, kind)
            if not path.exists():
                raise SystemExit("missing forecast table: " + str(path))
            table = np.load(path, mmap_mode="r")
            if table.shape != expected or table.dtype != np.float32:
                raise SystemExit("bad table %s: %s %s, expected %s" % (path, table.shape, table.dtype, expected))
            acc += accuracy_rows(cfg, dataset, kind, table, data.load, data.events, data.train_end, data.tune_end)
            warn += early_warning_rows(cfg, dataset, kind, table, data.load, data.events, data.train_end, data.tune_end)
            print(dataset, kind, "shape", table.shape, "ok", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    acc_df, warn_df = pd.DataFrame(acc), pd.DataFrame(warn)
    acc_df.to_csv(out, index=False, lineterminator="\n")
    warn_path = out.with_name(out.stem + "_early_warning" + out.suffix)
    warn_df.to_csv(warn_path, index=False, lineterminator="\n")
    ev = acc_df[acc_df["split"] == "eval"]
    print(ev.groupby(["dataset", "forecaster"])[["mase_all", "smape_all", "mase_event", "smape_event"]].mean().to_string())
    top = warn_df[(warn_df["split"] == "eval") & (warn_df["onset_class"] == "all")]
    print(top[["dataset", "forecaster", "n_events", "recall", "lead_median", "precision"]].to_string(index=False))
    print("wrote", out, "and", warn_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--forecaster", default="all")
    parser.add_argument("--stage", choices=["tables", "accuracy"], default="tables")
    parser.add_argument("--out")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    kinds = list(cfg["forecaster_names"]) if args.forecaster == "all" else args.forecaster.split(",")
    unknown = [k for k in kinds if k not in cfg["forecaster_names"]]
    if unknown:
        parser.error("unknown forecaster: " + ",".join(unknown))
    if args.stage == "tables":
        if not args.dataset:
            parser.error("--dataset is required for the tables stage")
        make_tables(cfg, args.dataset, kinds, args.resume)
    else:
        if not args.out:
            parser.error("--out is required for the accuracy stage")
        datasets = (args.dataset,) if args.dataset else DATASETS
        run_accuracy(cfg, datasets, kinds, Path(args.out))


if __name__ == "__main__":
    main()