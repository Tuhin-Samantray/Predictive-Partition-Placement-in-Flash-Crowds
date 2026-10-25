"""Q10 scoring (A50, A51): Chronos against the five Phase 6 forecasters on the identical origins.

Runs in the main environment (no torch). Per dataset: verify the Chronos table (shape, dtype, finite, and the rows that
differ from the naive fill are exactly the evaluation origins), score every forecaster on the identical origins with the
subset functions of src/accuracy_subset.py, then check that the Phase 6 forecasters' subset MASE is close to their Phase 6
evaluation MASE. Outputs are written, set read-only and hashed only if every check passes.
Every parameter is an argument or a config key; none is defined in this module.
"""
import argparse
import csv
import hashlib
import os
import stat
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.accuracy_subset import eval_origins, subset_accuracy_rows, subset_early_warning_rows  # noqa: E402
from src.config import load_config  # noqa: E402
from src.replicates import load_dataset  # noqa: E402

HOUR = timedelta(hours=1).total_seconds()


def table_path(cfg, dataset, name):
    return Path(cfg["data_root"]) / cfg["processed_dirname"] / cfg["forecast_dirname"] / ("%s_%s.npy" % (dataset, name))


def verify_table(table, load, origins, chunk, n_horizons):
    """Problems found in a Chronos table, and the number of rows that differ from the naive fill."""
    n_steps, n_part = load.shape
    if table.shape != (n_steps, n_horizons, n_part):
        return ["shape %s, expected %s" % (table.shape, (n_steps, n_horizons, n_part))], 0
    problems = []
    if table.dtype != np.float32:
        problems.append("dtype %s, expected float32" % table.dtype)
    finite = True
    differ = np.zeros(n_steps, dtype=bool)
    for lo in range(0, n_steps, chunk):
        block = np.asarray(table[lo:lo + chunk])
        finite = finite and bool(np.isfinite(block).all())
        base = np.asarray(load[lo:lo + chunk], dtype=np.float32)[:, None, :]
        differ[lo:lo + chunk] = (block != base).any(axis=1).any(axis=1)
    expected = np.zeros(n_steps, dtype=bool)
    expected[origins] = True
    if not finite:
        problems.append("non-finite values")
    missing = int((expected & ~differ).sum())
    extra = int((differ & ~expected).sum())
    if missing:
        problems.append("%d evaluation origins hold the naive fill" % missing)
    if extra:
        problems.append("%d rows outside the evaluation origins differ from the naive fill" % extra)
    return problems, int(differ.sum())


def score(cfg, dataset, tables, load, events, train_end, tune_end, origins):
    acc, warn = [], []
    for name, table in tables.items():
        acc += subset_accuracy_rows(cfg, dataset, name, table, load, events, train_end, tune_end, origins)
        warn += subset_early_warning_rows(cfg, dataset, name, table, load, events, train_end, tune_end, origins)
    return acc, warn


def sanity(acc_rows, phase6, tol):
    """Largest relative difference of mase_all between the subset rows and the Phase 6 evaluation rows."""
    ref = {(str(r.dataset), str(r.forecaster), int(r.horizon)): float(r.mase_all)
           for r in phase6.itertuples() if r.split == "eval"}
    worst, count = 0.0, 0
    for r in acc_rows:
        key = (r["dataset"], r["forecaster"], int(r["horizon"]))
        if key in ref:
            worst = max(worst, abs(r["mase_all"] - ref[key]) / ref[key])
            count += 1
    return worst, count, bool(count > 0 and worst <= tol)


def file_sha256(path, chunk_bytes):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv_readonly(frame, path):
    path = Path(path)
    if path.exists():
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    frame.to_csv(path, index=False)
    os.chmod(path, stat.S_IREAD)


def chronos_seconds(path, stage):
    if not Path(path).exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return [(r["dataset"], float(r["seconds"])) for r in csv.DictReader(handle) if r["stage"] == stage]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--datasets", nargs="+", required=True, choices=["D1", "D2"])
    parser.add_argument("--label", required=True)
    parser.add_argument("--stride", required=True, type=int)
    parser.add_argument("--chunk", required=True, type=int)
    parser.add_argument("--sanity-tol", required=True, type=float)
    args = parser.parse_args()
    cfg = load_config(args.config)
    problems, acc_rows, warn_rows = [], [], []
    for ds in args.datasets:
        data = load_dataset(cfg, ds)
        load = np.asarray(data.load)
        origins = eval_origins(data.tune_end, load.shape[0], args.stride, cfg["horizons"])
        names = list(cfg["forecaster_names"]) + [args.label]
        tables = {name: np.load(table_path(cfg, ds, name), mmap_mode="r") for name in names}
        found, changed = verify_table(tables[args.label], load, origins, args.chunk, len(cfg["horizons"]))
        print("%s: %d evaluation origins (stride %d), Chronos rows differing from the naive fill %d, table shape %s" % (
            ds, origins.size, args.stride, changed, tables[args.label].shape), flush=True)
        problems += ["%s: %s" % (ds, p) for p in found]
        acc, warn = score(cfg, ds, tables, load, data.events, data.train_end, data.tune_end, origins)
        acc_rows += acc
        warn_rows += warn
    acc_frame = pd.DataFrame(acc_rows)
    warn_frame = pd.DataFrame(warn_rows)
    for col in ("mase_all", "mase_event"):
        print("\n%s on the identical origins:" % col)
        print(acc_frame.pivot_table(index=["dataset", "forecaster"], columns="horizon", values=col).to_string())
    print("\nearly warning (all onset classes):")
    show = ["dataset", "forecaster", "n_events", "recall", "lead_median", "precision"]
    print(warn_frame[warn_frame["onset_class"] == "all"][show].to_string(index=False))
    phase6 = pd.read_csv(Path("tables") / "forecast_accuracy.csv")
    worst, count, ok = sanity(acc_rows, phase6, args.sanity_tol)
    print("\nsanity: %d Phase 6 rows compared, largest relative MASE difference %.4g, tolerance %g -> %s" % (
        count, worst, args.sanity_tol, "ok" if ok else "FAILED"))
    if not ok:
        problems.append("sanity check against the Phase 6 evaluation MASE failed")
    if problems:
        for p in problems:
            print("PROBLEM: " + p)
        print("FAIL (nothing written)")
        return 1
    out = [(Path("tables") / "forecast_accuracy_chronos.csv", acc_frame),
           (Path("tables") / "forecast_accuracy_chronos_early_warning.csv", warn_frame)]
    chunk_bytes = cfg["compute"]["io_chunk_bytes"]
    lines = []
    for path, frame in out:
        write_csv_readonly(frame, path)
        lines.append("%s  %s" % (file_sha256(path, chunk_bytes), path.as_posix()))
    hash_path = Path(cfg["logs_dirname"]) / "forecast_accuracy_chronos.sha256"
    hash_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nwritten and set read-only:")
    for line in lines:
        print("  " + line)
    for ds, seconds in chronos_seconds(Path(cfg["logs_dirname"]) / "timing.csv", "chronos"):
        print("table build cost, %s: %.0f s (%.2f h wall clock)" % (ds, seconds, seconds / HOUR))
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
