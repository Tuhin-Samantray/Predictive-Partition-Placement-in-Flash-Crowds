"""Q10 (X5, A50, A51): zero-shot Chronos-Bolt forecasts, accuracy only. Runs in .venv_chronos (torch + chronos-forecasting).

Stages (every parameter is an argument; none is defined in this module; a forecast at origin o uses rows 0..o only):
  benchmark: timing on tune-split origins only (no accuracy)
  smoke: one evaluation origin per dataset; repeat and future-perturbation checks, no accuracy
  tables: forecast table F[o, h_index, p] (float32) at the evaluation origins tune_end, tune_end + stride, ...;
          other rows hold the naive forecast and are never scored (src/accuracy_subset.py); checkpoints and resumes
"""
import argparse
import csv
import math
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
from chronos import BaseChronosPipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, split_points  # noqa: E402

HOUR = timedelta(hours=1).total_seconds()


def load_series(cfg, dataset):
    base = Path(cfg["data_root"]) / cfg["processed_dirname"]
    load = np.load(base / (dataset + "_L.npy"), mmap_mode="r")
    train_end, tune_end = split_points(cfg, dataset, load.shape[0])
    return load, train_end, tune_end


def forecast_origin(pipeline, load, origin, context, horizons, q_index, batch, partitions=None):
    """Forecasts at the given horizons for every partition from the rows up to `origin`: (partitions, horizons)."""
    lo = max(0, origin - context + 1)
    series = torch.from_numpy(np.ascontiguousarray(np.asarray(load[lo:origin + 1], dtype=np.float32).T))
    if partitions is not None:
        series = series[:partitions]
    out = []
    with torch.inference_mode():
        for i in range(0, series.shape[0], batch):
            pred = pipeline.predict(series[i:i + batch], prediction_length=max(horizons))
            out.append(pred[:, q_index, :].float().numpy()[:, [h - 1 for h in horizons]])
    return np.concatenate(out)


def open_pipeline(args):
    pipeline = BaseChronosPipeline.from_pretrained(args.model, device_map="cpu", torch_dtype=torch.float32)
    levels = list(pipeline.model.config.chronos_config["quantiles"])
    return pipeline, levels.index(args.quantile)


def append_timing(cfg, rows):
    log = Path(cfg["logs_dirname"]) / "timing.csv"
    log.parent.mkdir(parents=True, exist_ok=True)
    new_log = not log.exists()
    with open(log, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new_log:
            writer.writerow(["stage", "policy", "dataset", "iter", "runs", "seconds"])
        writer.writerows(rows)


def benchmark(cfg, args, pipeline, q_index):
    horizons = cfg["horizons"]
    threads = torch.get_num_threads()
    print("model %s, torch threads %d, horizons %s" % (args.model, threads, horizons), flush=True)
    for dataset in args.datasets:
        load, train_end, tune_end = load_series(cfg, dataset)
        n_part = load.shape[1]
        n_eval = load.shape[0] - min(horizons) - tune_end
        print("%s: steps %d, partitions %d, train_end %d, tune_end %d, evaluation origins %d" % (
            dataset, load.shape[0], n_part, train_end, tune_end, n_eval), flush=True)
        origins = [train_end + i for i in range(args.n_origins)]
        for context in args.contexts:
            forecast_origin(pipeline, load, origins[0], context, horizons, q_index, args.batch, partitions=1)
            start = time.perf_counter()
            for origin in origins:
                forecast_origin(pipeline, load, origin, context, horizons, q_index, args.batch)
            per_series = (time.perf_counter() - start) / (len(origins) * n_part)
            line = "%s context %4d: %.5f s per series" % (dataset, context, per_series)
            for stride in args.strides:
                count = math.ceil(n_eval / stride)
                hours = count * n_part * per_series / HOUR
                line += " | stride %d: %d origins, %.2f h wall, %.1f core-h" % (stride, count, hours, hours * threads)
            print(line, flush=True)


def smoke(cfg, args, pipeline, q_index):
    horizons = cfg["horizons"]
    for dataset in args.datasets:
        load, train_end, tune_end = load_series(cfg, dataset)
        origin = tune_end + args.stride
        start = time.perf_counter()
        base = forecast_origin(pipeline, load, origin, args.context, horizons, q_index, args.batch)
        seconds = time.perf_counter() - start
        again = forecast_origin(pipeline, load, origin, args.context, horizons, q_index, args.batch)
        future = np.array(load, dtype=np.float32)
        future[origin + 1:] *= args.perturb
        perturbed = forecast_origin(pipeline, future, origin, args.context, horizons, q_index, args.batch)
        print("%s origin %d: shape %s, finite %s, min %.4g, max %.4g, mean forecast %.4g, mean last observed row %.4g, seconds %.2f" % (
            dataset, origin, base.shape, bool(np.isfinite(base).all()), base.min(), base.max(), base.mean(),
            float(np.asarray(load[origin]).mean()), seconds))
        print("  repeat call max abs difference %.3g; future perturbed by factor %g: max abs difference %.3g, identical %s" % (
            float(np.abs(base - again).max()), args.perturb, float(np.abs(base - perturbed).max()),
            bool(np.array_equal(base, perturbed))), flush=True)


def make_table(cfg, args, pipeline, q_index, dataset):
    horizons = cfg["horizons"]
    load, train_end, tune_end = load_series(cfg, dataset)
    n_steps, n_part = load.shape
    origins = np.arange(tune_end, n_steps - min(horizons), args.stride)
    if args.limit is not None:
        origins = origins[:args.limit]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / ("%s_%s.npy" % (dataset, args.label))
    part = final.with_name(final.name + ".partial")
    prog = final.with_name(final.name + ".progress")
    if final.exists():
        print(dataset, "table exists, skipped", flush=True)
        return
    shape = (n_steps, len(horizons), n_part)
    done = 0
    if part.exists() and prog.exists():
        table = np.lib.format.open_memmap(part, mode="r+")
        if table.shape != shape:
            raise SystemExit("partial table has shape %s, expected %s" % (table.shape, shape))
        done = int(prog.read_text(encoding="utf-8"))
        print("%s resuming after %d of %d origins" % (dataset, done, origins.size), flush=True)
    else:
        table = np.lib.format.open_memmap(part, mode="w+", dtype=np.float32, shape=shape)
        table[:] = np.repeat(np.asarray(load, dtype=np.float32)[:, None, :], len(horizons), axis=1)
        table.flush()
    start = time.perf_counter()
    first = done
    for i in range(done, origins.size):
        origin = int(origins[i])
        pred = np.maximum(forecast_origin(pipeline, load, origin, args.context, horizons, q_index, args.batch), 0)
        if not np.isfinite(pred).all():
            raise SystemExit("non-finite forecast at origin %d" % origin)
        table[origin] = pred.T.astype(np.float32)
        if (i + 1) % args.checkpoint_every == 0 or i + 1 == origins.size:
            table.flush()
            prog.write_text(str(i + 1), encoding="utf-8")
            elapsed = time.perf_counter() - start
            rate = elapsed / (i + 1 - first)
            print("%s origins %d / %d, elapsed %.0f s, eta %.0f s" % (dataset, i + 1, origins.size, elapsed, rate * (origins.size - i - 1)), flush=True)
    seconds = time.perf_counter() - start
    del table
    os.replace(part, final)
    prog.unlink()
    append_timing(cfg, [("chronos", args.label, dataset, 0, (origins.size - first) * n_part, seconds)])
    print("%s table written: %s, %d origins, seconds %.0f" % (dataset, final, origins.size, seconds), flush=True)


def need(parser, args, names):
    for name in names:
        if getattr(args, name) is None:
            parser.error("--%s is required for stage %s" % (name.replace("_", "-"), args.stage))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=["benchmark", "smoke", "tables"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--quantile", required=True, type=float)
    parser.add_argument("--datasets", nargs="+", required=True, choices=["D1", "D2"])
    parser.add_argument("--batch", required=True, type=int)
    parser.add_argument("--contexts", nargs="+", type=int)
    parser.add_argument("--n-origins", type=int)
    parser.add_argument("--strides", nargs="+", type=int)
    parser.add_argument("--context", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--perturb", type=float)
    parser.add_argument("--label")
    parser.add_argument("--out-dir")
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.stage == "benchmark":
        need(parser, args, ["contexts", "n_origins", "strides"])
    elif args.stage == "smoke":
        need(parser, args, ["context", "stride", "perturb"])
    else:
        need(parser, args, ["context", "stride", "label", "out_dir", "checkpoint_every"])
    pipeline, q_index = open_pipeline(args)
    if args.stage == "benchmark":
        benchmark(cfg, args, pipeline, q_index)
    elif args.stage == "smoke":
        smoke(cfg, args, pipeline, q_index)
    else:
        for dataset in args.datasets:
            make_table(cfg, args, pipeline, q_index, dataset)


if __name__ == "__main__":
    main()
