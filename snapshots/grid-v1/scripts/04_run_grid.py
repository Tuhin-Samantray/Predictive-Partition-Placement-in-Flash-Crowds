"""Grid runner (Section 8): one JSON line per (cell, replicate), resumable, multiprocessing over replicates.

Usage:
  python scripts/04_run_grid.py --config configs/default.yaml --tuned configs/tuned.yaml --queue must --dry-run
  python scripts/04_run_grid.py --config configs/default.yaml --tuned configs/tuned.yaml --queue must --pilot --out results/pilot
  python scripts/04_run_grid.py --config configs/default.yaml --tuned configs/tuned.yaml --queue must --out results
  python scripts/04_run_grid.py --config configs/default.yaml --tuned configs/tuned.yaml --queue must --only-row <cell_id>:<replicate_id> --out logs/spot
A cell is (dataset, onset class, m, headroom, policy variant). Per (dataset, onset class) a seeded permutation of the
eligible events gives replicates_per_cell distinct events (evaluation split; the pilot uses the tune split, classes
pooled, default m and headroom). The events do not depend on policy, m or headroom, so every variant sees identical
replicates. Rows hold no timing (G2); timing goes to logs/timing.csv. Resume: finished rows are skipped.
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.replicates import (eligible_events, forecast_window, load_dataset, load_forecast,  # noqa: E402
                            make_replicate, run_replicate, select_events)
from src.seeds import derive_seed  # noqa: E402

DATASETS = ("D1", "D2")
BASE_VARIANTS = ("P0", "P1", "P2", "P5")
HOUR = timedelta(hours=1).total_seconds()
CTX = {}


def variant_names(cfg):
    return BASE_VARIANTS + tuple("P3_" + fc for fc in cfg["forecaster_names"]) + ("P4",)


def cell_id(dataset, cls, m, headroom, variant):
    return "%s-%s-m%d-h%s-%s" % (dataset, cls, m, headroom, variant)


def row_key(cell, replicate):
    return "%s:%d" % (cell, replicate)


def plan(cfg, pilot):
    variants = list(variant_names(cfg))
    tasks = []
    for ds in DATASETS:
        data = load_dataset(cfg, ds)
        if pilot:
            split = "tune"
            groups = [("all", select_events(cfg, data, split, cfg["pilot_replicates"]))]
            levels = [(cfg["migration_steps_m0"], cfg["headroom"])]
        else:
            split = "eval"
            el = eligible_events(cfg, data, split)
            groups = []
            for cls in sorted(set(el["onset_class"])):
                sub = el[el["onset_class"] == cls].reset_index(drop=True)
                if len(sub) < cfg["replicates_per_cell"]:
                    raise ValueError("%s %s has %d eligible events, need %d" % (ds, cls, len(sub), cfg["replicates_per_cell"]))
                rng = np.random.default_rng(derive_seed("grid_events", cfg["master_seed"], ds, split, cls))
                order = rng.permutation(len(sub))[:cfg["replicates_per_cell"]]
                groups.append((cls, [sub.iloc[int(i)] for i in order]))
            levels = [(m, h) for m in cfg["m_levels"] for h in cfg["headroom_levels"]]
        for cls, events in groups:
            for j, ev in enumerate(events):
                for m, h in levels:
                    tasks.append({"dataset": ds, "split": split, "cls": cls, "j": j, "event_id": str(ev["event_id"]),
                                  "m": int(m), "headroom": float(h), "variants": list(variants),
                                  "stage": "pilot" if pilot else "grid"})
    return tasks


def plan_check(cfg):
    variants = list(variant_names(cfg))
    tasks = []
    for ds in DATASETS:
        data = load_dataset(cfg, ds)
        split = "tune"
        ordered = select_events(cfg, data, split, len(eligible_events(cfg, data, split)))
        start = cfg["tune_replicates"]
        events = ordered[start:start + cfg["oracle_check_events"]]
        if not events:
            start = 0
            events = ordered
        for i, ev in enumerate(events):
            tasks.append({"dataset": ds, "split": split, "cls": "all", "j": start + i, "event_id": str(ev["event_id"]),
                          "m": int(cfg["migration_steps_m0"]), "headroom": float(cfg["headroom"]),
                          "variants": list(variants), "stage": "oracle_check"})
    return tasks


def read_done(out):
    done, bad = set(), 0
    for path in sorted(Path(out).glob("grid_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            done.add(row_key(r["cell"], r["replicate"]))
    if bad:
        print("WARNING: %d unreadable lines skipped (their rows will be recomputed)" % bad)
    return done


def append_timing(cfg, timing_rows):
    log = Path(cfg["logs_dirname"]) / "timing.csv"
    log.parent.mkdir(parents=True, exist_ok=True)
    new_log = not log.exists()
    with open(log, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new_log:
            writer.writerow(["stage", "policy", "dataset", "iter", "runs", "seconds"])
        writer.writerows(timing_rows)


def _py(obj):
    return obj.item()


def init(cfg, tuned, out):
    CTX.update(cfg=cfg, tuned=tuned, out=out, data={}, tables={})


def dataset_of(name):
    cache = CTX["data"]
    if name not in cache:
        data = load_dataset(CTX["cfg"], name)
        cache[name] = (data, {str(r["event_id"]): r for _, r in data.events.iterrows()})
    return cache[name]


def table_of(name, forecaster):
    key = (name, forecaster)
    if key not in CTX["tables"]:
        CTX["tables"][key] = load_forecast(CTX["cfg"], name, forecaster)
    return CTX["tables"][key]


def resolve(variant, tuned, dataset):
    params = {} if variant == "P0" else dict(tuned["tuned"][dataset][variant])
    parts = variant.split("_", 1)
    name = parts[0]
    if name == "P3":
        forecaster = parts[1]
    elif name == "P4":
        forecaster = params["forecaster"]
    else:
        forecaster = None
    return name, params, forecaster


def run_task(task):
    cfg, tuned = CTX["cfg"], CTX["tuned"]
    data, by_id = dataset_of(task["dataset"])
    rep = make_replicate(cfg, data, by_id[task["event_id"]], task["j"], task["headroom"])
    rows, timing = [], []
    for variant in task["variants"]:
        name, params, forecaster = resolve(variant, tuned, task["dataset"])
        window = None
        if forecaster is not None:
            window = forecast_window(table_of(task["dataset"], forecaster), rep)
        t0 = time.perf_counter()
        met = run_replicate(cfg, rep, name, params, m=task["m"], forecast=window)
        seconds = time.perf_counter() - t0
        cell = cell_id(task["dataset"], task["cls"], task["m"], task["headroom"], variant)
        if not met["conservation_ok"]:
            raise RuntimeError("conservation violated in " + cell)
        row = dict(met, cell=cell, dataset=task["dataset"], onset_class=task["cls"], event_class=rep.onset_class,
                   m=task["m"], headroom=task["headroom"], variant=variant, replicate=task["j"], event_id=rep.event_id)
        rows.append(json.dumps(row, sort_keys=True, default=_py))
        timing.append((task["stage"], variant, task["dataset"], task["j"], 1, seconds))
    out = Path(CTX["out"])
    out.mkdir(parents=True, exist_ok=True)
    with open(out / ("grid_%d.jsonl" % os.getpid()), "a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(rows) + "\n")
    return timing


def dry_run(cfg, tasks):
    cells = {cell_id(t["dataset"], t["cls"], t["m"], t["headroom"], v) for t in tasks for v in t["variants"]}
    counts = {}
    for t in tasks:
        for v in t["variants"]:
            counts[(t["dataset"], v)] = counts.get((t["dataset"], v), 0) + 1
    runs = sum(counts.values())
    print("cells %d, runs %d (%.1f runs per cell), tasks %d" % (len(cells), runs, runs / len(cells), len(tasks)))
    log = Path(cfg["logs_dirname"]) / "timing.csv"
    secs = {}
    if log.exists():
        with open(log, newline="", encoding="utf-8") as handle:
            for r in csv.DictReader(handle):
                if r["stage"] == "pilot":
                    secs.setdefault((r["dataset"], r["policy"]), []).append(float(r["seconds"]))
    missing = [k for k in counts if k not in secs]
    if missing:
        print("no pilot timing for %d (dataset, variant) pairs; run --pilot first" % len(missing))
        return
    total = 0.0
    print("%-4s %-16s %7s %10s %13s" % ("data", "variant", "runs", "t_run s", "core-seconds"))
    for key in sorted(counts):
        t_run = float(np.mean(secs[key]))
        total += counts[key] * t_run
        print("%-4s %-16s %7d %10.4f %13.1f" % (key[0], key[1], counts[key], t_run, counts[key] * t_run))
    print("estimated core-hours %.3f (runs x t_run); wall-clock hours on %d workers %.3f; compute_budget_core_hours %s" % (
        total / HOUR, cfg["workers"], total / HOUR / cfg["workers"], cfg["compute_budget_core_hours"]))


def restrict(tasks, spec):
    cell, replicate = spec.rsplit(":", 1)
    keep = []
    for t in tasks:
        if t["j"] != int(replicate):
            continue
        for v in t["variants"]:
            if cell_id(t["dataset"], t["cls"], t["m"], t["headroom"], v) == cell:
                keep.append(dict(t, variants=[v]))
    return keep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tuned", required=True)
    parser.add_argument("--queue", required=True, choices=["must"])
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--oracle-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-row")
    parser.add_argument("--out")
    args = parser.parse_args()
    cfg = load_config(args.config)
    tuned = yaml.safe_load(Path(args.tuned).read_text(encoding="utf-8"))
    tasks = plan_check(cfg) if args.oracle_check else plan(cfg, args.pilot)
    if args.dry_run:
        dry_run(cfg, tasks)
        return
    if not args.out:
        parser.error("--out is required unless --dry-run")
    if args.only_row:
        tasks = restrict(tasks, args.only_row)
        if not tasks:
            parser.error("--only-row matches no row")
    done = read_done(args.out) if Path(args.out).exists() else set()
    todo = []
    for t in tasks:
        left = [v for v in t["variants"]
                if row_key(cell_id(t["dataset"], t["cls"], t["m"], t["headroom"], v), t["j"]) not in done]
        if left:
            todo.append(dict(t, variants=left))
    n_runs = sum(len(t["variants"]) for t in todo)
    print("%d runs in %d tasks to do, %d rows already present" % (n_runs, len(todo), len(done)), flush=True)
    if not todo:
        print("nothing to do")
        return
    step = max(1, len(todo) // cfg["cpu_cores"])
    if args.only_row:
        init(cfg, tuned, args.out)
        append_timing(cfg, run_task(todo[0]))
        print("done")
        return
    with mp.Pool(processes=cfg["workers"], initializer=init, initargs=(cfg, tuned, args.out)) as pool:
        for k, timing in enumerate(pool.imap_unordered(run_task, todo, chunksize=1), start=1):
            append_timing(cfg, timing)
            if k % step == 0 or k == len(todo):
                print("tasks done %d / %d" % (k, len(todo)), flush=True)
    print("finished")


if __name__ == "__main__":
    main()
