"""Phase 3 driver. Stages: load, events, inject."""
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src import detect, injector, loaders
from src.config import load_config, split_points, step_seconds


def as_json(obj):
    return obj.item() if hasattr(obj, "item") else str(obj)


def paths(cfg):
    root = Path(cfg["data_root"])
    return root / cfg["raw_dirname"], root / cfg["processed_dirname"], Path(cfg["logs_dirname"])


def record_amendment(cfg, ident, change, reason):
    path = Path(cfg["amendments_file"])
    data = path.read_bytes()
    if f"| {ident} |".encode("utf-8") in data:
        print(f"amendment {ident} already recorded; nothing written")
        return
    prefix = "" if data.endswith(b"\n") else "\n"
    stamp = datetime.now().isoformat(timespec="seconds")
    previous = hashlib.sha256(data).hexdigest()
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{prefix}| {ident} | {stamp} | {change} | {reason} | yes | {previous} |\n")
    print(f"amendment {ident} recorded")


def detector_args(cfg, load, dataset):
    train_end, tune_end = split_points(cfg, dataset, load.shape[0])
    floor = cfg["eps"] * load[:train_end].mean()
    return dict(dataset=dataset, r=cfg["r"], w_b=cfg["W_b"], s=cfg["s"], floor=floor,
                rise_lo=cfg["rise_lo"], rise_hi=cfg["rise_hi"], train_end=train_end, tune_end=tune_end)


def stage_load(cfg, dataset):
    raw, processed, logs = paths(cfg)
    processed.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    if dataset == "D1":
        folder = raw / cfg["azure_dirname"]
        result = loaders.load_d1(cfg, folder)
        memory = loaders.verify_memory_files(cfg, folder)
        result["stats"]["memory_file_check"] = memory
        bad = [name for name, info in memory.items() if not info["columns_match"]]
        if bad:
            raise SystemExit(f"application-memory schema mismatch: {bad}")
    else:
        source = raw / cfg["twitter_dirname"] / cfg["d2_file"]
        result = loaders.load_d2(cfg, source, step_seconds(cfg, dataset))
    np.save(processed / f"{dataset}_L.npy", result["L"])
    np.save(processed / f"{dataset}_Bytes.npy", result["Bytes"])
    np.save(processed / f"{dataset}_demand.npy", result["demand"])
    text = json.dumps(result["stats"], indent=2, default=as_json)
    (logs / f"{dataset}_load_stats.json").write_text(text, encoding="utf-8")
    print(f"{dataset} L shape {result['L'].shape}, sum check passed")
    print(text)


def stage_events(cfg, dataset, record):
    _, processed, logs = paths(cfg)
    load = np.load(processed / f"{dataset}_L.npy")
    args = detector_args(cfg, load, dataset)
    events = detect.detect_events(load, **args)
    if events.empty:
        raise SystemExit("no events detected")
    cut = detect.onset_cut(events)
    events = detect.assign_onset_class(events, cut)
    events.to_csv(processed / f"{dataset}_events.csv", index=False)
    n_eval = int((events["split"] == "eval").sum())
    overall = events["onset_class"].value_counts().to_dict()
    by_split = events.groupby(["split", "onset_class"]).size().unstack(fill_value=0)
    summary = {
        "dataset": dataset, "n_steps": int(load.shape[0]), "train_end": int(args["train_end"]),
        "tune_end": int(args["tune_end"]), "floor": float(args["floor"]),
        "n_events": int(len(events)),
        "events_per_split": {str(k): int(v) for k, v in events["split"].value_counts().items()},
        "class_counts_by_split": {str(s): {str(c): int(n) for c, n in row.items()} for s, row in by_split.iterrows()},
        "median_rise_time": cut, "min_events": cfg["min_events"],
        "eval_events_pass": n_eval >= cfg["min_events"],
        "both_classes_present": all(overall.get(name, 0) > 0 for name in ("ramped", "abrupt")),
    }
    text = json.dumps(summary, indent=2, default=as_json)
    (logs / f"{dataset}_event_summary.json").write_text(text, encoding="utf-8")
    print(text)
    if not summary["both_classes_present"]:
        print("WARNING: one onset class is empty (tie at the median). Stop and decide before any policy run.")
    if dataset == "D2":
        print(f"DATA_LOG line: D2 attempt {cfg['d2_file']}: detector events {len(events)}, eval events {n_eval}, median rise {cut}")
    if record:
        change = (f"{dataset} onset-class cut point = median rise time {cut} steps over {len(events)} detected events; "
                  f"ramped if rise >= cut, abrupt otherwise; r {cfg['r']}, W_b {cfg['W_b']}, s {cfg['s']}, eps {cfg['eps']}")
        record_amendment(cfg, f"onset-{dataset}", change, "Section 2 onset rule, computed once in Phase 3")


def stage_inject(cfg, dataset):
    _, processed, logs = paths(cfg)
    load = np.load(processed / f"{dataset}_L.npy")
    args = detector_args(cfg, load, dataset)
    out, table = injector.inject_spikes(
        load, dataset=dataset, master_seed=cfg["master_seed"], n_events=cfg["injection_events"],
        amp_min=cfg["injection_amplitude_min"], amp_max=cfg["injection_amplitude_max"],
        rise_min=cfg["injection_rise_min"], rise_max=cfg["injection_rise_max"],
        dur_min=cfg["injection_dur_min"], dur_max=cfg["injection_dur_max"],
        hot_min=cfg["injection_hot_min"], hot_max=cfg["injection_hot_max"],
        w_b=cfg["W_b"], floor=args["floor"], start=args["tune_end"], label=cfg["inject_label"])
    np.save(processed / f"{dataset}_D3_L.npy", out)
    table.to_csv(processed / f"{dataset}_D3_events.csv", index=False)
    found = detect.detect_events(out, **args)
    by_part = {int(p): g["onset"].to_numpy() for p, g in found.groupby("partition")}
    hits = 0
    for _, row in table.iterrows():
        onsets = by_part.get(int(row["partition"]), np.array([], dtype=np.int64))
        hits += int(np.any(np.abs(onsets - int(row["onset"])) <= cfg["inject_tolerance"]))
    summary = {
        "dataset": dataset, "spikes": int(table["event_id"].nunique()), "spike_partition_rows": int(len(table)),
        "recovered_within_tolerance": hits, "recovered_share": hits / len(table),
        "rise_steps_min": int(table["rise_steps"].min()), "rise_steps_max": int(table["rise_steps"].max()),
    }
    text = json.dumps(summary, indent=2, default=as_json)
    (logs / f"{dataset}_inject_summary.json").write_text(text, encoding="utf-8")
    print(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True, choices=["D1", "D2"])
    parser.add_argument("--stage", required=True, choices=["load", "events", "inject"])
    parser.add_argument("--record-amendment", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.stage == "load":
        stage_load(cfg, args.dataset)
    elif args.stage == "events":
        stage_events(cfg, args.dataset, args.record_amendment)
    else:
        stage_inject(cfg, args.dataset)


if __name__ == "__main__":
    main()