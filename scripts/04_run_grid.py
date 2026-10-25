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
from src.d3 import load_d3  # noqa: E402
from src.layer_runs import load_cross, run_replicate_layers  # noqa: E402
from src.replicates import (eligible_events, forecast_window, load_dataset, load_forecast,  # noqa: E402
                            make_replicate, make_replicate_skewed, run_replicate, select_events)
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


def eval_groups(cfg, data, ds, split):
    el = eligible_events(cfg, data, split)
    groups = []
    for cls in sorted(set(el["onset_class"])):
        sub = el[el["onset_class"] == cls].reset_index(drop=True)
        if len(sub) < cfg["replicates_per_cell"]:
            raise ValueError("%s %s has %d eligible events, need %d" % (ds, cls, len(sub), cfg["replicates_per_cell"]))
        rng = np.random.default_rng(derive_seed("grid_events", cfg["master_seed"], ds, split, cls))
        order = rng.permutation(len(sub))[:cfg["replicates_per_cell"]]
        groups.append((cls, [sub.iloc[int(i)] for i in order]))
    return groups


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
            groups = eval_groups(cfg, data, ds, split)
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


def init(cfg, tuned, out, ext=None, layers=None):
    CTX.update(cfg=cfg, tuned=tuned, ext=ext, layers=layers, out=out, data={}, tables={}, cross={})


def load_any(cfg, name):
    suffix = "_" + cfg["inject_label"]
    if name.endswith(suffix):
        return load_d3(cfg, name[: -len(suffix)])
    return load_dataset(cfg, name)


def dataset_of(name):
    cache = CTX["data"]
    if name not in cache:
        data = load_any(CTX["cfg"], name)
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


Q1_ALL = ("P1", "P2", "P3", "P4", "P4a", "P5")
Q1_HYST = ("P2", "P3", "P4", "P4a", "P5")
UNLIMITED = "unlimited"


def q1_label(base, factor, param, level):
    return "%s+%s:%s=%s" % (base, factor, param, level)


def q1_base_variants(tuned, ds):
    return ["P1", "P2", "P3_" + tuned["tuned"][ds]["P4"]["forecaster"], "P4", "P4a", "P5"]


def q1_specs(cfg, tuned, ext, ds):
    t = tuned["tuned"][ds]
    p3 = "P3_" + t["P4"]["forecaster"]

    def base_name(p):
        return p3 if p == "P3" else p

    def default(p):
        b = base_name(p)
        return dict(ext["tuned_ext"][ds]["P4a"]) if b == "P4a" else dict(t[b])

    out = []

    def add(base, factor, param, level, applied):
        out.append({"queue": "q1", "base": base, "factor": factor, "param": param, "level": level,
                    "applied": applied})

    for level in cfg["lead_levels"]:
        add(p3, "AA2", "lead", level, level)
    for p in Q1_HYST:
        for level in cfg["hysteresis_grid"]:
            if level != default(p)["hysteresis"]:
                add(base_name(p), "AA3", "hysteresis", level, level)
    for p in Q1_ALL:
        for level in cfg["cooldown_grid"]:
            if level != default(p)["cooldown"]:
                add(base_name(p), "AA4", "cooldown", level, level)
        for level in cfg["budget_levels"]:
            add(base_name(p), "AA5", "budget", level, cfg["n_partitions"] if level == UNLIMITED else level)
    for b in ("P4", "P4a"):
        for level in cfg["margin_levels"]:
            if b == "P4a" or level != cfg["margin_quantile"]:
                add(b, "AA6", "margin_quantile", level, level)
    for p in ("P1", "P2"):
        for level in cfg["ewma_alpha_grid"]:
            if level != default(p)["ewma_alpha"]:
                add(p, "AA9", "ewma_alpha", level, level)
    for level in cfg["theta_hi_grid"]:
        if level != default("P1")["theta_hi"]:
            add("P1", "AA9", "theta_hi", level, level)
    return out


def plan_q1(cfg, tuned, ext, pilot):
    tasks = []
    for ds in DATASETS:
        data = load_dataset(cfg, ds)
        labels = {q1_label(s["base"], s["factor"], s["param"], s["level"]): s for s in q1_specs(cfg, tuned, ext, ds)}
        if pilot:
            split = "tune"
            groups = [("all", select_events(cfg, data, split, cfg["pilot_replicates"]))]
            variants, labels = q1_base_variants(tuned, ds), {}
        else:
            split = "eval"
            groups = eval_groups(cfg, data, ds, split)
            variants = list(labels)
        for cls, events in groups:
            for j, ev in enumerate(events):
                tasks.append({"dataset": ds, "split": split, "cls": cls, "j": j, "event_id": str(ev["event_id"]),
                              "m": int(cfg["migration_steps_m0"]), "headroom": float(cfg["headroom"]),
                              "variants": list(variants), "specs": labels, "stage": "pilot" if pilot else "grid"})
    return tasks


def plan_q2(cfg):
    tasks = []
    for base in DATASETS:
        data = load_d3(cfg, base)
        variants = ["P1", cfg["q2_forecast_variants"][base], "P5"]
        el = eligible_events(cfg, data, "eval")
        for cls in sorted(set(el["onset_class"])):
            sub = el[el["onset_class"] == cls].reset_index(drop=True)
            for j in range(cfg["replicates_per_cell"]):
                ev = sub.iloc[j % len(sub)]
                for m in cfg["m_levels"]:
                    tasks.append({"dataset": data.name, "base": base, "split": "eval", "cls": cls, "j": j,
                                  "event_id": str(ev["event_id"]), "m": int(m), "headroom": float(cfg["headroom"]),
                                  "variants": list(variants), "stage": "grid", "queue": "q2"})
    return tasks


def resolve_spec(variant, spec, tuned, ext, dataset):
    base = variant if spec is None else spec["base"]
    if base == "P4a":
        params = dict(ext["tuned_ext"][dataset]["P4a"])
        name, forecaster = "P4a", params["forecaster"]
    else:
        name, params, forecaster = resolve(base, tuned, dataset)
    kwargs = {}
    if spec is not None:
        if spec["param"] == "source":
            donor = dict(ext["tuned_ext"][spec["applied"]]["P4a"]) if base == "P4a" else resolve(base, tuned, spec["applied"])[1]
            params.update({key: value for key, value in donor.items() if key != "forecaster"})
        elif spec["param"] in ("lead", "budget", "margin_quantile", "kappa", "log", "d", "k"):
            kwargs[spec["param"]] = spec["applied"]
        else:
            params[spec["param"]] = spec["applied"]
    return name, params, forecaster, kwargs


Q6_AA7 = ("P1", "P3")
Q6_AA8 = ("P1", "P3", "P5")


def q6_specs(cfg, tuned, ds):
    p3 = "P3_" + tuned["tuned"][ds]["P4"]["forecaster"]
    out = []
    for factor, param, key, default, bases in (
            ("AA7", "d", "monitor_delay_levels", cfg["monitor_delay_d"], Q6_AA7),
            ("AA8", "k", "decision_interval_levels", cfg["decision_interval_k"], Q6_AA8)):
        for base in bases:
            for level in cfg[key]:
                if level != default:
                    out.append({"queue": "q6", "base": p3 if base == "P3" else base, "factor": factor,
                                "param": param, "level": level, "applied": level})
    return out


def plan_q6(cfg, tuned):
    tasks = []
    for ds in DATASETS:
        data = load_dataset(cfg, ds)
        labels = {q1_label(s["base"], s["factor"], s["param"], s["level"]): s for s in q6_specs(cfg, tuned, ds)}
        for cls, events in eval_groups(cfg, data, ds, "eval"):
            for j, ev in enumerate(events):
                tasks.append({"dataset": ds, "split": "eval", "cls": cls, "j": j, "event_id": str(ev["event_id"]),
                              "m": int(cfg["migration_steps_m0"]), "headroom": float(cfg["headroom"]),
                              "variants": list(labels), "specs": labels, "stage": "grid", "queue": "q6"})
    return tasks


def q8_specs(tuned, ds):
    """XA2 (A54): every tunable policy on the target dataset with the tuned parameters of each source dataset,
    the other dataset (transfer) and the dataset itself (control)."""
    return [{"queue": "q8", "base": base, "factor": "XA2", "param": "source", "level": source, "applied": source}
            for base in q1_base_variants(tuned, ds) for source in DATASETS]


def plan_q8(cfg, tuned):
    tasks = []
    for ds in DATASETS:
        data = load_dataset(cfg, ds)
        labels = {q1_label(s["base"], s["factor"], s["param"], s["level"]): s for s in q8_specs(tuned, ds)}
        for cls, events in eval_groups(cfg, data, ds, "eval"):
            for j, ev in enumerate(events):
                tasks.append({"dataset": ds, "split": "eval", "cls": cls, "j": j, "event_id": str(ev["event_id"]),
                              "m": int(cfg["migration_steps_m0"]), "headroom": float(cfg["headroom"]),
                              "variants": list(labels), "specs": labels, "stage": "grid", "queue": "q8"})
    return tasks


Q3_POLICIES = ("P1", "P3", "P6", "P5")
COUPLED = "coupled"
CONST_M = "const_m"


def q3_label(base, h_net, mode, phi):
    return "%s:q3+h_net=%s+mode=%s+phi=%s" % (base, h_net, mode, phi)


def q3_specs(cfg):
    out = []
    for base in Q3_POLICIES:
        for h_net in cfg["h_net_levels"]:
            for phi in cfg["phi_levels"]:
                out.append({"queue": "q3", "base": base, "h_net": h_net, "mode": COUPLED, "phi": phi})
            out.append({"queue": "q3", "base": base, "h_net": h_net, "mode": CONST_M, "phi": None})
    return out


def plan_q3(cfg, pilot):
    tasks = []
    for ds in DATASETS:
        data = load_dataset(cfg, ds)
        if pilot:
            split = "tune"
            groups = [("all", select_events(cfg, data, split, cfg["pilot_replicates"]))]
            specs = {b + ":q3": {"queue": "q3", "base": b, "h_net": cfg["h_net"], "mode": COUPLED, "phi": cfg["phi"]}
                     for b in Q3_POLICIES}
        else:
            split = "eval"
            groups = eval_groups(cfg, data, ds, split)
            specs = {q3_label(s["base"], s["h_net"], s["mode"], s["phi"]): s for s in q3_specs(cfg)}
        for cls, events in groups:
            for j, ev in enumerate(events):
                tasks.append({"dataset": ds, "split": split, "cls": cls, "j": j, "event_id": str(ev["event_id"]),
                              "m": int(cfg["migration_steps_m0"]), "headroom": float(cfg["headroom"]),
                              "variants": list(specs), "specs": specs, "stage": "pilot" if pilot else "grid",
                              "queue": "q3"})
    return tasks


def cross_of(name, data, h_net):
    key = (name, h_net)
    if key not in CTX["cross"]:
        CTX["cross"][key] = load_cross(CTX["cfg"], data, h_net)
    return CTX["cross"][key]


def run_task_q3(task):
    cfg = CTX["cfg"]
    data, by_id = dataset_of(task["dataset"])
    rep = make_replicate(cfg, data, by_id[task["event_id"]], task["j"], task["headroom"])
    rows, timing = [], []
    for variant in task["variants"]:
        spec = task["specs"][variant]
        params = dict(CTX["layers"]["tuned_layers"][task["dataset"]][spec["base"]])
        forecaster = params.pop("forecaster", None)
        window = None
        if forecaster is not None:
            window = forecast_window(table_of(task["dataset"], forecaster), rep)
        cross = cross_of(task["dataset"], data, spec["h_net"])
        t0 = time.perf_counter()
        met = run_replicate_layers(cfg, rep, cross, spec["base"], params, m=task["m"], forecast=window,
                                   mode=spec["mode"], phi=spec["phi"], cl=True, replay=spec["mode"] == COUPLED)
        seconds = time.perf_counter() - t0
        cell = cell_id(task["dataset"], task["cls"], task["m"], task["headroom"], variant)
        if not met["conservation_ok"]:
            raise RuntimeError("conservation violated in " + cell)
        row = dict(met, cell=cell, dataset=task["dataset"], onset_class=task["cls"], event_class=rep.onset_class,
                   m=task["m"], headroom=task["headroom"], variant=spec["base"], replicate=task["j"],
                   event_id=rep.event_id, queue=spec["queue"], h_net=spec["h_net"], mode=spec["mode"],
                   phi=spec["phi"])
        rows.append(json.dumps(row, sort_keys=True, default=_py))
        timing.append((task["stage"], variant.split("+", 1)[0], task["dataset"], task["j"], 1, seconds))
    out = Path(CTX["out"])
    out.mkdir(parents=True, exist_ok=True)
    with open(out / ("grid_%d.jsonl" % os.getpid()), "a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(rows) + "\n")
    return timing


Q4_FORECAST = ("P3", "P6")


def q4_label(spec):
    if spec["factor"] == "AA11":
        return "%s:q4+%s:%s=%s" % (spec["base"], spec["factor"], spec["param"], spec["level"])
    return "%s+%s:%s=%s" % (spec["base"], spec["factor"], spec["param"], spec["level"])


def q4_specs(cfg, tuned, ds):
    out = []
    for base in q1_base_variants(tuned, ds):
        for level in cfg["kappa_levels"]:
            if level != cfg["migration_overhead_kappa"]:
                out.append({"queue": "q4", "base": base, "factor": "AA10", "param": "kappa", "level": level,
                            "applied": level})
    for base in Q4_FORECAST:
        out.append({"queue": "q4", "base": base, "factor": "AA11", "param": "layers", "level": "off",
                    "applied": "off"})
    return out


def plan_q4(cfg, tuned, pilot):
    tasks = []
    for ds in DATASETS:
        data = load_dataset(cfg, ds)
        specs = {q4_label(s): s for s in q4_specs(cfg, tuned, ds)}
        if pilot:
            split = "tune"
            groups = [("all", select_events(cfg, data, split, cfg["pilot_replicates"]))]
            specs = {k: s for k, s in specs.items() if s["factor"] == "AA11"}
        else:
            split = "eval"
            groups = eval_groups(cfg, data, ds, split)
        for cls, events in groups:
            for j, ev in enumerate(events):
                tasks.append({"dataset": ds, "split": split, "cls": cls, "j": j, "event_id": str(ev["event_id"]),
                              "m": int(cfg["migration_steps_m0"]), "headroom": float(cfg["headroom"]),
                              "variants": list(specs), "specs": specs, "stage": "pilot" if pilot else "grid",
                              "queue": "q4"})
    return tasks


def run_task_q4(task):
    cfg, tuned = CTX["cfg"], CTX["tuned"]
    ds = task["dataset"]
    data, by_id = dataset_of(ds)
    rep = make_replicate(cfg, data, by_id[task["event_id"]], task["j"], task["headroom"])
    rows, timing = [], []
    for variant in task["variants"]:
        spec = task["specs"][variant]
        t0 = time.perf_counter()
        if spec["factor"] == "AA11":
            params = dict(CTX["layers"]["tuned_layers"][ds][spec["base"]])
            forecaster = params.pop("forecaster", None)
            window = None if forecaster is None else forecast_window(table_of(ds, forecaster), rep)
            met = run_replicate_layers(cfg, rep, cross_of(ds, data, cfg["h_net"]), spec["base"], params,
                                       m=task["m"], forecast=window, mode=CONST_M, nic=False)
        else:
            name, params, forecaster, kwargs = resolve_spec(variant, spec, tuned, CTX["ext"], ds)
            window = None if forecaster is None else forecast_window(table_of(ds, forecaster), rep)
            met = run_replicate(cfg, rep, name, params, m=task["m"], forecast=window, **kwargs)
        seconds = time.perf_counter() - t0
        cell = cell_id(ds, task["cls"], task["m"], task["headroom"], variant)
        if not met["conservation_ok"]:
            raise RuntimeError("conservation violated in " + cell)
        row = dict(met, cell=cell, dataset=ds, onset_class=task["cls"], event_class=rep.onset_class,
                   m=task["m"], headroom=task["headroom"], variant=spec["base"], replicate=task["j"],
                   event_id=rep.event_id, queue=spec["queue"], factor=spec["factor"], param=spec["param"],
                   level=spec["level"], applied=spec["applied"])
        rows.append(json.dumps(row, sort_keys=True, default=_py))
        timing.append((task["stage"], variant.split("+", 1)[0], ds, task["j"], 1, seconds))
    out = Path(CTX["out"])
    out.mkdir(parents=True, exist_ok=True)
    with open(out / ("grid_%d.jsonl" % os.getpid()), "a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(rows) + "\n")
    return timing


Q11_RV4 = ("P4", "P4a")
RV4_ON = "on"


def q11_label(base):
    return "%s:q11+RV4:log=%s" % (base, RV4_ON)


def q11_specs():
    return {q11_label(b): {"queue": "q11", "base": b, "factor": "RV4", "param": "log", "level": RV4_ON,
                           "applied": True} for b in Q11_RV4}


def q11_variants(tuned, ds):
    p3 = "P3_" + tuned["tuned"][ds]["P4"]["forecaster"]
    return ["P0", "P1", "P2", p3, q11_label("P4"), q11_label("P4a"), "P5"]


def plan_q11(cfg, tuned, pilot):
    """Q11 RV0 (A44). D1 and D2: P4 and P4a only, with RV4 logging, on the Must-grid evaluation cells at
    default m0 and headroom (the pilot uses the tuning split). D3: all seven policies, Q2 event scheme."""
    specs = q11_specs()
    common = {"m": int(cfg["migration_steps_m0"]), "headroom": float(cfg["headroom"]), "specs": specs,
              "queue": "q11"}
    tasks = []
    for ds in DATASETS:
        data = load_dataset(cfg, ds)
        if pilot:
            split = "tune"
            groups = [("all", select_events(cfg, data, split, cfg["pilot_replicates"]))]
        else:
            split = "eval"
            groups = eval_groups(cfg, data, ds, split)
        for cls, events in groups:
            for j, ev in enumerate(events):
                tasks.append(dict(common, dataset=ds, split=split, cls=cls, j=j, event_id=str(ev["event_id"]),
                                  variants=list(specs), stage="pilot" if pilot else "grid"))
    if pilot:
        return tasks
    for base in DATASETS:
        data = load_d3(cfg, base)
        el = eligible_events(cfg, data, "eval")
        for cls in sorted(set(el["onset_class"])):
            sub = el[el["onset_class"] == cls].reset_index(drop=True)
            for j in range(cfg["replicates_per_cell"]):
                ev = sub.iloc[j % len(sub)]
                tasks.append(dict(common, dataset=data.name, base=base, split="eval", cls=cls, j=j,
                                  event_id=str(ev["event_id"]), variants=q11_variants(tuned, base),
                                  stage="grid"))
    return tasks


Q11B_RV1 = ("P1", "P3", "P4a", "P5")
Q11B_RV2 = ("P0", "P1", "P4a")
Q11B_RV3 = ("P1", "P4a")
BALANCED = "balanced"
SKEWED = "skewed"
RV1_ROWS = (("full", COUPLED, False), ("cpu_off", COUPLED, True), ("nic_off", CONST_M, False),
            ("both_off", CONST_M, True))


def q11b_spec(base, factor, param, level, placement, mode=None, zero_kappa=False, budget=None):
    return {"queue": "q11b", "base": base, "factor": factor, "param": param, "level": level,
            "placement": placement, "mode": mode, "kappa": 0 if zero_kappa else None, "budget": budget}


def q11b_specs(cfg):
    out = {}
    for base in Q11B_RV1:
        for row, mode, zero in RV1_ROWS:
            out["%s:q11b-rv1+%s" % (base, row)] = q11b_spec(base, "RV1", "contention", row, BALANCED, mode, zero)
    for base in Q11B_RV2:
        out["%s:q11b-rv2+%s" % (base, SKEWED)] = q11b_spec(base, "RV2", "placement", SKEWED, SKEWED)
    for base in Q11B_RV3:
        for placement in (BALANCED, SKEWED):
            level = cfg["throttle_budget_B"]
            out["%s:q11b-rv3+B%s-%s" % (base, level, placement)] = q11b_spec(base, "RV3", "budget", level,
                                                                              placement, budget=level)
    return out


def plan_q11b(cfg, tuned, pilot):
    """Q11 RV1 to RV3 (A45) on D1 and D2: the Must-grid evaluation events (the pilot: tuning split, 5 events)."""
    specs = q11b_specs(cfg)
    tasks = []
    for ds in DATASETS:
        data = load_dataset(cfg, ds)
        if pilot:
            split = "tune"
            groups = [("all", select_events(cfg, data, split, cfg["pilot_replicates"]))]
        else:
            split = "eval"
            groups = eval_groups(cfg, data, ds, split)
        for cls, events in groups:
            for j, ev in enumerate(events):
                tasks.append({"dataset": ds, "split": split, "cls": cls, "j": j, "event_id": str(ev["event_id"]),
                              "m": int(cfg["migration_steps_m0"]), "headroom": float(cfg["headroom"]),
                              "variants": list(specs), "specs": specs, "stage": "pilot" if pilot else "grid",
                              "queue": "q11b"})
    return tasks


def run_task_q11b(task):
    cfg, tuned, ext, layers = CTX["cfg"], CTX["tuned"], CTX["ext"], CTX["layers"]
    ds = task["dataset"]
    data, by_id = dataset_of(ds)
    ev = by_id[task["event_id"]]
    reps = {}
    rows, timing = [], []
    for variant in task["variants"]:
        spec = task["specs"][variant]
        base = spec["base"]
        if spec["placement"] not in reps:
            build = make_replicate_skewed if spec["placement"] == SKEWED else make_replicate
            reps[spec["placement"]] = build(cfg, data, ev, task["j"], task["headroom"])
        rep = reps[spec["placement"]]
        t0 = time.perf_counter()
        extra = {}
        if spec["factor"] == "RV1":
            params = dict(ext["tuned_ext"][ds][base]) if base == "P4a" else dict(layers["tuned_layers"][ds][base])
            forecaster = params.pop("forecaster", None)
            window = None if forecaster is None else forecast_window(table_of(ds, forecaster), rep)
            met, res = run_replicate_layers(cfg, rep, cross_of(ds, data, cfg["h_net"]), base, params, m=task["m"],
                                            forecast=window, mode=spec["mode"], kappa=spec["kappa"], cl=True,
                                            log=base == "P4a", return_result=True)
            starts = [mv[0] for mv in res.moves]
            ends = [mv[4] for mv in res.moves]
            spans = [e - s for s, e in zip(starts, ends)]
            extra = {"mode": spec["mode"],
                     "mig_duration_mean": float(np.mean(spans)) if spans else None,
                     "mig_duration_max": max(spans) if spans else None,
                     "mig_censored": int(sum(e >= rep.load.shape[0] for e in ends))}
        else:
            name, params, forecaster, _kw = resolve_spec(base, None, tuned, ext, ds)
            window = None if forecaster is None else forecast_window(table_of(ds, forecaster), rep)
            kwargs = {}
            if spec["budget"] is not None:
                kwargs["budget"] = spec["budget"]
            if base == "P4a":
                kwargs["log"] = True
            met = run_replicate(cfg, rep, name, params, m=task["m"], forecast=window, **kwargs)
        seconds = time.perf_counter() - t0
        cell = cell_id(ds, task["cls"], task["m"], task["headroom"], variant)
        if not met["conservation_ok"]:
            raise RuntimeError("conservation violated in " + cell)
        row = dict(met, cell=cell, dataset=ds, onset_class=task["cls"], event_class=rep.onset_class,
                   m=task["m"], headroom=task["headroom"], variant=base, replicate=task["j"],
                   event_id=rep.event_id, queue="q11b", factor=spec["factor"], param=spec["param"],
                   level=spec["level"], placement=spec["placement"],
                   budget=cfg["migration_budget_B"] if spec["budget"] is None else spec["budget"],
                   kappa_applied=cfg["migration_overhead_kappa"] if spec["kappa"] is None else spec["kappa"])
        row.update(extra)
        rows.append(json.dumps(row, sort_keys=True, default=_py))
        timing.append((task["stage"], variant.split("+", 1)[0], ds, task["j"], 1, seconds))
    out = Path(CTX["out"])
    out.mkdir(parents=True, exist_ok=True)
    with open(out / ("grid_%d.jsonl" % os.getpid()), "a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(rows) + "\n")
    return timing


def run_task(task):
    if task.get("queue") == "q3":
        return run_task_q3(task)
    if task.get("queue") == "q4":
        return run_task_q4(task)
    if task.get("queue") == "q11b":
        return run_task_q11b(task)
    cfg, tuned = CTX["cfg"], CTX["tuned"]
    data, by_id = dataset_of(task["dataset"])
    rep = make_replicate(cfg, data, by_id[task["event_id"]], task["j"], task["headroom"])
    rows, timing = [], []
    for variant in task["variants"]:
        spec = task.get("specs", {}).get(variant)
        name, params, forecaster, kwargs = resolve_spec(variant, spec, tuned, CTX["ext"], task.get("base", task["dataset"]))
        window = None
        if forecaster is not None:
            window = forecast_window(table_of(task["dataset"], forecaster), rep)
        t0 = time.perf_counter()
        met = run_replicate(cfg, rep, name, params, m=task["m"], forecast=window, **kwargs)
        seconds = time.perf_counter() - t0
        cell = cell_id(task["dataset"], task["cls"], task["m"], task["headroom"], variant)
        if not met["conservation_ok"]:
            raise RuntimeError("conservation violated in " + cell)
        row = dict(met, cell=cell, dataset=task["dataset"], onset_class=task["cls"], event_class=rep.onset_class,
                   m=task["m"], headroom=task["headroom"], variant=variant if spec is None else spec["base"],
                   replicate=task["j"], event_id=rep.event_id)
        if spec is not None:
            row.update(queue=spec["queue"], factor=spec["factor"], param=spec["param"], level=spec["level"],
                       applied=spec["applied"])
        if task.get("queue"):
            row["queue"] = task["queue"]
        rows.append(json.dumps(row, sort_keys=True, default=_py))
        timing.append((task["stage"], variant.split("+", 1)[0], task["dataset"], task["j"], 1, seconds))
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
            key = (t.get("base", t["dataset"]), v.split("+", 1)[0])
            counts[key] = counts.get(key, 0) + 1
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
    parser.add_argument("--tuned-ext")
    parser.add_argument("--tuned-layers")
    parser.add_argument("--queue", required=True, choices=["must", "q1", "q2", "q3", "q4", "q6", "q8", "q11", "q11b"])
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--oracle-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-row")
    parser.add_argument("--out")
    args = parser.parse_args()
    cfg = load_config(args.config)
    tuned = yaml.safe_load(Path(args.tuned).read_text(encoding="utf-8"))
    ext = yaml.safe_load(Path(args.tuned_ext).read_text(encoding="utf-8")) if args.tuned_ext else None
    layers = yaml.safe_load(Path(args.tuned_layers).read_text(encoding="utf-8")) if args.tuned_layers else None
    if args.queue == "q1":
        if ext is None:
            parser.error("--tuned-ext is required for q1")
        if args.oracle_check:
            parser.error("--oracle-check applies to the must queue only")
        tasks = plan_q1(cfg, tuned, ext, args.pilot)
    elif args.queue == "q2":
        if args.pilot or args.oracle_check:
            parser.error("q2 has no pilot and no oracle check")
        tasks = plan_q2(cfg)
    elif args.queue == "q6":
        if args.pilot or args.oracle_check:
            parser.error("q6 has no pilot and no oracle check")
        tasks = plan_q6(cfg, tuned)
    elif args.queue == "q8":
        if ext is None:
            parser.error("--tuned-ext is required for q8")
        if args.pilot or args.oracle_check:
            parser.error("q8 has no pilot and no oracle check")
        tasks = plan_q8(cfg, tuned)
    elif args.queue == "q3":
        if layers is None:
            parser.error("--tuned-layers is required for q3")
        if args.oracle_check:
            parser.error("--oracle-check applies to the must queue only")
        tasks = plan_q3(cfg, args.pilot)
    elif args.queue == "q4":
        if ext is None or layers is None:
            parser.error("--tuned-ext and --tuned-layers are required for q4")
        if args.oracle_check:
            parser.error("--oracle-check applies to the must queue only")
        tasks = plan_q4(cfg, tuned, args.pilot)
    elif args.queue == "q11":
        if ext is None:
            parser.error("--tuned-ext is required for q11")
        if args.oracle_check:
            parser.error("--oracle-check applies to the must queue only")
        tasks = plan_q11(cfg, tuned, args.pilot)
    elif args.queue == "q11b":
        if ext is None or layers is None:
            parser.error("--tuned-ext and --tuned-layers are required for q11b")
        if args.oracle_check:
            parser.error("--oracle-check applies to the must queue only")
        tasks = plan_q11b(cfg, tuned, args.pilot)
    else:
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
        init(cfg, tuned, args.out, ext, layers)
        append_timing(cfg, run_task(todo[0]))
        print("done")
        return
    with mp.Pool(processes=cfg["workers"], initializer=init, initargs=(cfg, tuned, args.out, ext, layers)) as pool:
        for k, timing in enumerate(pool.imap_unordered(run_task, todo, chunksize=1), start=1):
            append_timing(cfg, timing)
            if k % step == 0 or k == len(todo):
                print("tasks done %d / %d" % (k, len(todo)), flush=True)
    print("finished")


if __name__ == "__main__":
    main()
