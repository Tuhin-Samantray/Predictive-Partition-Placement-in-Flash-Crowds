"""Accuracy and early warning on a subset of forecast origins (A51, used for the Chronos comparison).

With the subset equal to all evaluation origins the rows equal the evaluation rows of accuracy_rows and
early_warning_rows (tested). Tables are indexed by forecast origin: F[o, h_index, p] forecasts L[o + h, p].
"""
import numpy as np
import pandas as pd

from src.accuracy import error_sums, event_cells, ratio, scales


def eval_origins(tune_end, n_steps, stride, horizons):
    """Evaluation origins tune_end, tune_end + stride, ... that have at least the shortest horizon ahead."""
    return np.arange(tune_end, n_steps - min(horizons), stride)


def subset_accuracy_rows(cfg, label, kind, table, load, events, train_end, tune_end, origins):
    hs = cfg["horizons"]
    n_steps, n_part = load.shape
    scale = scales(load, train_end, hs)
    in_event = event_cells(events[events["split"] == "eval"], n_steps, n_part)
    origins = np.asarray(origins)
    rows = []
    for j, h in enumerate(hs):
        o = origins[origins + h < n_steps]
        pred = np.asarray(table[o, j, :], dtype=np.float64)
        truth = np.asarray(load[o + h], dtype=np.float64)
        a = error_sums(pred, truth, scale[j], cfg["smape_eps"])
        e = error_sums(pred, truth, scale[j], cfg["smape_eps"], in_event[o + h])
        rows.append({
            "dataset": label, "forecaster": kind, "split": "eval", "horizon": h,
            "mase_all": ratio(a["mase_sum"], a["n_mase_cells"]),
            "smape_all": ratio(a["smape_sum"], a["n_cells"]),
            "mase_event": ratio(e["mase_sum"], e["n_mase_cells"]),
            "smape_event": ratio(e["smape_sum"], e["n_cells"]),
            "n_cells_all": a["n_cells"], "n_cells_event": e["n_cells"],
            "n_zero_scale": a["n_zero_scale"], "n_origins": int(o.size),
        })
    return rows


def subset_early_warning_rows(cfg, label, kind, table, load, events, train_end, tune_end, origins):
    n_steps, n_part = load.shape
    hm = max(cfg["horizons"])
    origins = np.asarray(origins)
    peak = np.asarray(table[origins]).max(axis=1)
    ref = pd.DataFrame(np.asarray(load, dtype=np.float64)).rolling(
        cfg["warn_baseline_window"], min_periods=1).mean().to_numpy()
    alarm = peak >= cfg["warn_ratio"] * (ref[origins] + cfg["smape_eps"])
    active = event_cells(events, n_steps, n_part)
    onsets = np.zeros((n_steps, n_part), dtype=np.int32)
    np.add.at(onsets, (events["onset"].to_numpy(), events["partition"].to_numpy()), 1)
    csum = np.cumsum(onsets, axis=0)
    keep = origins < n_steps - hm
    kept = origins[keep]
    cells = alarm[keep] & ~active[kept]
    future = csum[kept + hm] - csum[kept]
    n_alarm = int(cells.sum())
    precision = ratio(float((cells & (future > 0)).sum()), n_alarm)
    ev = events[events["split"] == "eval"]
    usable = ev[ev["onset"] - hm >= tune_end].reset_index(drop=True)
    warned = np.zeros(len(usable), dtype=bool)
    lead = np.zeros(len(usable), dtype=np.int64)
    for i, (ons, p) in enumerate(zip(usable["onset"].to_numpy(), usable["partition"].to_numpy())):
        first = int(np.searchsorted(origins, ons - hm, side="left"))
        stop = int(np.searchsorted(origins, ons, side="left"))
        window = alarm[first:stop, p]
        if window.any():
            warned[i] = True
            lead[i] = ons - int(origins[first + int(window.argmax())])
    rows = []
    classes = ("all",) + tuple(sorted(usable["onset_class"].unique()))
    for cls in classes:
        sel = np.ones(len(usable), dtype=bool) if cls == "all" else (usable["onset_class"] == cls).to_numpy()
        got = lead[sel & warned]
        rows.append({
            "dataset": label, "forecaster": kind, "split": "eval", "onset_class": cls,
            "n_events": int(sel.sum()), "n_excluded_events": int(len(ev) - len(usable)),
            "recall": ratio(float((warned & sel).sum()), int(sel.sum())),
            "lead_median": float(np.median(got)) if got.size else float("nan"),
            "lead_mean": float(got.mean()) if got.size else float("nan"),
            "n_alarm_cells": n_alarm,
            "precision": precision if cls == "all" else float("nan"),
            "n_origins": int(origins.size),
        })
    return rows
