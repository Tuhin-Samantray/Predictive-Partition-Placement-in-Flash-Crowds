"""Forecast accuracy (MASE, sMAPE), early warning and forecaster tuning (Section 4.3, Phase 6).

All definitions are fixed in amendment A9. Tables are indexed by forecast origin: F[o, h_index, p]
forecasts L[o + h, p].
"""
import numpy as np
import pandas as pd

from src.forecasters import fit_models, predict_origins

FORECAST_GRID = {"ridge": "ridge_alpha", "lightgbm": "lightgbm_leaves"}


def split_spans(train_end, tune_end, n_steps):
    return (("tune", train_end, tune_end), ("eval", tune_end, n_steps))


def scales(load, train_end, horizons):
    prefix = np.asarray(load[:train_end], dtype=np.float64)
    return np.stack([np.abs(prefix[h:] - prefix[:prefix.shape[0] - h]).mean(axis=0) for h in horizons])


def event_cells(events, n_steps, n_part):
    """Boolean (n_steps, n_part): step t lies inside [onset, end] of a detected event on that partition."""
    diff = np.zeros((n_steps + 1, n_part), dtype=np.int32)
    part = events["partition"].to_numpy()
    np.add.at(diff, (events["onset"].to_numpy(), part), 1)
    np.add.at(diff, (np.minimum(events["end"].to_numpy() + 1, n_steps), part), -1)
    return np.cumsum(diff, axis=0)[:n_steps] > 0


def error_sums(pred, truth, scale_h, eps, mask=None):
    err = np.abs(pred - truth)
    keep = np.ones(err.shape, dtype=bool) if mask is None else mask
    positive = scale_h > 0
    use = keep & positive[None, :]
    safe = np.where(positive, scale_h, np.ones_like(scale_h))
    return {
        "n_cells": int(keep.sum()),
        "n_mase_cells": int(use.sum()),
        "mase_sum": float((err / safe[None, :])[use].sum()),
        "smape_sum": float(((err + err) / (np.abs(pred) + np.abs(truth) + eps))[keep].sum()),
        "n_zero_scale": int((~positive).sum()),
    }


def ratio(total, count):
    return total / count if count else float("nan")


def accuracy_rows(cfg, label, kind, table, load, events, train_end, tune_end):
    hs = cfg["horizons"]
    n_steps, n_part = load.shape
    scale = scales(load, train_end, hs)
    rows = []
    for split, lo, hi in split_spans(train_end, tune_end, n_steps):
        in_event = event_cells(events[events["split"] == split], n_steps, n_part)
        for j, h in enumerate(hs):
            pred = np.asarray(table[lo:hi - h, j, :], dtype=np.float64)
            truth = np.asarray(load[lo + h:hi], dtype=np.float64)
            a = error_sums(pred, truth, scale[j], cfg["smape_eps"])
            e = error_sums(pred, truth, scale[j], cfg["smape_eps"], in_event[lo + h:hi])
            rows.append({
                "dataset": label, "forecaster": kind, "split": split, "horizon": h,
                "mase_all": ratio(a["mase_sum"], a["n_mase_cells"]),
                "smape_all": ratio(a["smape_sum"], a["n_cells"]),
                "mase_event": ratio(e["mase_sum"], e["n_mase_cells"]),
                "smape_event": ratio(e["smape_sum"], e["n_cells"]),
                "n_cells_all": a["n_cells"], "n_cells_event": e["n_cells"],
                "n_zero_scale": a["n_zero_scale"],
            })
    return rows


def early_warning_rows(cfg, label, kind, table, load, events, train_end, tune_end):
    n_steps, n_part = load.shape
    hm = max(cfg["horizons"])
    peak = np.asarray(table).max(axis=1)
    ref = pd.DataFrame(np.asarray(load, dtype=np.float64)).rolling(
        cfg["warn_baseline_window"], min_periods=1).mean().to_numpy()
    alarm = peak >= cfg["warn_ratio"] * (ref + cfg["smape_eps"])
    active = event_cells(events, n_steps, n_part)
    onsets = np.zeros((n_steps, n_part), dtype=np.int32)
    np.add.at(onsets, (events["onset"].to_numpy(), events["partition"].to_numpy()), 1)
    csum = np.cumsum(onsets, axis=0)
    rows = []
    for split, lo, hi in split_spans(train_end, tune_end, n_steps):
        origins = np.arange(lo, max(lo, hi - hm))
        cells = alarm[origins] & ~active[origins]
        future = csum[origins + hm] - csum[origins]
        n_alarm = int(cells.sum())
        precision = ratio(float((cells & (future > 0)).sum()), n_alarm)
        ev = events[events["split"] == split]
        usable = ev[ev["onset"] - hm >= lo].reset_index(drop=True)
        warned = np.zeros(len(usable), dtype=bool)
        lead = np.zeros(len(usable), dtype=np.int64)
        for i, (ons, p) in enumerate(zip(usable["onset"].to_numpy(), usable["partition"].to_numpy())):
            window = alarm[ons - hm:ons, p]
            if window.any():
                warned[i] = True
                lead[i] = hm - int(window.argmax())
        classes = ("all",) + tuple(sorted(usable["onset_class"].unique()))
        for cls in classes:
            sel = np.ones(len(usable), dtype=bool) if cls == "all" else (usable["onset_class"] == cls).to_numpy()
            got = lead[sel & warned]
            rows.append({
                "dataset": label, "forecaster": kind, "split": split, "onset_class": cls,
                "n_events": int(sel.sum()), "n_excluded_events": int(len(ev) - len(usable)),
                "recall": ratio(float((warned & sel).sum()), int(sel.sum())),
                "lead_median": float(np.median(got)) if got.size else float("nan"),
                "lead_mean": float(got.mean()) if got.size else float("nan"),
                "n_alarm_cells": n_alarm,
                "precision": precision if cls == "all" else float("nan"),
            })
    return rows


def tune_forecaster(cfg, label, kind, load, train_end, tune_end, day, n_jobs):
    """Exhaustive search of the forecaster grid: model 0 (training prefix only), tune split, mean MASE."""
    hs = cfg["horizons"]
    log_load = np.log1p(np.asarray(load, dtype=np.float32))
    scale = scales(load, train_end, hs)
    origins = np.arange(train_end, tune_end - min(hs), cfg["origin_stride"])
    results = []
    for hp in cfg[FORECAST_GRID[kind]]:
        models = fit_models(cfg, label, kind, hp, log_load, train_end, day, n_jobs)
        pred = predict_origins(cfg, models, log_load, origins, day)
        per_h = []
        for j, h in enumerate(hs):
            sel = origins + h < tune_end
            truth = np.asarray(load[origins[sel] + h], dtype=np.float64)
            sums = error_sums(pred[sel, j, :].astype(np.float64), truth, scale[j], cfg["smape_eps"])
            per_h.append(ratio(sums["mase_sum"], sums["n_mase_cells"]))
        results.append({"hp": hp, "mase_by_horizon": per_h, "objective": float(np.mean(per_h))})
    best = min(range(len(results)), key=lambda i: (results[i]["objective"], i))
    return results, best