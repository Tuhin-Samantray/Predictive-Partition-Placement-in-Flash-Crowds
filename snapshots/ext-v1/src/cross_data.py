"""Forecast-only cross-data tables (Q5, A37): XA1 transfer, XA7 event versus non-event, XA5 accuracy by day."""
import numpy as np

from src.accuracy import error_sums, event_cells, ratio, scales
from src.forecasters import fit_models, predict_origins


def group_masks(in_event, lo, hi, h, day):
    """Cell masks over the (origins, partitions) block scoring truth rows lo + h .. hi - 1."""
    rows = np.arange(lo + h, hi)
    ev = in_event[lo + h:hi]
    masks = {"all": np.ones(ev.shape, dtype=bool), "event": ev, "nonevent": ~ev}
    days = rows // day - lo // day
    for d in np.unique(days):
        sel = np.zeros(ev.shape, dtype=bool)
        sel[days == d, :] = True
        masks["day%d" % d] = sel
    return masks


def metric_rows(cfg, tgt, day, pred_by_h, meta):
    hs = cfg["horizons"]
    n_steps, n_part = tgt.load.shape
    lo, hi = tgt.tune_end, tgt.n_steps
    scale = scales(tgt.load, tgt.train_end, hs)
    in_event = event_cells(tgt.events[tgt.events["split"] == "eval"], n_steps, n_part)
    rows = []
    for j, h in enumerate(hs):
        truth = np.asarray(tgt.load[lo + h:hi], dtype=np.float64)
        pred = np.asarray(pred_by_h[j], dtype=np.float64)
        for group, mask in group_masks(in_event, lo, hi, h, day).items():
            s = error_sums(pred, truth, scale[j], cfg["smape_eps"], mask)
            rows.append({**meta, "horizon": h, "group": group, "n_cells": s["n_cells"],
                         "n_mase_cells": s["n_mase_cells"], "mase": ratio(s["mase_sum"], s["n_mase_cells"]),
                         "smape": ratio(s["smape_sum"], s["n_cells"])})
    return rows


def rolling_predictions(cfg, table, tgt):
    lo, hi = tgt.tune_end, tgt.n_steps
    return [np.asarray(table[lo:hi - h, j, :]) for j, h in enumerate(cfg["horizons"])]


def fixed_model_predictions(cfg, src, tgt, kind, hp, day_src, day_tgt, n_jobs):
    """One fixed model per horizon fitted on the source training prefix, predicting every target evaluation origin."""
    log_src = np.log1p(np.asarray(src.load, dtype=np.float32))
    models = fit_models(cfg, src.name, kind, hp, log_src, src.train_end, day_src, n_jobs)
    log_tgt = np.log1p(np.asarray(tgt.load, dtype=np.float32))
    hs = cfg["horizons"]
    lo, hi = tgt.tune_end, tgt.n_steps
    origins = np.arange(lo, hi - min(hs))
    pred = predict_origins(cfg, models, log_tgt, origins, day_tgt)
    return [pred[:hi - lo - h, j, :] for j, h in enumerate(hs)]
