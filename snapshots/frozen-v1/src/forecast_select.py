"""Pre-registered forecaster choice for P4 (A13): the forecaster with the lowest tuning-split MASE.

MASE is computed from the stored forecast tables with the A9 definition: absolute errors are
divided by each partition's mean absolute h-step naive change on the training prefix, partitions
with zero scale are excluded, origins run from train_end to tune_end - h - 1 on origin_stride,
and the objective is the mean over horizons. Ties go to the earlier name in forecaster_names.
"""
import numpy as np

from src.replicates import load_forecast


def tune_mase(cfg, table, load, train_end, tune_end):
    per_h = []
    for i, h in enumerate(cfg["horizons"]):
        scale = np.abs(np.asarray(load[h:train_end], dtype=np.float64)
                       - np.asarray(load[:train_end - h], dtype=np.float64)).mean(axis=0)
        ok = scale > 0
        origins = np.arange(train_end, tune_end - h, cfg["origin_stride"])
        pred = np.asarray(table[origins, i, :], dtype=np.float64)
        truth = np.asarray(load[origins + h], dtype=np.float64)
        per_h.append(float((np.abs(pred - truth)[:, ok] / scale[ok]).mean()))
    return float(np.mean(per_h)), per_h


def choose_forecaster(cfg, dataset, load, train_end, tune_end):
    scores = {}
    for name in cfg["forecaster_names"]:
        scores[name] = tune_mase(cfg, load_forecast(cfg, dataset, name), load, train_end, tune_end)
    best = min(scores, key=lambda n: scores[n][0])
    return best, scores
