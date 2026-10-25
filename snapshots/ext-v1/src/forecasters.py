"""Forecasters (Section 4.3) and rolling-origin forecast tables.

F[o, h_index, p] forecasts L[o + h, p] using only rows 0..o. A policy deciding at step t with
monitor delay d may read only F[t - d] and earlier (the simulator enforces this in Phase 7).
Learned models are refit every refit_interval steps on all rows known at the refit time; origins
before the first usable origin carry the naive forecast.
"""
import math

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.config import step_seconds
from src.seeds import derive_seed

BASELINES = ("naive", "ewma", "seasonal_naive")
LEARNED = ("ridge", "lightgbm")


def day_steps(cfg, dataset):
    return int(round(cfg["day_seconds"] / step_seconds(cfg, dataset)))


def naive_table(load, horizons):
    base = np.asarray(load, dtype=np.float32)
    return np.repeat(base[:, None, :], len(horizons), axis=1)


def ewma_table(load, horizons, alpha):
    frame = pd.DataFrame(np.asarray(load, dtype=np.float64))
    value = frame.ewm(alpha=alpha, adjust=False).mean().to_numpy().astype(np.float32)
    return np.repeat(value[:, None, :], len(horizons), axis=1)


def seasonal_naive_table(load, horizons, period):
    arr = np.asarray(load, dtype=np.float32)
    out = naive_table(arr, horizons)
    rows = np.arange(arr.shape[0])
    for i, h in enumerate(horizons):
        src = rows + h - period
        ok = (src >= 0) & (src <= rows)
        out[ok, i, :] = arr[src[ok]]
    return out


def feature_block(cfg, log_load, origins, day):
    lookback, roll = cfg["lookback_L"], cfg["roll_window"]
    if roll > lookback:
        raise ValueError("roll_window must not exceed lookback_L")
    n_p = log_load.shape[1]
    idx = origins[:, None] - np.arange(lookback)[None, :]
    lags = log_load[idx].transpose(0, 2, 1).reshape(origins.size * n_p, lookback)
    recent = lags[:, :roll]
    mean = recent.mean(axis=1, keepdims=True)
    std = recent.std(axis=1, keepdims=True)
    phase = math.tau * (origins % day) / day
    tod = np.repeat(np.stack([np.sin(phase), np.cos(phase)], axis=1), n_p, axis=0)
    return np.hstack([lags, mean, std, tod]).astype(np.float32)


def build_features(cfg, log_load, origins, day):
    chunk = cfg["feature_chunk_origins"]
    parts = [feature_block(cfg, log_load, origins[i:i + chunk], day)
             for i in range(0, origins.size, chunk)]
    return np.concatenate(parts)


def make_model(kind, hp, cfg, seed, n_jobs):
    if kind == "ridge":
        return make_pipeline(FunctionTransformer(np.asarray, kw_args={"dtype": np.float64}), StandardScaler(), Ridge(alpha=hp))
    if kind == "lightgbm":
        return lgb.LGBMRegressor(
            n_estimators=cfg["lgbm_trees"], learning_rate=cfg["lgbm_lr"], num_leaves=int(hp),
            random_state=seed, n_jobs=n_jobs, deterministic=True, force_row_wise=True,
            verbosity=cfg["lgbm_verbosity"])
    raise ValueError("unknown learned forecaster: " + str(kind))


def fit_models(cfg, label, kind, hp, log_load, known_end, day, n_jobs):
    """One model per horizon, trained only on samples whose target row is below known_end."""
    models = []
    for h in cfg["horizons"]:
        origins = np.arange(cfg["lookback_L"] - 1, known_end - h, cfg["origin_stride"])
        if origins.size == 0:
            raise ValueError("no training origins for horizon %d" % h)
        x = build_features(cfg, log_load, origins, day)
        y = log_load[origins + h].ravel()
        seed = derive_seed("forecaster", cfg["master_seed"], label, kind, h, known_end) % (1 << cfg["seed_bits"])
        model = make_model(kind, hp, cfg, seed, n_jobs)
        model.fit(x, y)
        models.append(model)
    return models


def predict_origins(cfg, models, log_load, origins, day):
    n_p = log_load.shape[1]
    out = np.empty((origins.size, len(models), n_p), dtype=np.float32)
    chunk = cfg["feature_chunk_origins"]
    for i0 in range(0, origins.size, chunk):
        sub = origins[i0:i0 + chunk]
        x = feature_block(cfg, log_load, sub, day)
        for j, model in enumerate(models):
            pred = model.predict(x).reshape(sub.size, n_p)
            out[i0:i0 + sub.size, j, :] = np.maximum(np.expm1(pred), 0)
    return out


def learned_table(cfg, label, kind, hp, load, train_end, day, n_jobs):
    n_steps = load.shape[0]
    log_load = np.log1p(np.asarray(load, dtype=np.float32))
    table = naive_table(load, cfg["horizons"])
    k = 0
    while True:
        known = train_end + k * cfg["refit_interval"]
        lo = known - 1
        if lo >= n_steps:
            break
        hi = min(known + cfg["refit_interval"] - 1, n_steps)
        models = fit_models(cfg, label, kind, hp, log_load, known, day, n_jobs)
        table[lo:hi] = predict_origins(cfg, models, log_load, np.arange(lo, hi), day)
        k += 1
    return table


def build_table(cfg, label, kind, load, train_end, day, hp=None, n_jobs=1):
    horizons = cfg["horizons"]
    if kind == "naive":
        return naive_table(load, horizons)
    if kind == "ewma":
        return ewma_table(load, horizons, cfg["forecast_ewma_alpha"])
    if kind == "seasonal_naive":
        return seasonal_naive_table(load, horizons, day)
    if kind in LEARNED:
        return learned_table(cfg, label, kind, hp, load, train_end, day, n_jobs)
    raise ValueError("unknown forecaster: " + str(kind))