"""Subset accuracy and early warning (A51): equal to the Phase 6 functions on all evaluation origins."""
import numpy as np
import pandas as pd

from src.accuracy import accuracy_rows, early_warning_rows
from src.accuracy_subset import eval_origins, subset_accuracy_rows, subset_early_warning_rows

TRAIN_END, TUNE_END, N_STEPS, N_PART = 20, 35, 80, 7


def _world():
    rng = np.random.default_rng(7)
    load = rng.gamma(2.0, 2.0, size=(N_STEPS, N_PART))
    boost = rng.choice([1.0, 1.0, 1.0, 9.0], size=(N_STEPS, 3, N_PART))
    table = (load[:, None, :] * boost).astype(np.float32)
    cfg = {"horizons": [1, 2, 3], "smape_eps": 0.001, "warn_ratio": 3.0, "warn_baseline_window": 5}
    events = pd.DataFrame({
        "partition": [3, 5, 1, 4, 2, 6], "onset": [22, 26, 45, 55, 62, 70], "end": [24, 27, 47, 58, 63, 72],
        "split": ["tune", "tune", "eval", "eval", "eval", "eval"],
        "onset_class": ["abrupt", "ramped", "abrupt", "ramped", "abrupt", "ramped"]})
    return cfg, load, table, events


def _same(base, got):
    assert len(base) == len(got) and len(base) > 0
    for a, b in zip(base, got):
        for key, value in a.items():
            if isinstance(value, str):
                assert b[key] == value
            else:
                assert np.isclose(value, b[key], equal_nan=True), (key, value, b[key])


def test_full_origins_equal_phase6_accuracy():
    cfg, load, table, events = _world()
    base = [r for r in accuracy_rows(cfg, "X", "k", table, load, events, TRAIN_END, TUNE_END) if r["split"] == "eval"]
    origins = eval_origins(TUNE_END, N_STEPS, 1, cfg["horizons"])
    _same(base, subset_accuracy_rows(cfg, "X", "k", table, load, events, TRAIN_END, TUNE_END, origins))


def test_full_origins_equal_phase6_early_warning():
    cfg, load, table, events = _world()
    base = [r for r in early_warning_rows(cfg, "X", "k", table, load, events, TRAIN_END, TUNE_END) if r["split"] == "eval"]
    origins = eval_origins(TUNE_END, N_STEPS, 1, cfg["horizons"])
    _same(base, subset_early_warning_rows(cfg, "X", "k", table, load, events, TRAIN_END, TUNE_END, origins))


def test_strided_origins_counts_and_values():
    cfg, load, table, events = _world()
    stride = 3
    origins = eval_origins(TUNE_END, N_STEPS, stride, cfg["horizons"])
    assert origins[0] == TUNE_END and np.all(np.diff(origins) == stride)
    rows = subset_accuracy_rows(cfg, "X", "k", table, load, events, TRAIN_END, TUNE_END, origins)
    for row in rows:
        n_ok = int((origins + row["horizon"] < N_STEPS).sum())
        assert row["n_origins"] == n_ok and row["n_cells_all"] == n_ok * N_PART
    o = origins[origins + 1 < N_STEPS]
    pred = table[o, 0, :].astype(np.float64)
    truth = load[o + 1]
    smape = np.mean((np.abs(pred - truth) * 2) / (np.abs(pred) + np.abs(truth) + cfg["smape_eps"]))
    assert np.isclose(rows[0]["smape_all"], smape)


def test_strided_early_warning_is_sane():
    cfg, load, table, events = _world()
    origins = eval_origins(TUNE_END, N_STEPS, 3, cfg["horizons"])
    rows = subset_early_warning_rows(cfg, "X", "k", table, load, events, TRAIN_END, TUNE_END, origins)
    assert rows[0]["onset_class"] == "all" and rows[0]["n_events"] == 4
    for row in rows:
        assert 0 <= row["recall"] <= 1
