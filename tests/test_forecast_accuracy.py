import json

import numpy as np
import pandas as pd
import pytest

from small_world import real_cfg
from src.accuracy import accuracy_rows, early_warning_rows, tune_forecaster
from src.forecasters import naive_table

TRAIN_END, TUNE_END = 100, 200


def cfg_small():
    cfg = real_cfg()
    cfg.update(lookback_L=12, roll_window=5, refit_interval=40, origin_stride=2,
               lgbm_trees=5, feature_chunk_origins=16, warn_baseline_window=20)
    return cfg


def events_frame(rows):
    cols = ["partition", "onset", "end", "split", "onset_class"]
    return pd.DataFrame(rows, columns=cols).astype({"partition": int, "onset": int, "end": int})


def pick(rows, **want):
    return [r for r in rows if all(r[k] == v for k, v in want.items())]


def test_naive_on_a_ramp_has_mase_one():
    cfg = cfg_small()
    load = np.tile(np.arange(300.0)[:, None], (1, 3))
    events = events_frame([(0, 210, 215, "eval", "abrupt")])
    rows = accuracy_rows(cfg, "S", "naive", naive_table(load, cfg["horizons"]), load, events, TRAIN_END, TUNE_END)
    ev = pick(rows, split="eval")
    assert len(ev) == len(cfg["horizons"])
    for r in ev:
        assert r["mase_all"] == pytest.approx(1.0) and r["mase_event"] == pytest.approx(1.0)
        assert r["n_cells_event"] > 0
    assert np.isnan(pick(rows, split="tune")[0]["mase_event"])


def test_perfect_forecast_has_zero_error_and_zero_scale_is_excluded():
    cfg = cfg_small()
    load = np.random.default_rng(1).gamma(2.0, 3.0, size=(300, 4))
    load[:, 3] = 5.0
    table = np.repeat(load[:, None, :], len(cfg["horizons"]), axis=1).astype(np.float32)
    for j, h in enumerate(cfg["horizons"]):
        table[:300 - h, j, :] = load[h:]
    events = events_frame([(0, 210, 215, "eval", "abrupt")])
    rows = accuracy_rows(cfg, "S", "oracle", table, load, events, TRAIN_END, TUNE_END)
    for r in rows:
        assert r["mase_all"] == pytest.approx(0.0, abs=1e-5) and r["smape_all"] == pytest.approx(0.0, abs=1e-5)
        assert r["n_zero_scale"] == 1


def test_early_warning_planted_recall_lead_precision():
    cfg = cfg_small()
    load = np.full((300, 3), 10.0)
    load[220:226, 1] = 100.0
    table = np.full((300, len(cfg["horizons"]), 3), 10.0, dtype=np.float32)
    table[217:220, :, 1] = 100.0
    table[250, :, 2] = 100.0
    events = events_frame([(1, 220, 225, "eval", "abrupt")])
    rows = early_warning_rows(cfg, "S", "planted", table, load, events, TRAIN_END, TUNE_END)
    top = pick(rows, split="eval", onset_class="all")[0]
    assert top["n_events"] == 1 and top["recall"] == 1.0 and top["lead_median"] == 3.0
    assert top["n_alarm_cells"] == 4 and top["precision"] == pytest.approx(0.75)
    quiet = early_warning_rows(cfg, "S", "quiet", np.full_like(table, 10.0), load, events, TRAIN_END, TUNE_END)
    assert pick(quiet, split="eval", onset_class="all")[0]["recall"] == 0.0


@pytest.mark.parametrize("kind", ["ridge", "lightgbm"])
def test_tune_forecaster_is_deterministic_and_exhaustive(kind):
    cfg = cfg_small()
    load = np.random.default_rng(3).gamma(2.0, 3.0, size=(300, 6))
    a, best_a = tune_forecaster(cfg, "S", kind, load, TRAIN_END, TUNE_END, 60, 1)
    b, best_b = tune_forecaster(cfg, "S", kind, load, TRAIN_END, TUNE_END, 60, 1)
    grid = cfg["ridge_alpha"] if kind == "ridge" else cfg["lightgbm_leaves"]
    assert [r["hp"] for r in a] == list(grid) and 0 <= best_a < len(grid)
    assert best_a == best_b and json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert all(len(r["mase_by_horizon"]) == len(cfg["horizons"]) for r in a)