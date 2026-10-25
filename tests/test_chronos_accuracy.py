import importlib.util
import os
import stat
from pathlib import Path

import numpy as np
import pandas as pd

from src.accuracy_subset import eval_origins

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("chronos_accuracy", ROOT / "scripts" / "02d_chronos_accuracy.py")
ca = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ca)

N_STEPS, N_PART, N_HORIZONS, TRAIN_END, TUNE_END = 40, 5, 2, 12, 20


def _load():
    return np.random.default_rng(3).gamma(2.0, 2.0, size=(N_STEPS, N_PART)) + 1.0


def _table(load, origins):
    table = np.repeat(load.astype(np.float32)[:, None, :], N_HORIZONS, axis=1)
    for o in origins:
        table[o] = (load[o] * 1.5 + 1.0).astype(np.float32)[None, :]
    return table


def test_verify_accepts_a_correct_table():
    load = _load()
    origins = eval_origins(TUNE_END, N_STEPS, 4, [1, 2])
    problems, changed = ca.verify_table(_table(load, origins), load, origins, 7, N_HORIZONS)
    assert problems == [] and changed == origins.size


def test_verify_flags_missing_extra_shape_dtype_and_nonfinite():
    load = _load()
    origins = eval_origins(TUNE_END, N_STEPS, 4, [1, 2])
    good = _table(load, origins)
    missing = good.copy()
    missing[origins[1]] = load[origins[1]].astype(np.float32)[None, :]
    assert any("naive fill" in p for p in ca.verify_table(missing, load, origins, 7, N_HORIZONS)[0])
    extra = good.copy()
    extra[origins[0] + 1] = (load[origins[0] + 1] * 2.0).astype(np.float32)[None, :]
    assert any("outside" in p for p in ca.verify_table(extra, load, origins, 7, N_HORIZONS)[0])
    assert any("shape" in p for p in ca.verify_table(good[:-1], load, origins, 7, N_HORIZONS)[0])
    assert any("dtype" in p for p in ca.verify_table(good.astype(np.float64), load, origins, 7, N_HORIZONS)[0])
    bad = good.copy()
    bad[origins[0], 0, 0] = np.nan
    assert any("non-finite" in p for p in ca.verify_table(bad, load, origins, 7, N_HORIZONS)[0])


def test_score_covers_every_forecaster_on_the_identical_origins():
    load = _load()
    origins = eval_origins(TUNE_END, N_STEPS, 4, [1, 2])
    cfg = {"horizons": [1, 2], "smape_eps": 0.001, "warn_ratio": 3.0, "warn_baseline_window": 5}
    events = pd.DataFrame({
        "partition": [1, 3, 2], "onset": [25, 30, 34], "end": [27, 32, 36],
        "split": ["eval", "eval", "eval"], "onset_class": ["abrupt", "ramped", "abrupt"]})
    tables = {"naive": np.repeat(load.astype(np.float32)[:, None, :], N_HORIZONS, axis=1),
              "chronos": _table(load, origins)}
    acc, warn = ca.score(cfg, "X", tables, load, events, TRAIN_END, TUNE_END, origins)
    assert sorted({r["forecaster"] for r in acc}) == ["chronos", "naive"]
    assert len(acc) == 2 * N_HORIZONS
    assert {r["n_origins"] for r in acc if r["horizon"] == 1} == {int((origins + 1 < N_STEPS).sum())}
    assert sorted({r["forecaster"] for r in warn}) == ["chronos", "naive"]


def test_sanity_compares_only_phase6_forecasters_against_the_tolerance():
    rows = [{"dataset": "D1", "forecaster": "naive", "horizon": 1, "mase_all": 1.10},
            {"dataset": "D1", "forecaster": "chronos", "horizon": 1, "mase_all": 9.0}]
    phase6 = pd.DataFrame({"dataset": ["D1", "D1"], "forecaster": ["naive", "naive"], "split": ["eval", "tune"],
                           "horizon": [1, 1], "mase_all": [1.0, 5.0]})
    worst, count, ok = ca.sanity(rows, phase6, 0.2)
    assert count == 1 and abs(worst - 0.1) < 1e-12 and ok
    assert not ca.sanity(rows, phase6, 0.05)[2]
    assert not ca.sanity([rows[1]], phase6, 0.2)[2]


def test_write_readonly_and_hash(tmp_path):
    path = tmp_path / "out.csv"
    frame = pd.DataFrame({"a": [1, 2], "b": [3.5, 4.5]})
    ca.write_csv_readonly(frame, path)
    assert not os.access(path, os.W_OK)
    first = ca.file_sha256(path, 4)
    ca.write_csv_readonly(frame, path)
    assert ca.file_sha256(path, 4) == first
    os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
