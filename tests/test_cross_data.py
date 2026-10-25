import numpy as np
import pandas as pd

from src.accuracy import accuracy_rows
from src.config import load_config
from src.cross_data import fixed_model_predictions, group_masks, metric_rows, rolling_predictions
from src.forecasters import build_table
from src.replicates import DataSet


def small(seed, n_steps=700, n_part=6):
    rng = np.random.default_rng(seed)
    load = rng.gamma(2.0, 1.0, size=(n_steps, n_part)) * (1 + np.sin(np.arange(n_steps) / 20.0))[:, None]
    events = pd.DataFrame({"partition": [1, 3], "onset": [450, 600], "end": [470, 640], "split": ["eval", "eval"]})
    return DataSet("DX", load, events, 300, 450, n_steps)


def test_groups_partition_the_cells():
    cfg = load_config("configs/default.yaml")
    data = small(1)
    day = 100
    in_event = np.zeros(data.load.shape, dtype=bool)
    in_event[460:480, 1] = True
    for h in cfg["horizons"]:
        masks = group_masks(in_event, data.tune_end, data.n_steps, h, day)
        n_all = int(masks["all"].sum())
        assert int(masks["event"].sum()) + int(masks["nonevent"].sum()) == n_all
        days = [m for k, m in masks.items() if k.startswith("day")]
        assert sum(int(m.sum()) for m in days) == n_all
        assert not np.logical_and.reduce([m.any() for m in days]) or len(days) >= 1


def test_rolling_rows_reproduce_the_phase6_accuracy_rows():
    cfg = load_config("configs/default.yaml")
    data = small(2)
    table = build_table(cfg, "DX", "ewma", data.load, data.train_end, 100)
    ref = [r for r in accuracy_rows(cfg, "DX", "ewma", table, data.load, data.events, data.train_end, data.tune_end)
           if r["split"] == "eval"]
    got = metric_rows(cfg, data, 100, rolling_predictions(cfg, table, data), {"protocol": "rolling"})
    for r in ref:
        a = [g for g in got if g["horizon"] == r["horizon"] and g["group"] == "all"][0]
        e = [g for g in got if g["horizon"] == r["horizon"] and g["group"] == "event"][0]
        assert abs(a["mase"] - r["mase_all"]) < 1e-12 and a["n_cells"] == r["n_cells_all"]
        assert abs(e["mase"] - r["mase_event"]) < 1e-12 and e["n_cells"] == r["n_cells_event"]


def test_fixed_model_transfer_runs_and_is_finite():
    cfg = load_config("configs/default.yaml")
    src, tgt = small(3), small(4)
    pred = fixed_model_predictions(cfg, src, tgt, "ridge", 1.0, 100, 100, 1)
    assert [p.shape[0] for p in pred] == [tgt.n_steps - tgt.tune_end - h for h in cfg["horizons"]]
    rows = metric_rows(cfg, tgt, 100, pred, {"protocol": "fixed"})
    assert all(np.isfinite(r["mase"]) for r in rows if r["n_mase_cells"] > 0)
    assert {"all", "event", "nonevent", "day0"} <= {r["group"] for r in rows}
