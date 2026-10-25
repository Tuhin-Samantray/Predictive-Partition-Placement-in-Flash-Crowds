import json

import pytest

from small_world import small
from src.policies import TUNABLE
from src.replicates import tune_replicates
from src.tuning import plateau_gain, tune_policy


def test_budget_and_fairness():
    cfg, data = small()
    reps = tune_replicates(cfg, data)
    assert len(reps) == cfg["tune_replicates"]
    summaries = {}
    for name in ("P1", "P2"):
        rows, summary, timing = tune_policy(cfg, name, data.name, reps)
        assert len(rows) == cfg["tuning_budget"] == summary["n_configs"] == len(timing)
        assert len({json.dumps(r["params"], sort_keys=True) for r in rows}) == len(rows)
        curve = [r["best_so_far"] for r in rows]
        assert all(b <= a for a, b in zip(curve, curve[1:]))
        assert all(len(r["damages"]) == len(reps) for r in rows)
        assert set(rows[0]["params"]) == set(TUNABLE[name])
        assert rows[summary["best_iter"]]["params"] == summary["best_params"]
        summaries[name] = summary
    assert summaries["P1"]["event_ids"] == summaries["P2"]["event_ids"]
    assert summaries["P1"]["n_replicates"] == summaries["P2"]["n_replicates"]


def test_tuner_deterministic():
    cfg, data = small()
    reps = tune_replicates(cfg, data)
    rows_a, sum_a, _ = tune_policy(cfg, "P2", data.name, reps)
    rows_b, sum_b, _ = tune_policy(cfg, "P2", data.name, reps)
    assert json.dumps(rows_a, sort_keys=True) == json.dumps(rows_b, sort_keys=True)
    assert json.dumps(sum_a, sort_keys=True) == json.dumps(sum_b, sort_keys=True)


def test_plateau_gain_planted():
    assert plateau_gain([10, 8, 6, 4, 4, 4, 4, 4], 0.25) == 0.0
    assert abs(plateau_gain([10, 9, 8, 7, 6, 5, 4, 2], 0.25) - 0.6) < 1e-12
    assert plateau_gain([0.0, 0.0, 0.0, 0.0], 0.25) == 0.0


def test_grid_smaller_than_budget_raises():
    cfg, data = small()
    cfg["tuning_budget"] = 1000
    with pytest.raises(ValueError):
        tune_policy(cfg, "P1", data.name, tune_replicates(cfg, data))