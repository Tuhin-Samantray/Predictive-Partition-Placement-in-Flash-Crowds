import pytest
import yaml

from src.tuned_layers import render_tuned_layers


def summary(ds, policy, params, forecaster=None):
    s = {"dataset": ds, "policy": policy, "best_params": params, "best_mean_damage": 1.5, "n_replicates": 35,
         "n_configs": 12, "grid_size": 12, "mode": "coupled", "h_net": 2.0, "phi": 1.0}
    if forecaster is not None:
        s["forecaster"] = forecaster
    return s


def full(ds):
    return [summary(ds, "P1", {"cooldown": 5, "ewma_alpha": 0.1, "theta_hi": 1.0}),
            summary(ds, "P3", {"cooldown": 0, "hysteresis": 0.5}, "lightgbm"),
            summary(ds, "P5", {"cooldown": 15, "hysteresis": 0.0, "oracle_window": 10}),
            summary(ds, "P6", {"cooldown": 15, "hysteresis": 0.0, "psi": 0.2}, "lightgbm")]


def test_render_roundtrips_and_every_leaf_is_tagged():
    text = render_tuned_layers(full("D2") + full("D1"), ("D1", "D2"))
    data = yaml.safe_load(text)["tuned_layers"]
    assert set(data) == {"D1", "D2"}
    assert data["D1"]["P3"] == {"forecaster": "lightgbm", "cooldown": 0, "hysteresis": 0.5}
    assert data["D2"]["P6"] == {"forecaster": "lightgbm", "cooldown": 15, "hysteresis": 0.0, "psi": 0.2}
    assert data["D1"]["P5"]["oracle_window"] == 10
    assert "forecaster" not in data["D1"]["P1"] and "forecaster" not in data["D1"]["P5"]
    for line in text.splitlines()[1:]:
        if not line.rstrip().endswith(":"):
            assert "# [" in line, line


def test_missing_or_duplicate_summary_is_an_error():
    rows = full("D1")
    with pytest.raises(ValueError):
        render_tuned_layers(rows[:3], ("D1",))
    with pytest.raises(ValueError):
        render_tuned_layers(rows + [rows[0]], ("D1",))


def test_other_policies_are_ignored():
    rows = full("D1") + [summary("D1", "P4", {"cooldown": 0, "hysteresis": 0.0}, "lightgbm")]
    data = yaml.safe_load(render_tuned_layers(rows, ("D1",)))["tuned_layers"]
    assert set(data["D1"]) == {"P1", "P3", "P5", "P6"}
