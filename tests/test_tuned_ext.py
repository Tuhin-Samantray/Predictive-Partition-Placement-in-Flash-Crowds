import pytest
import yaml

from src.tuned_ext import render_tuned_ext


def summary(ds, params, policy="P4a"):
    return {"dataset": ds, "policy": policy, "forecaster": "lightgbm", "best_params": params,
            "best_mean_damage": 1.5, "n_replicates": 35, "n_configs": 12, "grid_size": 12}


def test_render_roundtrips_and_every_leaf_is_tagged():
    text = render_tuned_ext([summary("D2", {"hysteresis": 0.1, "cooldown": 15}),
                             summary("D1", {"hysteresis": 0.0, "cooldown": 5})], ("D1", "D2"))
    data = yaml.safe_load(text)["tuned_ext"]
    assert data["D1"]["P4a"] == {"forecaster": "lightgbm", "cooldown": 5, "hysteresis": 0.0}
    assert data["D2"]["P4a"] == {"forecaster": "lightgbm", "cooldown": 15, "hysteresis": 0.1}
    for line in text.splitlines()[1:]:
        if not line.rstrip().endswith(":"):
            assert "# [" in line, line


def test_missing_or_duplicate_summary_is_an_error():
    one = summary("D1", {"hysteresis": 0.0, "cooldown": 5})
    with pytest.raises(ValueError):
        render_tuned_ext([one], ("D1", "D2"))
    with pytest.raises(ValueError):
        render_tuned_ext([one, one], ("D1",))


def test_other_policies_are_ignored():
    p4 = summary("D1", {"hysteresis": 0.5, "cooldown": 0}, policy="P4")
    p4a = summary("D1", {"hysteresis": 0.0, "cooldown": 5})
    data = yaml.safe_load(render_tuned_ext([p4, p4a], ("D1",)))["tuned_ext"]
    assert data["D1"]["P4a"]["hysteresis"] == 0.0
