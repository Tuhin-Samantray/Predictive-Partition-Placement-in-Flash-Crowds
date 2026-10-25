import yaml

from src.tuned_config import best_reactive, render_tuned


def _s(policy, dataset, damage, params, forecaster=None, moves=3.0):
    s = {"type": "summary", "dataset": dataset, "policy": policy, "best_params": params,
         "best_mean_damage": damage, "best_mean_moves": moves, "n_replicates": 10,
         "n_configs": 12, "grid_size": 12}
    if forecaster is not None:
        s["forecaster"] = forecaster
    return s


def _all():
    out = []
    for ds, d1, d2 in (("D1", 2.0, 3.0), ("D2", 5.0, 4.0)):
        out += [_s("P1", ds, d1, {"ewma_alpha": 0.8, "theta_hi": 0.9, "cooldown": 5}),
                _s("P2", ds, d2, {"ewma_alpha": 0.3, "hysteresis": 0.1, "cooldown": 0}),
                _s("P3", ds, 1.5, {"hysteresis": 0.0, "cooldown": 15}, "ridge"),
                _s("P3", ds, 1.6, {"hysteresis": 0.1, "cooldown": 0}, "naive"),
                _s("P4", ds, 1.4, {"hysteresis": 0.2, "cooldown": 5}, "lightgbm"),
                _s("P5", ds, 1.0, {"hysteresis": 0.0, "cooldown": 0})]
    return out


def test_best_reactive_is_per_dataset_lower_damage():
    s = _all()
    assert best_reactive(s, "D1")[0]["policy"] == "P1"
    assert best_reactive(s, "D2")[0]["policy"] == "P2"


def test_rendered_yaml_parses_and_every_leaf_is_commented():
    text = render_tuned(_all(), ("D1", "D2"))
    cfg = yaml.safe_load(text)
    assert cfg["best_reactive"] == {"D1": "P1", "D2": "P2"}
    assert cfg["tuned"]["D1"]["P3_ridge"] == {"hysteresis": 0.0, "cooldown": 15}
    assert cfg["tuned"]["D2"]["P4"]["forecaster"] == "lightgbm"
    for line in text.splitlines()[1:]:
        if not line.rstrip().endswith(":"):
            assert "# [design]" in line, line


def test_render_is_deterministic_and_order_independent():
    s = _all()
    assert render_tuned(s, ("D1", "D2")) == render_tuned(list(reversed(s)), ("D1", "D2"))
