from pathlib import Path

import numpy as np

from src.config import load_config
from src.layer_runs import load_cross, run_replicate_layers
from src.replicates import forecast_window, load_dataset, load_forecast, make_replicate, run_replicate, select_events

ROOT = Path(__file__).resolve().parents[1]
P1 = {"ewma_alpha": 0.3, "theta_hi": 0.9, "cooldown": 5}
P3 = {"hysteresis": 0.1, "cooldown": 5}


def case(name):
    cfg = load_config(str(ROOT / "configs" / "default.yaml"))
    data = load_dataset(cfg, name)
    rep = make_replicate(cfg, data, select_events(cfg, data, "tune", 1)[0], 0, cfg["headroom"])
    return cfg, data, rep, load_cross(cfg, data)


def test_layers_off_reproduces_run_replicate_on_d2():
    cfg, data, rep, cross = case("D2")
    assert (cross.rel == 1).all()
    ref = run_replicate(cfg, rep, "P1", P1)
    assert ref == run_replicate_layers(cfg, rep, cross, "P1", P1, mode="const_m", nic=False)
    fw = forecast_window(load_forecast(cfg, "D2", "lightgbm"), rep)
    ref = run_replicate(cfg, rep, "P3", P3, forecast=fw)
    assert ref == run_replicate_layers(cfg, rep, cross, "P3", P3, forecast=fw, mode="const_m", nic=False)


def test_layers_on_is_deterministic_conserves_and_attributes():
    for name in ("D1", "D2"):
        cfg, data, rep, cross = case(name)
        a = run_replicate_layers(cfg, rep, cross, "P1", P1)
        b = run_replicate_layers(cfg, rep, cross, "P1", P1)
        assert a == b and a["conservation_ok"]
        if a["attr_violating_steps"] > 0:
            assert abs(a["attr_cpu_share"] + a["attr_net_share"] - 1) < 1e-12
        else:
            assert a["attr_cpu_share"] == 0 and a["attr_net_share"] == 0


def test_cross_data_follows_the_a24_formulas():
    cfg, data, rep, cross = case("D1")
    train = np.asarray(data.load[: data.train_end])
    expected = cfg["h_net"] * np.asarray(cross.nbytes[: data.train_end]).sum(axis=1).mean() / cfg["n_nodes"]
    assert np.isclose(cross.link, expected)
    assert np.isclose(cross.sizes.mean(), cfg["migration_steps_m0"] * cfg["rho"] * cross.link)
    assert cross.rel.shape == (cfg["n_partitions"],) and (cross.rel > 0).all() and train.shape[0] == data.train_end


def test_the_bandwidth_tier_scales_link_and_sizes_together():
    cfg, data, rep, cross = case("D2")
    lo = load_cross(cfg, data, h_net=cfg["h_net_levels"][0])
    hi = load_cross(cfg, data, h_net=cfg["h_net_levels"][-1])
    assert np.isclose(lo.link / hi.link, cfg["h_net_levels"][0] / cfg["h_net_levels"][-1])
    assert np.allclose(lo.sizes / lo.link, hi.sizes / hi.link)
