from pathlib import Path

import numpy as np

from src import detect, injector
from src.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_injector_recovery():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    w_b = cfg["W_b"]
    n_events = 4
    gap = w_b + cfg["injection_dur_max"] + cfg["injection_rise_max"]
    n_steps = w_b + n_events * (gap + 10) + cfg["injection_dur_max"]
    rng = np.random.default_rng(5)
    load = 1000 * (1 + 0.01 * rng.standard_normal((n_steps, 6)))
    floor = cfg["eps"] * load.mean()
    out, table = injector.inject_spikes(
        load, dataset="TEST", master_seed=cfg["master_seed"], n_events=n_events,
        amp_min=cfg["injection_amplitude_min"], amp_max=cfg["injection_amplitude_max"],
        rise_min=cfg["injection_rise_min"], rise_max=cfg["injection_rise_max"],
        dur_min=cfg["injection_dur_min"], dur_max=cfg["injection_dur_max"],
        hot_min=cfg["injection_hot_min"], hot_max=cfg["injection_hot_max"],
        w_b=w_b, floor=floor, start=w_b, label=cfg["inject_label"])
    events = detect.detect_events(
        out, dataset="TEST", r=cfg["r"], w_b=w_b, s=cfg["s"], floor=floor,
        rise_lo=cfg["rise_lo"], rise_hi=cfg["rise_hi"], train_end=w_b, tune_end=w_b)
    assert len(events) == len(table)
    for _, row in table.iterrows():
        hits = events[(events["partition"] == row["partition"])
                      & ((events["onset"] - row["onset"]).abs() <= cfg["inject_tolerance"])]
        assert len(hits) == 1
    again, same = injector.inject_spikes(
        load, dataset="TEST", master_seed=cfg["master_seed"], n_events=n_events,
        amp_min=cfg["injection_amplitude_min"], amp_max=cfg["injection_amplitude_max"],
        rise_min=cfg["injection_rise_min"], rise_max=cfg["injection_rise_max"],
        dur_min=cfg["injection_dur_min"], dur_max=cfg["injection_dur_max"],
        hot_min=cfg["injection_hot_min"], hot_max=cfg["injection_hot_max"],
        w_b=w_b, floor=floor, start=w_b, label=cfg["inject_label"])
    assert np.array_equal(out, again)
    assert table.equals(same)