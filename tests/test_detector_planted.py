from pathlib import Path

import numpy as np

from src import detect
from src.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_detector_planted():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    r, w_b, s = cfg["r"], cfg["W_b"], cfg["s"]
    base = 100.0
    onset = 200
    load = np.full((400, 5), base)
    load[onset:onset + 30, 2] = base * (r + 2)          # abrupt spike: must be found
    load[100:100 + s - 1, 1] = base * (r + 2)           # shorter than s: must be ignored
    ramp = base * (1 + np.arange(10))
    load[250:260, 3] = ramp                             # ramp then plateau: found, rise > 0
    load[260:285, 3] = ramp[-1]
    events = detect.detect_events(
        load, dataset="TEST", r=r, w_b=w_b, s=s, floor=cfg["eps"] * load.mean(),
        rise_lo=cfg["rise_lo"], rise_hi=cfg["rise_hi"], train_end=100, tune_end=onset)
    assert sorted(events["partition"].tolist()) == [2, 3]
    abrupt = events[events["partition"] == 2].iloc[0]
    assert abrupt["onset"] == onset
    assert abrupt["peak"] == base * (r + 2)
    assert abrupt["rise_time"] == 0
    assert abrupt["split"] == "eval"
    ramped = events[events["partition"] == 3].iloc[0]
    assert ramped["rise_time"] >= 1
    assert ramped["duration"] >= s