from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.replicates import DataSet

ROOT = Path(__file__).resolve().parents[1]


def real_cfg():
    return load_config(ROOT / "configs" / "default.yaml")


def small():
    """Real config with a few overrides, plus a synthetic 400-step dataset (train 100, tune 250)."""
    cfg = real_cfg()
    cfg.update(window_steps=60, n_nodes=4, n_partitions=24, tune_replicates=4, tuning_budget=6)
    rng = np.random.default_rng(5)
    load = rng.gamma(2.0, 3.0, size=(400, 24))
    load[130:140, 3] *= 10
    spec = [(20, "train"), (60, "train")]
    spec += [(o, "tune") for o in range(105, 250, 12)]
    spec += [(o, "eval") for o in (260, 300, 340)]
    events = pd.DataFrame({
        "event_id": ["S-%d" % i for i in range(len(spec))],
        "onset": [o for o, _ in spec],
        "end": [o + 5 for o, _ in spec],
        "split": [s for _, s in spec],
        "onset_class": ["ramped" if i % 2 else "abrupt" for i in range(len(spec))],
    })
    return cfg, DataSet("S", load, events, 100, 250, 400)