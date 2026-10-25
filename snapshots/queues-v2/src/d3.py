"""D3 loader (Q2, A36): injected-spike datasets as simulator-ready events with equal-count rise-time buckets."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import split_points
from src.replicates import DataSet

BUCKET_PREFIX = "rise"


def event_table(raw, n_buckets, prefix):
    """One row per injected event; buckets of equal event count ordered by rise_steps, then event number."""
    one = raw.groupby("event_id", sort=False).agg(onset=("onset", "first"), rise_steps=("rise_steps", "first"),
                                                  duration=("duration", "first")).reset_index()
    one["serial"] = one["event_id"].str.rsplit("-", n=1).str[1].astype(int)
    one = one.sort_values(["rise_steps", "serial"]).reset_index(drop=True)
    if len(one) % n_buckets:
        raise ValueError("%d events do not split into %d equal buckets" % (len(one), n_buckets))
    one["onset_class"] = [prefix + str(i * n_buckets // len(one) + 1) for i in range(len(one))]
    one["end"] = one["onset"] + one["duration"] - 1
    one["split"] = "eval"
    return one.drop(columns="serial")


def load_d3(cfg, base):
    root = Path(cfg["data_root"]) / cfg["processed_dirname"]
    name = base + "_" + cfg["inject_label"]
    load = np.load(root / (name + "_L.npy"), mmap_mode="r")
    events = event_table(pd.read_csv(root / (name + "_events.csv")), cfg["rise_buckets"], BUCKET_PREFIX)
    train_end, tune_end = split_points(cfg, base, load.shape[0])
    if (events["onset"] < tune_end).any():
        raise ValueError("injected events before the evaluation span in " + name)
    return DataSet(name, load, events, train_end, tune_end, load.shape[0])
