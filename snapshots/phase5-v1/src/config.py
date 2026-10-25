"""Configuration access (G1): every number lives in configs/default.yaml."""
from pathlib import Path

import yaml


def load_config(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))


def step_seconds(cfg, dataset):
    return cfg["step_seconds"] if dataset == "D1" else cfg["d2_step_seconds"]


def split_points(cfg, dataset, n_steps):
    """Return (train_end, tune_end); evaluation runs from tune_end to the end."""
    if dataset == "D1":
        train_end = cfg["d1_train_days"] * cfg["azure_mins_per_day"]
        tune_end = train_end + cfg["d1_tune_days"] * cfg["azure_mins_per_day"]
    else:
        train_end = int(cfg["d2_split_fractions"][0] * n_steps)
        tune_end = train_end + int(cfg["d2_split_fractions"][1] * n_steps)
    return train_end, tune_end