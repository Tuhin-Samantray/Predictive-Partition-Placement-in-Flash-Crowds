from pathlib import Path

import yaml

from src.analysis_data import load_folder, pair_rows, select_variant
from src.config import load_config

COUNTS = {"must": 14400, "q1": 7920, "q2": 3600, "q3": 5760, "q4": 1680, "q6": 1200, "q8": 1440,
          "q11_rv0": 2340, "q11_rv123": 2760}
KEYS = ["dataset", "onset_class", "m", "headroom", "replicate"]


def test_result_folders_hold_the_registered_row_counts():
    cfg = load_config("configs/default.yaml")
    assert set(cfg["result_folders"]) == set(COUNTS)
    for name, folder in cfg["result_folders"].items():
        assert len(load_folder(folder)) == COUNTS[name], name


def test_must_rows_pair_the_best_reactive_policy_with_the_oracle_event_for_event():
    cfg = load_config("configs/default.yaml")
    best = yaml.safe_load(Path("configs/tuned.yaml").read_text(encoding="utf-8-sig"))["best_reactive"]
    must = load_folder(cfg["result_folders"]["must"])
    reactive = select_variant(must, best)
    oracle = must[must["variant"] == "P5"]
    out = pair_rows(reactive, oracle, KEYS, "damage")
    assert len(out) == len(oracle) == len(reactive) and out["damage_a"].notna().all() and out["damage_b"].notna().all()
