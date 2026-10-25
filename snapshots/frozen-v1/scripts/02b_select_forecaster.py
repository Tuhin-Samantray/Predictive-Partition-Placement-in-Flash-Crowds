"""Apply the pre-registered P4 forecaster rule (A13) and write the choice per dataset.

Usage: python scripts/02b_select_forecaster.py --config configs/default.yaml --out results/tune/forecaster_choice.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, split_points  # noqa: E402
from src.forecast_select import choose_forecaster  # noqa: E402

DATASETS = ("D1", "D2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    base = Path(cfg["data_root"]) / cfg["processed_dirname"]
    result = {}
    for dataset in DATASETS:
        load = np.load(base / (dataset + "_L.npy"), mmap_mode="r")
        train_end, tune_end = split_points(cfg, dataset, load.shape[0])
        best, scores = choose_forecaster(cfg, dataset, load, train_end, tune_end)
        result[dataset] = {"choice": best,
                           "tune_mase": {n: s[0] for n, s in scores.items()},
                           "tune_mase_by_horizon": {n: s[1] for n, s in scores.items()}}
        for name, (obj, _per_h) in scores.items():
            print("%s %-15s tune MASE %.6f%s" % (dataset, name, obj, "  <- P4" if name == best else ""))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
