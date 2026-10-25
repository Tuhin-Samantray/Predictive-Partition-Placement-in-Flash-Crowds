"""Build the learned D3 forecast tables RV0 needs (A42): lightgbm on each injected dataset.

python scripts/make_d3_learned_tables.py --config configs/default.yaml
Tables go to data/processed/forecasts/<base>_<inject_label>_<kind>.npy; an existing table is kept.
The hyperparameter is the tuned value of the source dataset (<base>_<kind>_tuning.json); nothing is re-tuned.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.config import load_config  # noqa: E402
from src.d3 import load_d3  # noqa: E402
from src.forecasters import LEARNED, build_table, day_steps  # noqa: E402

DATASETS = ("D1", "D2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    folder = Path(cfg["data_root"]) / cfg["processed_dirname"] / cfg["forecast_dirname"]
    for base in DATASETS:
        data = load_d3(cfg, base)
        for kind in cfg["d3_learned_forecasters"]:
            if kind not in LEARNED:
                raise SystemExit("make_d3_learned_tables builds learned forecasters only, got " + kind)
            path = folder / ("%s_%s.npy" % (data.name, kind))
            if path.exists():
                print(path.name, "exists, kept")
                continue
            tuning = json.loads((folder / ("%s_%s_tuning.json" % (base, kind))).read_text(encoding="utf-8"))
            if tuning["dataset"] != base or tuning["forecaster"] != kind:
                raise SystemExit("tuning file does not match " + base + " " + kind)
            table = build_table(cfg, data.name, kind, data.load, data.train_end, day_steps(cfg, base),
                                hp=tuning["best_hp"], n_jobs=cfg["workers"])
            expected = (data.n_steps, len(cfg["horizons"]), data.load.shape[1])
            if table.shape != expected or table.dtype != np.float32:
                raise SystemExit("bad table %s: %s %s, expected %s" % (path.name, table.shape, table.dtype, expected))
            tmp = path.with_name(path.name + ".tmp")
            with open(tmp, "wb") as handle:
                np.save(handle, table)
            os.replace(tmp, path)
            print(path.name, table.shape, "hp", tuning["best_hp"], flush=True)


if __name__ == "__main__":
    main()
