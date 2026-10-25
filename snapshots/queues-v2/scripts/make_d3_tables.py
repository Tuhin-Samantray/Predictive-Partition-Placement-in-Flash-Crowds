"""Build the D3 forecast tables Q2 needs (A36): the baseline forecaster of config q2_forecast_variants per dataset.

python scripts/make_d3_tables.py --config configs/default.yaml
Tables go to data/processed/forecasts/<base>_<inject_label>_<kind>.npy; an existing table is kept.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.config import load_config  # noqa: E402
from src.d3 import load_d3  # noqa: E402
from src.forecasters import BASELINES, build_table, day_steps  # noqa: E402

DATASETS = ("D1", "D2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    folder = Path(cfg["data_root"]) / cfg["processed_dirname"] / cfg["forecast_dirname"]
    for base in DATASETS:
        kind = cfg["q2_forecast_variants"][base].split("_", 1)[1]
        if kind not in BASELINES:
            raise SystemExit("make_d3_tables builds baseline forecasters only, got " + kind)
        data = load_d3(cfg, base)
        path = folder / ("%s_%s.npy" % (data.name, kind))
        if path.exists():
            print(path.name, "exists, kept")
            continue
        table = build_table(cfg, data.name, kind, data.load, data.train_end, day_steps(cfg, base))
        expected = (data.n_steps, len(cfg["horizons"]), data.load.shape[1])
        if table.shape != expected or table.dtype != np.float32:
            raise SystemExit("bad table %s: %s %s, expected %s" % (path.name, table.shape, table.dtype, expected))
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as handle:
            np.save(handle, table)
        os.replace(tmp, path)
        print(path.name, table.shape)


if __name__ == "__main__":
    main()
