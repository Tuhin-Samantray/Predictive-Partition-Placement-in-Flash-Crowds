"""Q5 forecast-only cross-data tables (A37).

python scripts/forecast_cross.py --config configs/default.yaml --out tables/forecast_cross.csv
Rolling rows: the existing Phase 6 tables, all forecasters, in-dataset. Fixed rows: ridge and lightgbm fitted once on the
source training prefix (source tuned hyperparameter), applied to the target evaluation span, 2 x 2 sources and targets.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.cross_data import fixed_model_predictions, metric_rows, rolling_predictions  # noqa: E402
from src.forecasters import LEARNED, day_steps  # noqa: E402
from src.replicates import load_dataset, load_forecast  # noqa: E402

DATASETS = ("D1", "D2")


def best_hp(cfg, dataset, kind):
    path = Path(cfg["data_root"]) / cfg["processed_dirname"] / cfg["forecast_dirname"] / ("%s_%s_tuning.json" % (dataset, kind))
    return json.loads(path.read_text(encoding="utf-8"))["best_hp"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(args.out)
    if out.exists():
        parser.error("refusing to overwrite " + str(out))
    data = {ds: load_dataset(cfg, ds) for ds in DATASETS}
    day = {ds: day_steps(cfg, ds) for ds in DATASETS}
    rows = []
    for ds in DATASETS:
        for kind in cfg["forecaster_names"]:
            meta = {"protocol": "rolling", "source": ds, "target": ds, "forecaster": kind, "hp": ""}
            pred = rolling_predictions(cfg, load_forecast(cfg, ds, kind), data[ds])
            rows += metric_rows(cfg, data[ds], day[ds], pred, meta)
            print("rolling", ds, kind, flush=True)
    for src in DATASETS:
        for tgt in DATASETS:
            for kind in LEARNED:
                hp = best_hp(cfg, src, kind)
                start = time.perf_counter()
                pred = fixed_model_predictions(cfg, data[src], data[tgt], kind, hp, day[src], day[tgt], cfg["workers"])
                meta = {"protocol": "fixed", "source": src, "target": tgt, "forecaster": kind, "hp": hp}
                rows += metric_rows(cfg, data[tgt], day[tgt], pred, meta)
                print("fixed", src, "->", tgt, kind, "hp", hp, "seconds %.1f" % (time.perf_counter() - start), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out, index=False, lineterminator="\n")
    fixed = frame[(frame["protocol"] == "fixed") & (frame["group"] == "all")]
    print(fixed.groupby(["source", "target", "forecaster"])["mase"].mean().to_string())
    print("wrote", out, "rows", len(frame))


if __name__ == "__main__":
    main()
