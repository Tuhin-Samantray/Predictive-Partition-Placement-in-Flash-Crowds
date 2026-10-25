import re
from pathlib import Path

import numpy as np

from src.config import load_config
from src.d3 import load_d3
from src.replicates import eligible_events


def test_d3_events_buckets_and_eligibility():
    cfg = load_config("configs/default.yaml")
    for base in ("D1", "D2"):
        data = load_d3(cfg, base)
        ev = data.events
        assert len(ev) == cfg["injection_events"] and ev["event_id"].nunique() == len(ev)
        assert (ev["split"] == "eval").all() and (ev["onset"] >= data.tune_end).all()
        assert (ev["end"] == ev["onset"] + ev["duration"] - 1).all()
        sizes = ev.groupby("onset_class").size()
        assert len(sizes) == cfg["rise_buckets"] and sizes.nunique() == 1
        spans = ev.groupby("onset_class")["rise_steps"].agg(["min", "max"]).sort_index()
        assert (spans["max"].to_numpy()[:-1] <= spans["min"].to_numpy()[1:]).all()
        assert len(eligible_events(cfg, data, "eval")) == len(ev)


def tuned_damage(text):
    out, ds, var = {}, None, None
    for line in text.splitlines():
        if line.startswith("  D") and line.rstrip().endswith(":"):
            ds = line.strip()[:-1]
        elif line.startswith("    ") and not line.startswith("      ") and line.rstrip().endswith(":"):
            var = line.strip()[:-1]
        elif var and "mean damage" in line and (ds, var) not in out:
            out[(ds, var)] = float(re.search(r"mean damage ([0-9.eE+-]+)", line).group(1))
    return out


def test_q2_variant_is_the_lowest_tuned_damage():
    cfg = load_config("configs/default.yaml")
    damage = tuned_damage(Path("configs/tuned.yaml").read_text(encoding="utf-8"))
    for base, variant in cfg["q2_forecast_variants"].items():
        cands = {v: d for (ds, v), d in damage.items() if ds == base and (v.startswith("P3_") or v == "P4")}
        assert len(cands) == len(cfg["forecaster_names"]) + 1
        assert min(cands, key=cands.get) == variant


def test_d3_tables_are_causal_baselines():
    cfg = load_config("configs/default.yaml")
    folder = Path(cfg["data_root"]) / cfg["processed_dirname"] / cfg["forecast_dirname"]
    for base in ("D1", "D2"):
        data = load_d3(cfg, base)
        kind = cfg["q2_forecast_variants"][base].split("_", 1)[1]
        table = np.load(folder / ("%s_%s.npy" % (data.name, kind)), mmap_mode="r")
        assert table.shape == (data.n_steps, len(cfg["horizons"]), data.load.shape[1]) and table.dtype == np.float32
    naive = np.load(folder / "D1_D3_naive.npy", mmap_mode="r") if (folder / "D1_D3_naive.npy").exists() else None
    if naive is not None:
        d1 = load_d3(cfg, "D1")
        assert np.array_equal(np.asarray(naive[:, 0, :]), np.asarray(d1.load, dtype=np.float32))
