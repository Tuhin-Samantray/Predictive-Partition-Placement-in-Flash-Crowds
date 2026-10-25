import json
from pathlib import Path

import numpy as np
import yaml
from layers_world import PLANS, Scripted, run, world

from src.config import load_config
from src.metrics import compute_metrics
from src.replicates import load_dataset, make_replicate, run_replicate
from src.simulator import simulate

ROOT = Path(__file__).resolve().parents[1]
WANT = [("D1", "P0"), ("D1", "P1"), ("D2", "P2"), ("D2", "P5"), ("D1", "P5")]


def test_layers_off_equals_the_l1_simulator():
    w = world(1)
    ref = simulate(w["load"], w["node0"], w["mu"], m=3, kappa=0.3, budget=4, d=1, k=1,
                   policy=Scripted(PLANS), cooldown=2)
    lay = run(w, PLANS, mode="const_m", nic=False)
    for name in ("arrivals", "fg", "wait", "queue", "served", "node_hist"):
        assert np.array_equal(getattr(ref, name), getattr(lay, name)), name
    assert len(ref.moves) > 0 and ref.moves == lay.moves
    assert ref.dropped_moves == lay.dropped_moves
    kw = dict(slo=1, wasted_window=5, wait_quantile=0.99)
    assert compute_metrics(ref, w["load"], **kw) == compute_metrics(lay, w["load"], **kw)


def test_five_hashed_grid_rows_are_reproduced_with_layers_off():
    cfg = load_config(str(ROOT / "configs" / "default.yaml"))
    assert cfg["layers"]["enabled"] is False
    tuned = yaml.safe_load((ROOT / "configs" / "tuned.yaml").read_text(encoding="utf-8"))
    rows = {}
    for path in sorted((ROOT / "results").glob("grid_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line) if line else None
            if r and (r["dataset"], r["variant"]) in WANT and (r["dataset"], r["variant"]) not in rows:
                rows[(r["dataset"], r["variant"])] = r
        if len(rows) == len(WANT):
            break
    assert len(rows) == len(WANT), "grid rows missing in results/"
    data = {ds: load_dataset(cfg, ds) for ds in ("D1", "D2")}
    for (ds, variant), row in rows.items():
        events = data[ds].events
        ev = events[events["event_id"].astype(str) == row["event_id"]].iloc[0]
        rep = make_replicate(cfg, data[ds], ev, row["replicate"], row["headroom"])
        params = {} if variant == "P0" else dict(tuned["tuned"][ds][variant])
        met = run_replicate(cfg, rep, variant, params, m=row["m"])
        for key, value in met.items():
            assert row[key] == value, (ds, variant, key)
