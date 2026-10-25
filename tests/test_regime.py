import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.regime import CONSTANT, FEW, NO_CI, NO_R, regime_must, regime_q2

BEST = {"D1": "P1", "D2": "P1"}
VARIANTS = {"D1": "P3_naive", "D2": "P3_ewma"}
SHARE = 0.05
BOOT = 999
LEVEL = 0.95
MIN_EVENTS = 24
SEED = 42
BIG = 8.0
TINY = 0.2


def make_q2(r_fn, tax_fn=lambda bucket, e: BIG, events=8, reps=30):
    """Q2-shaped rows: replicate j uses event j mod events of its bucket, as in the real rows."""
    rows = []
    for ds, var in (("D1_D3", "P3_naive"), ("D2_D3", "P3_ewma")):
        for b in range(1, 6):
            bucket = "rise%d" % b
            for m in (1, 2, 5, 10):
                for j in range(reps):
                    e = j % events
                    tax = tax_fn(bucket, e)
                    r = r_fn(b, m, e)
                    key = dict(dataset=ds, onset_class=bucket, m=m, headroom=1.6, replicate=j,
                               event_id="%s-%s-%d" % (ds, bucket, e))
                    rows.append(dict(key, variant="P1", damage=10.0))
                    rows.append(dict(key, variant="P5", damage=10.0 - tax))
                    rows.append(dict(key, variant=var, damage=10.0 - r * tax))
    return pd.DataFrame(rows)


def q2_map(must):
    return regime_q2(must, BEST, VARIANTS, SHARE, BOOT, LEVEL, MIN_EVENTS, SEED)


def pick(table, dataset, bucket, m):
    row = table[(table["dataset"] == dataset) & (table["onset_class"] == bucket) & (table["m"] == m)]
    assert len(row) == 1
    return row.iloc[0]


def test_q2_map_planted_values_counts_and_policy_labels():
    t = q2_map(make_q2(lambda b, m, e: 0.1 * b + 0.01 * e))
    assert len(t) == 2 * 5 * 4 and (t["headroom"] == 1.6).all()
    row = pick(t, "D1_D3", "rise3", 5)
    assert row["policy"] == "P3_naive" and pick(t, "D2_D3", "rise3", 5)["policy"] == "P3_ewma"
    assert row["median_R"] == pytest.approx(0.3 + 0.035, abs=1e-9)
    assert row["n_events"] == 8 and row["n_events_defined"] == 8 and row["n_replicates"] == 30
    assert row["share_defined"] == 1 and row["median_tax"] == pytest.approx(8.0)
    assert row["note"] == FEW
    assert np.isfinite(row["ci_low"]) and np.isfinite(row["ci_high"]) and row["ci_low"] <= row["ci_high"]
    assert (t["n_events"] == 8).all()


def test_q2_cell_without_defined_r_is_kept_and_marked():
    t = q2_map(make_q2(lambda b, m, e: 0.5, lambda bucket, e: TINY if bucket == "rise1" else BIG))
    row = pick(t, "D1_D3", "rise1", 2)
    assert row["n_events_defined"] == 0 and row["share_defined"] == 0 and np.isnan(row["median_R"])
    assert np.isnan(row["ci_low"]) and NO_R in row["note"] and NO_CI in row["note"]
    assert row["median_tax"] == pytest.approx(TINY)
    assert pick(t, "D1_D3", "rise2", 2)["n_events_defined"] == 8


def test_q2_partly_defined_cell_counts_events_and_replicates():
    t = q2_map(make_q2(lambda b, m, e: 0.2 + 0.05 * e, lambda bucket, e: TINY if (bucket == "rise2" and e == 0) else BIG))
    row = pick(t, "D2_D3", "rise2", 10)
    assert row["n_events"] == 8 and row["n_events_defined"] == 7
    assert row["share_defined"] == pytest.approx(26 / 30)
    assert row["median_R"] == pytest.approx(0.2 + 0.05 * 4)


def test_constant_r_has_no_interval_and_a_note():
    row = pick(q2_map(make_q2(lambda b, m, e: 0.4)), "D1_D3", "rise4", 1)
    assert row["median_R"] == pytest.approx(0.4) and np.isnan(row["ci_low"]) and CONSTANT in row["note"]


def make_must(r_fn, n=30):
    """Must-grid shaped rows: the replicate field is the event index within its class."""
    rows = []
    for ds in ("D1", "D2"):
        for cls in ("abrupt", "ramped"):
            for i in range(n):
                for m in (1, 2, 5, 10):
                    for h in (1.3, 1.6, 2.0):
                        key = dict(dataset=ds, onset_class=cls, m=m, headroom=h, replicate=i,
                                   event_id="%s-%s-%d" % (ds, cls, i))
                        rows.append(dict(key, variant="P1", damage=10.0))
                        rows.append(dict(key, variant="P5", damage=2.0))
                        for var in ("P3_lightgbm", "P4"):
                            rows.append(dict(key, variant=var, damage=10.0 - r_fn(var, cls, i, m, h) * BIG))
    return pd.DataFrame(rows)


def must_map(must):
    head = {"D1": "P3_lightgbm", "D2": "P3_lightgbm"}
    p4 = {"D1": "P4", "D2": "P4"}
    return regime_must(must, BEST, [head, p4], SHARE, BOOT, LEVEL, MIN_EVENTS, SEED)


def planted(var, cls, i, m, h):
    if var == "P4":
        return 0.1 + 0.1 * i / 29
    return (0.3 if cls == "abrupt" else 0.6) + 0.2 * i / 29


def test_must_map_planted_values_one_block_per_policy():
    t = must_map(make_must(planted))
    assert len(t) == 2 * 2 * 2 * 4 * 3 and set(t["policy"]) == {"P3_lightgbm", "P4"}
    cell = t[(t["dataset"] == "D1") & (t["onset_class"] == "ramped") & (t["m"] == 5) & (t["headroom"] == 1.6)]
    by = cell.set_index("policy")
    assert by.loc["P3_lightgbm", "median_R"] == pytest.approx(0.7, abs=1e-9) and by.loc["P4", "median_R"] == pytest.approx(0.15, abs=1e-9)
    assert (cell["n_events"] == 30).all() and (cell["n_events_defined"] == 30).all() and (cell["note"] == "").all()
    assert (cell["share_defined"] == 1).all() and cell["ci_low"].notna().all() and (cell["ci_low"] <= cell["ci_high"]).all()
    assert (t["note"] == "").all()


def test_tables_reproduce_exactly():
    must = make_must(planted)
    pd.testing.assert_frame_equal(must_map(must), must_map(must))
    q2 = make_q2(lambda b, m, e: 0.1 * b + 0.01 * e)
    pd.testing.assert_frame_equal(q2_map(q2), q2_map(q2))


def test_stages_are_registered():
    spec = importlib.util.spec_from_file_location("analyze", Path(__file__).resolve().parents[1] / "scripts" / "05_analyze.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert {"regime_map_q2", "regime_map_must"} <= set(mod.STAGES)
