import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.maincomp import EXPLORATORY, F9, F10, HIGHER, LOWER, c11_table, c13_table
from src.stats_tools import holm_adjust

N = 30
BEST = {"D1": "P1", "D2": "P1"}
HEAD = {"D1": "P3_lightgbm", "D2": "P3_lightgbm"}
ALPHA = 0.05
BOOT = 999
LEVEL = 0.95
SEED = 42
DELTA = {"P1": lambda i: 0.0, "P0": lambda i: 20.0 + 0.1 * i, "P2": lambda i: 0.5 * (-1) ** i,
         "P3_lightgbm": lambda i: -0.5 - 0.01 * i, "P3_naive": lambda i: 0.3, "P4": lambda i: -1.0,
         "P5": lambda i: -3.0 - 0.02 * i}
MEAN_IDX = 14.5


def event_key(ds, cls, i):
    return dict(dataset=ds, onset_class=cls, replicate=i, event_id="%s-%s-%d" % (ds, cls, i))


def make_must(n=N):
    rows = []
    for ds in ("D1", "D2"):
        for cls in ("abrupt", "ramped"):
            for i in range(n):
                for m in (1, 5):
                    for h in (1.3, 1.6):
                        shift = 0.0 if (m == 5 and h == 1.6) else 100.0
                        for var, delta in DELTA.items():
                            rows.append(dict(event_key(ds, cls, i), m=m, headroom=h, variant=var,
                                             damage=10.0 + 0.1 * i + delta(i) + shift))
    return pd.DataFrame(rows)


def make_q1(n=N):
    rows = []
    for ds in ("D1", "D2"):
        for cls in ("abrupt", "ramped"):
            for i in range(n):
                base = 10.0 + 0.1 * i
                for level in (0.8, 0.9, 0.95):
                    rows.append(dict(event_key(ds, cls, i), m=5, headroom=1.6, variant="P4a", factor="AA6",
                                     param="margin_quantile", level=level, applied=level,
                                     damage=base + (-2.0 if level == 0.9 else 50.0)))
                rows.append(dict(event_key(ds, cls, i), m=5, headroom=1.6, variant="P4", factor="AA6",
                                 param="margin_quantile", level=0.9, applied=0.9, damage=base + 99.0))
                rows.append(dict(event_key(ds, cls, i), m=5, headroom=1.6, variant="P4a", factor="AA3",
                                 param="hysteresis", level=0.1, applied=0.1, damage=base + 77.0))
    return pd.DataFrame(rows)


def make_q3(n=N):
    rows = []
    for ds in ("D1", "D2"):
        for cls in ("abrupt", "ramped"):
            for i in range(n):
                for mode, phis in (("coupled", (0.01, 0.1, 1.0)), ("const_m", (None,))):
                    for h_net in (1.3, 2.0, 4.0):
                        for phi in phis:
                            hit = mode == "coupled" and h_net == 2.0 and phi == 1.0
                            for var, extra in (("P1", 0.0), ("P6", -2.0 if hit else 9.0)):
                                rows.append(dict(event_key(ds, cls, i), m=5, headroom=1.6, variant=var, mode=mode,
                                                 h_net=h_net, phi=phi, damage=50.0 + i + extra))
    return pd.DataFrame(rows)


def run11(must=None, q1=None, q3=None):
    return c11_table(make_must() if must is None else must, make_q1() if q1 is None else q1,
                     make_q3() if q3 is None else q3, BEST, HEAD, 5, 1.6, 0.9, 2.0, 1.0, ALPHA, BOOT, LEVEL, SEED)


def run13(must=None, q1=None):
    return c13_table(make_must() if must is None else must, make_q1() if q1 is None else q1, HEAD, 5, 1.6, 0.9,
                     ALPHA, BOOT, LEVEL, SEED)


def sel(t, policy, ds, cls):
    row = t[(t["policy"] == policy) & (t["dataset"] == ds) & (t["onset_class"] == cls)]
    assert len(row) == 1
    return row.iloc[0]


def test_c11_shape_and_planted_means():
    t = run11()
    f9 = t[t["family"] == F9]
    assert len(f9) == 28 and set(f9["policy"]) == {"P0", "P2", "P3", "P4", "P4a", "P5", "P6"}
    assert (t["n_events"] == N).all() and (t["n_replicates"] == N).all()
    expected = {"P0": 20.0 + 0.1 * MEAN_IDX, "P2": 0.0, "P3": -0.5 - 0.01 * MEAN_IDX, "P4": -1.0,
                "P4a": -2.0, "P5": -3.0 - 0.02 * MEAN_IDX, "P6": -2.0}
    for policy, value in expected.items():
        for ds in ("D1", "D2"):
            for cls in ("abrupt", "ramped"):
                assert sel(t, policy, ds, cls)["mean_diff"] == pytest.approx(value, abs=1e-9)
    row = sel(t, "P0", "D1", "abrupt")
    assert row["direction"] == HIGHER and row["cliffs_delta"] == pytest.approx(1.0) and row["p_raw"] < 1e-6
    assert sel(t, "P5", "D2", "ramped")["direction"] == LOWER and sel(t, "P5", "D2", "ramped")["cliffs_delta"] == pytest.approx(-1.0)
    assert sel(t, "P2", "D1", "ramped")["p_raw"] > 0.3


def test_c11_row_selection_and_references():
    t = run11()
    p4a = sel(t, "P4a", "D1", "abrupt")
    assert "0.9" in p4a["source"] and p4a["model"] == "layers off"
    assert p4a["mean_reference"] == pytest.approx(10.0 + 0.1 * MEAN_IDX)
    p6 = sel(t, "P6", "D2", "ramped")
    assert p6["mean_reference"] == pytest.approx(50.0 + MEAN_IDX) and "coupled" in p6["model"]
    assert p6["relative_diff"] == pytest.approx(-2.0 / (50.0 + MEAN_IDX))
    assert (t["reference"] == "P1").all()


def test_c11_holm_covers_only_the_pre_registered_family():
    t = run11()
    f9 = t[t["family"] == F9]
    assert np.allclose(f9["p_holm"].to_numpy(), holm_adjust(f9["p_raw"].to_numpy()))
    assert (f9["p_holm"] >= f9["p_raw"] - 1e-15).all() and (f9["p_holm"] <= 1).all()
    extra = t[t["family"] == EXPLORATORY]
    assert len(extra) == 4 and set(extra["policy"]) == {"P3_naive"} and extra["p_holm"].isna().all()
    assert abs(float(extra["mean_diff"].iloc[0]) - 0.3) < 1e-9
    assert f9["holm_below_alpha"].isin([True, False]).all() and bool(sel(t, "P0", "D1", "abrupt")["holm_below_alpha"])


def test_c13_pairs_pool_classes_and_use_the_default_level():
    t = run13()
    assert len(t) == 10 and (t["family"] == F10).all() and (t["n_events"] == 2 * N).all()
    assert list(t["comparison"].iloc[::2]) == ["P4a vs P1", "P4a vs P0", "P4a vs P3", "P4a vs P4", "P4a vs P5"]
    by = t[t["dataset"] == "D1"].set_index("comparison")
    assert by.loc["P4a vs P1", "mean_diff"] == pytest.approx(-2.0, abs=1e-9)
    assert by.loc["P4a vs P0", "mean_diff"] == pytest.approx(-2.0 - (20.0 + 0.1 * MEAN_IDX), abs=1e-9)
    assert by.loc["P4a vs P4", "mean_diff"] == pytest.approx(-1.0, abs=1e-9)
    assert by.loc["P4a vs P5", "mean_diff"] == pytest.approx(-2.0 + 3.0 + 0.02 * MEAN_IDX, abs=1e-9)
    assert np.allclose(t["p_holm"].to_numpy(), holm_adjust(t["p_raw"].to_numpy()))


def test_pairing_failures_raise():
    q1 = make_q1()
    drop = q1[(q1["variant"] == "P4a") & (q1["factor"] == "AA6") & (q1["applied"] == 0.9)].index[0]
    with pytest.raises(ValueError, match="no partner"):
        run13(q1=q1.drop(index=drop))
    bad = q1.copy()
    bad.loc[drop, "event_id"] = "other"
    with pytest.raises(ValueError, match="no partner|different event_id"):
        run13(q1=bad)
    must = make_must()
    idx = must[(must["variant"] == "P0") & (must["m"] == 5) & (must["headroom"] == 1.6)].index[0]
    with pytest.raises(ValueError, match="no partner"):
        run11(must=must.drop(index=idx))


def test_tables_reproduce_exactly():
    pd.testing.assert_frame_equal(run11(), run11())
    pd.testing.assert_frame_equal(run13(), run13())


def test_stages_are_registered():
    spec = importlib.util.spec_from_file_location("analyze", Path(__file__).resolve().parents[1] / "scripts" / "05_analyze.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert {"c11_main_comparison", "c13_p4a_pairs"} <= set(mod.STAGES)
