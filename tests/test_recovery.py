import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.recovery import ABRUPT, RAMPED, c2_table, c3_table, event_recovery, recovery_frame
from src.stats_tools import INCONCLUSIVE, NOT_SUPPORTED, SUPPORTED

N = 30
BEST = {"D1": "P1", "D2": "P1"}
HEAD = {"D1": "P3_lightgbm", "D2": "P3_lightgbm"}
SHARE = 0.05


def make_must(r_fn, n=N):
    """Must-grid shaped rows: the replicate field is the event index within its class, as in the real rows."""
    rows = []
    for ds in ("D1", "D2"):
        for cls in (ABRUPT, RAMPED):
            for i in range(n):
                for m in (1, 2, 5, 10):
                    for h in (1.6, 1.3):
                        r = r_fn(cls, i, m) if h == 1.6 else 0.99
                        key = dict(dataset=ds, onset_class=cls, m=m, headroom=h, replicate=i,
                                   event_id="%s-%s-%d" % (ds, cls, i))
                        rows.append(dict(key, variant="P1", damage=10.0))
                        rows.append(dict(key, variant="P5", damage=2.0))
                        rows.append(dict(key, variant="P3_lightgbm", damage=10.0 - r * 8.0))
    return pd.DataFrame(rows)


def spread(base, i, n=N):
    return base + 0.2 * i / (n - 1)


def test_recovery_frame_planted_values_and_undefined_cases():
    key = dict(dataset="D1", onset_class=ABRUPT, m=5, headroom=1.6)
    rows = []
    for rep, (p1, x, p5) in enumerate([(10.0, 4.0, 2.0), (10.0, 9.9, 9.8), (0.0, 0.0, 0.0)]):
        k = dict(key, replicate=rep, event_id="e%d" % rep)
        rows += [dict(k, variant="P1", damage=p1), dict(k, variant="P3_lightgbm", damage=x),
                 dict(k, variant="P5", damage=p5)]
    out = recovery_frame(pd.DataFrame(rows), {"D1": "P1"}, {"D1": "P3_lightgbm"}, SHARE).sort_values("replicate")
    assert list(out["defined"]) == [True, False, False]
    assert out["R"].iloc[0] == pytest.approx(0.75)
    assert np.isnan(out["R"].iloc[1]) and np.isnan(out["R"].iloc[2])
    assert out["tax"].iloc[0] == pytest.approx(8.0)


def test_recovery_frame_rejects_unpaired_and_mismatched_event_ids():
    must = make_must(lambda cls, i, m: 0.5, n=3)
    assert len(recovery_frame(must, BEST, HEAD, SHARE)) == len(must) // 3
    idx = must[must["variant"] == "P3_lightgbm"].index[0]
    bad = must.copy()
    bad.loc[idx, "event_id"] = "other"
    with pytest.raises(ValueError, match="different event_id"):
        recovery_frame(bad, BEST, HEAD, SHARE)
    with pytest.raises(ValueError, match="no partner"):
        recovery_frame(must.drop(index=idx), BEST, HEAD, SHARE)


def test_event_recovery_is_the_median_of_defined_values():
    frame = pd.DataFrame({"dataset": "D1", "onset_class": ABRUPT, "event_id": "e1",
                          "R": [0.2, 0.4, 0.9, np.nan], "defined": [True, True, True, False]})
    assert event_recovery(frame)["R"].iloc[0] == pytest.approx(0.4)


def c2(r_fn, n=N):
    rec = recovery_frame(make_must(r_fn, n), BEST, HEAD, SHARE)
    return c2_table(rec, 5, 1.6, 0.5, 999, 0.95, 0.05, 24, 42)


def test_c2_supported_when_the_split_is_planted():
    t = c2(lambda cls, i, m: spread(0.1 if cls == ABRUPT else 0.7, i))
    assert (t["h2_label"] == SUPPORTED).all() and t["met"].all()
    assert t["median_abrupt"].iloc[0] == pytest.approx(0.2, abs=0.02)
    assert t["median_ramped"].iloc[0] == pytest.approx(0.8, abs=0.02)
    assert t["cliffs_delta"].iloc[0] == pytest.approx(1.0) and (t["p_holm"] < 0.05).all()
    assert (t["n_events_abrupt"] == N).all() and (t["share_defined"] == 1).all()


def test_c2_flipped_pattern_is_not_supported_and_equal_classes_are_inconclusive():
    flipped = c2(lambda cls, i, m: spread(0.7 if cls == ABRUPT else 0.1, i))
    assert (flipped["h2_label"] == NOT_SUPPORTED).all() and not flipped["met"].any()
    same = c2(lambda cls, i, m: spread(0.4, i))
    assert (same["h2_label"] == INCONCLUSIVE).all()


def test_c2_small_sample_is_inconclusive():
    t = c2(lambda cls, i, m: spread(0.1 if cls == ABRUPT else 0.7, i, 10), n=10)
    assert (t["h2_label"] == INCONCLUSIVE).all()


def c3(r_low, n=N):
    rec = recovery_frame(make_must(lambda cls, i, m: r_low(i) if m == 1 else spread(0.6, i), n), BEST, HEAD, SHARE)
    return c3_table(rec, [1, 2, 5, 10], 5, 1.6, 0.2, 999, 0.95, 0.05, 24, 42)


def test_c3_supported_when_r_is_small_at_m1_and_larger_at_higher_m():
    t = c3(lambda i: 0.02 + 0.1 * i / (N - 1))
    assert (t["h3_label"] == SUPPORTED).all() and (t["ci_high"] < 0.2).all()
    assert t["median_m_low"].iloc[0] == pytest.approx(0.07, abs=0.02)
    assert (t["n_events_m_low"] == 2 * N).all() and (t["n_pairs"] == 2 * N).all()
    assert (t["p_raw"] < 0.05).all() and (t["p_wilcoxon_exploratory"] < 0.05).all()
    assert t["median_pairs_m_high"].iloc[0] == pytest.approx(0.7, abs=0.02)


def test_c3_not_supported_when_r_stays_high_and_inconclusive_when_small():
    high = c3(lambda i: 0.5 + 0.2 * i / (N - 1))
    assert (high["h3_label"] == NOT_SUPPORTED).all()
    few = c3(lambda i: 0.02 + 0.1 * i / 9, n=10)
    assert (few["h3_label"] == INCONCLUSIVE).all()


def test_stages_are_registered():
    spec = importlib.util.spec_from_file_location("analyze", Path(__file__).resolve().parents[1] / "scripts" / "05_analyze.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert {"c1_lag_tax", "c2_recovery_class", "c3_cheap_migration"} <= set(mod.STAGES)
