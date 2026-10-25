"""Planted tests for the RV tables (A76): known damages give known means, differences, recovery fractions and coverage."""
import numpy as np
import pandas as pd
import pytest

from src import rvtables as rv

N_EVENTS = 12
BOOT = 200
LEVEL = 0.9
SEED = 7
M = 4
HEADROOM = 0.2
Q = 0.9
HEADLINE = {"D1": "P3_lightgbm"}


def block(variant, damages, dataset="D1", **extra):
    rows = []
    for i, d in enumerate(damages):
        row = dict(dataset=dataset, variant=variant, onset_class="abrupt" if i % 2 == 0 else "ramped", m=M,
                   headroom=HEADROOM, replicate=i // 2, event_id="e%d" % i, damage=float(d))
        row.update(extra)
        rows.append(row)
    return rows


def noise():
    return np.linspace(0.0, 1.0, N_EVENTS)


def test_rv0_means():
    e = noise()
    must = pd.DataFrame(sum([block(v, 10 + k + e) for k, v in enumerate(["P0", "P1", "P2", "P3_lightgbm", "P4", "P5"])], []))
    q1 = pd.DataFrame(block("P4a", 20 + e, factor="AA6", param="margin_quantile", applied=Q))
    rv0 = pd.DataFrame(block("P1", 5 + e, dataset="D1_D3") + block("P3_lightgbm", 6 + e, dataset="D1_D3"))
    table = rv.rv0_table(must, q1, rv0, HEADLINE, M, HEADROOM, Q, BOOT, LEVEL, SEED)
    get = lambda d, p: table[(table["dataset"] == d) & (table["policy"] == p)].iloc[0]
    assert get("D1", "P1")["mean_damage"] == pytest.approx(11 + e.mean())
    assert get("D1", "P3")["mean_damage"] == pytest.approx(13 + e.mean())
    assert get("D1", "P4a")["mean_damage"] == pytest.approx(20 + e.mean())
    assert get("D1_D3", "P3")["source"] == "q11_rv0:P3_lightgbm"
    assert get("D1", "P0")["n_events"] == N_EVENTS
    assert get("D1", "P0")["ci_low"] <= get("D1", "P0")["mean_damage"] <= get("D1", "P0")["ci_high"]


def rv1_frame(offsets):
    e = noise()
    rows = []
    for policy in rv.RV1_POLICIES:
        for level, shift in offsets.items():
            mode = "coupled" if level in (rv.FULL, rv.CPU_OFF) else "const_m"
            rows += block(policy, 10 + shift + e * (1 + 0.1 * shift), factor="RV1", level=level, mode=mode,
                          mig_censored=1, mig_duration_mean=3.0)
    return pd.DataFrame(rows)


def test_rv1_recovery_fraction():
    offsets = {rv.FULL: 0.0, rv.CPU_OFF: -2.0, rv.NIC_OFF: -4.0, rv.BOTH_OFF: -6.0}
    table = rv.rv1_table(rv1_frame(offsets), M, HEADROOM, BOOT, LEVEL, SEED)
    sub = table[(table["policy"] == "P1") & (table["level"] == rv.CPU_OFF)].iloc[0]
    assert sub["recovery_defined"]
    assert 0.2 < sub["recovery_fraction"] < 0.5
    nic = table[(table["policy"] == "P1") & (table["level"] == rv.NIC_OFF)].iloc[0]
    assert nic["recovery_fraction"] > sub["recovery_fraction"]
    full = table[(table["policy"] == "P1") & (table["level"] == rv.FULL)].iloc[0]
    assert full["mean_diff_to_full"] == 0 and not full["recovery_defined"]
    assert full["censored_moves"] == N_EVENTS
    assert len(table) == len(rv.RV1_POLICIES) * len(rv.LEVELS)


def test_rv1_undefined_when_both_off_not_below_full():
    offsets = {rv.FULL: 0.0, rv.CPU_OFF: -2.0, rv.NIC_OFF: -4.0, rv.BOTH_OFF: 1.0}
    table = rv.rv1_table(rv1_frame(offsets), M, HEADROOM, BOOT, LEVEL, SEED)
    off = table[table["level"].isin(rv.OFF_ROWS)]
    assert (~off["recovery_defined"]).all() and off["recovery_fraction"].isna().all()


def test_rv1_event_mismatch_raises():
    frame = rv1_frame({rv.FULL: 0.0, rv.CPU_OFF: -2.0, rv.NIC_OFF: -4.0, rv.BOTH_OFF: -6.0})
    bad = (frame["variant"] == "P1") & (frame["level"] == rv.CPU_OFF) & (frame["event_id"] == "e0")
    frame.loc[bad, "event_id"] = "other"
    with pytest.raises(ValueError):
        rv.rv1_table(frame, M, HEADROOM, BOOT, LEVEL, SEED)


def test_rv23_differences():
    e = noise()
    must = pd.DataFrame(block("P0", 10 + e) + block("P1", 8 + e))
    q1 = pd.DataFrame(block("P4a", 7 + e, factor="AA6", param="margin_quantile", applied=Q))
    rows = []
    shifts = {"P0": 1.0, "P1": 2.0, "P4a": -1.0}
    base = {"P0": 10, "P1": 8, "P4a": 7}
    for p in rv.RV23_POLICIES:
        rows += block(p, base[p] + shifts[p] + e * 2, factor="RV2", placement="skewed", budget=M)
    for p in rv.RV3_POLICIES:
        rows += block(p, base[p] + 3 + e, factor="RV3", placement="balanced", budget=1)
        rows += block(p, base[p] + shifts[p] + 5 + e * 2, factor="RV3", placement="skewed", budget=1)
    rv123 = pd.DataFrame(rows)
    rv2, rv3 = rv.rv23_tables(must, q1, rv123, M, HEADROOM, Q, BOOT, LEVEL, SEED)
    r = rv2[rv2["policy"] == "P1"].iloc[0]
    assert r["mean_diff"] == pytest.approx(2.0 + e.mean())
    assert r["direction"] == "higher damage" and r["reference_source"] == "must:P1"
    a = rv2[rv2["policy"] == "P4a"].iloc[0]
    assert a["reference_source"].startswith("q1:AA6")
    assert len(rv3) == 2 * len(rv.RV3_POLICIES)
    s = rv3[(rv3["policy"] == "P1") & (rv3["comparison"].str.endswith("skewed"))].iloc[0]
    assert s["mean_diff"] == pytest.approx(5.0)
    b = rv3[(rv3["policy"] == "P4a") & (rv3["comparison"].str.endswith("balanced"))].iloc[0]
    assert b["mean_diff"] == pytest.approx(3.0)


def test_rv4_pooled_coverage():
    e = noise()
    extra = {m: 0.5 for m in rv.RV4_METRICS}
    rows = block("P4", 1 + e, factor="RV4", **extra)
    for r in rows:
        r.update(rv4_trials=10, rv4_fix_hits=9, rv4_ad_hits=8)
    rows[0]["rv4_ad_alpha_last"] = None
    table = rv.rv4_table(pd.DataFrame(rows), M, HEADROOM, BOOT, LEVEL, SEED)
    fix = table[table["metric"] == "rv4_fix_pooled_coverage"].iloc[0]
    ad = table[table["metric"] == "rv4_ad_pooled_coverage"].iloc[0]
    assert fix["mean"] == pytest.approx(0.9) and ad["mean"] == pytest.approx(0.8)
    last = table[table["metric"] == "rv4_ad_alpha_last"].iloc[0]
    assert last["n_null_rows"] == 1 and last["n_events"] == N_EVENTS - 1
