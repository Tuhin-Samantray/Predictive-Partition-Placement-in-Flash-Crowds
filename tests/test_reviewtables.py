"""Planted tests for the Phase 14 review tables (A89): known metric values give known means, spreads and ratios."""
import numpy as np
import pandas as pd
import pytest

from src import reviewtables as rt

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
                   headroom=HEADROOM, replicate=i // 2, event_id="e%d" % i, damage=float(d),
                   slo_violation_frac=float(d) / 100.0, peak_backlog_steps=float(d) * 2.0, moves=float(d) + 5.0)
        row.update(extra)
        rows.append(row)
    return rows


def noise():
    return np.linspace(0.0, 1.0, N_EVENTS)


def frames(p2_scale=2.0):
    e = noise()
    base = {"P0": 30 + e, "P1": 10 + e, "P2": p2_scale * (10 + e) + e ** 2, "P3_lightgbm": 12 + e, "P4": 13 + e,
            "P5": 8 + e}
    must = pd.DataFrame(sum([block(v, d) for v, d in base.items()], []))
    q1 = pd.DataFrame(block("P4a", 14 + e, factor="AA6", param="margin_quantile", applied=Q))
    rv0 = pd.DataFrame(block("P1", 5 + e, dataset="D1_D3") + block("P3_lightgbm", 6 + e, dataset="D1_D3"))
    return must, q1, rv0, base


def test_policy_metrics_values_and_ratio():
    must, q1, rv0, base = frames()
    table = rt.policy_metrics_table(must, q1, rv0, HEADLINE, M, HEADROOM, Q, BOOT, LEVEL, SEED)
    get = lambda d, p, m: table[(table["dataset"] == d) & (table["policy"] == p) & (table["metric"] == m)].iloc[0]
    e = noise()
    assert get("D1", "P1", "damage")["mean"] == pytest.approx(10 + e.mean())
    assert get("D1", "P1", "damage")["sd"] == pytest.approx(np.std(e, ddof=1))
    assert get("D1", "P1", "damage")["median"] == pytest.approx(10 + np.median(e))
    assert get("D1", "P3", "moves")["mean"] == pytest.approx(17 + e.mean())
    assert get("D1", "P4a", "peak_backlog_steps")["mean"] == pytest.approx(2 * (14 + e.mean()))
    assert get("D1", "P4a", "damage")["source"].startswith("q1:AA6")
    assert get("D1_D3", "P3", "damage")["source"] == "q11_rv0:P3_lightgbm"
    assert get("D1", "P0", "damage")["n_events"] == N_EVENTS
    row = get("D1", "P2", "damage")
    assert row["ratio_to_p1"] == pytest.approx(base["P2"].sum() / base["P1"].sum())
    assert row["ratio_ci_low"] <= row["ratio_to_p1"] <= row["ratio_ci_high"]
    one = get("D1", "P1", "slo_violation_frac")
    assert one["ratio_to_p1"] == 1.0 and one["ratio_ci_low"] == 1.0 and one["ratio_ci_high"] == 1.0
    assert get("D1_D3", "P3", "damage")["ratio_to_p1"] == pytest.approx((6 * N_EVENTS + noise().sum()) / (5 * N_EVENTS + noise().sum()))
    assert len(table) == (len(rt.POLICIES) + 2) * len(rt.METRICS)


def test_policy_metrics_ratio_undefined_when_reference_sum_is_zero():
    must, q1, rv0, base = frames()
    must.loc[must["variant"] == "P1", "moves"] = 0.0
    table = rt.policy_metrics_table(must, q1, rv0, HEADLINE, M, HEADROOM, Q, BOOT, LEVEL, SEED)
    sub = table[(table["dataset"] == "D1") & (table["metric"] == "moves") & (table["policy"] == "P2")].iloc[0]
    assert np.isnan(sub["ratio_to_p1"]) and np.isnan(sub["ratio_ci_low"])


def test_policy_metrics_event_mismatch_raises():
    must, q1, rv0, base = frames()
    bad = (must["variant"] == "P2") & (must["event_id"] == "e0")
    must.loc[bad, "event_id"] = "other"
    with pytest.raises(ValueError):
        rt.policy_metrics_table(must, q1, rv0, HEADLINE, M, HEADROOM, Q, BOOT, LEVEL, SEED)


def test_policy_metrics_missing_reference_raises():
    must, q1, rv0, base = frames()
    with pytest.raises(ValueError):
        rt.policy_metrics_table(must[must["variant"] != "P1"], q1, rv0, HEADLINE, M, HEADROOM, Q, BOOT, LEVEL, SEED)


def test_rv1_metrics_table():
    e = noise()
    rows = []
    shifts = {"full": 0.0, "cpu_off": -2.0, "nic_off": -4.0, "both_off": -6.0}
    for policy in rt.RV1_POLICIES:
        for level, shift in shifts.items():
            mode = "coupled" if level in ("full", "cpu_off") else "const_m"
            rows += block(policy, 10 + shift + e, factor="RV1", level=level, mode=mode)
    table = rt.rv1_metrics_table(pd.DataFrame(rows), M, HEADROOM, BOOT, LEVEL, SEED)
    assert len(table) == len(rt.RV1_POLICIES) * len(rt.LEVELS) * len(rt.METRICS)
    sub = table[(table["policy"] == "P3") & (table["level"] == "nic_off") & (table["metric"] == "damage")].iloc[0]
    assert sub["mean"] == pytest.approx(6 + e.mean()) and sub["mode"] == "const_m"
    mv = table[(table["policy"] == "P1") & (table["level"] == "full") & (table["metric"] == "moves")].iloc[0]
    assert mv["mean"] == pytest.approx(15 + e.mean()) and mv["n_events"] == N_EVENTS
    assert mv["ci_low"] <= mv["mean"] <= mv["ci_high"]
