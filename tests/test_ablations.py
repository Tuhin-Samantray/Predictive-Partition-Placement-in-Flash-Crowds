"""Planted tests for the ablation tables (A78): known damage shifts give known level-minus-reference differences."""
import numpy as np
import pandas as pd
import pytest

from src import ablations as ab

N = 12
BOOT = 200
LEVEL = 0.9
SEED = 5
M = 4
HEADROOM = 0.2
Q = 0.9
HEADLINE = {"D1": "P3_lightgbm"}


def noise():
    return np.linspace(0.0, 1.0, N)


def block(variant, damages, dataset="D1", **extra):
    rows = []
    for i, d in enumerate(damages):
        row = dict(dataset=dataset, variant=variant, onset_class="abrupt" if i % 2 == 0 else "ramped", m=M,
                   headroom=HEADROOM, replicate=i // 2, event_id="e%d" % i, damage=float(d))
        row.update(extra)
        rows.append(row)
    return rows


def frames():
    e = noise()
    must = pd.DataFrame(block("P1", 10 + e) + block("P3_lightgbm", 13 + e) + block("P3_naive", 15 + e)
                        + block("P3_ridge", 12 + e) + block("P4", 30 + e))
    q1 = pd.DataFrame(
        block("P1", 12 + e, factor="AA3", param="hysteresis", level=0.1, applied=0.1)
        + block("P1", 10 + e, factor="AA3", param="hysteresis", level=0.2, applied=0.2)
        + block("P1", 9 + e, factor="AA5", param="budget", level="unlimited", applied=512)
        + block("P4a", 20 + e, factor="AA6", param="margin_quantile", level=Q, applied=Q)
        + block("P4a", 23 + e, factor="AA6", param="margin_quantile", level=0.8, applied=0.8))
    q3 = pd.DataFrame(block("P3", 11 + e, mode="coupled", h_net=2.0, phi=1.0)
                      + block("P3", 99 + e, mode="coupled", h_net=1.3, phi=1.0)
                      + block("P3", 98 + e, mode="const_m", h_net=2.0, phi=None))
    q4 = pd.DataFrame(block("P1", 11 + e, factor="AA10", param="kappa", level=0.5, applied=0.5)
                      + block("P3", 9 + e, factor="AA11", param="layers", level="off", applied="off"))
    q6 = pd.DataFrame(block("P1", 14 + e, factor="AA7", param="d", level=3, applied=3))
    return must, q1, q3, q4, q6


def build(must, q1, q3, q4, q6):
    return ab.ablation_table(must, q1, q3, q4, q6, HEADLINE, M, HEADROOM, Q, 2.0, 1.0, 0.05, BOOT, LEVEL, SEED)


def get(table, ablation, variant, level):
    sel = table[(table["ablation"] == ablation) & (table["variant"] == variant) & (table["level"] == level)]
    assert len(sel) == 1
    return sel.iloc[0]


def test_known_differences():
    table = build(*frames())
    assert len(table) == 9
    assert get(table, "AA3", "P1", "0.1")["mean_diff"] == pytest.approx(2.0)
    assert get(table, "AA3", "P1", "0.2")["mean_diff"] == pytest.approx(0.0)
    assert get(table, "AA5", "P1", "unlimited")["mean_diff"] == pytest.approx(-1.0)
    assert get(table, "AA6", "P4a", "0.8")["mean_diff"] == pytest.approx(3.0)
    assert get(table, "AA6", "P4a", "0.8")["reference"] == "q1:AA6 P4a margin_quantile=0.9"
    assert get(table, "AA7", "P1", "3")["mean_diff"] == pytest.approx(4.0)
    assert get(table, "AA10", "P1", "0.5")["mean_diff"] == pytest.approx(1.0)
    assert get(table, "AA1", "P3", "P3_naive")["mean_diff"] == pytest.approx(2.0)
    assert get(table, "AA1", "P3", "P3_ridge")["mean_diff"] == pytest.approx(-1.0)
    assert get(table, "AA11", "P3", "off")["mean_diff"] == pytest.approx(-2.0)
    assert get(table, "AA11", "P3", "off")["reference"] == "q3:coupled h_net=2.0 phi=1.0"


def test_default_levels_not_tested_and_interval_holds_effect():
    table = build(*frames())
    assert not ((table["ablation"] == "AA6") & (table["level"] == "0.9")).any()
    row = get(table, "AA3", "P1", "0.1")
    assert row["ci_low"] <= row["mean_diff"] <= row["ci_high"]
    assert row["n_events"] == N
    assert row["family"] == "C12:AA3"


def test_holm_family_per_ablation():
    table = build(*frames())
    fam = table[table["ablation"] == "AA3"]
    assert (fam["n_tests"] == 2).all()
    assert (fam["p_holm"] >= fam["p_raw"] - 1e-12).all()
    assert get(table, "AA3", "P1", "0.2")["p_raw"] == 1
    assert get(table, "AA7", "P1", "3")["n_tests"] == 1
    assert get(table, "AA7", "P1", "3")["p_holm"] == pytest.approx(get(table, "AA7", "P1", "3")["p_raw"])
    assert bool(get(table, "AA7", "P1", "3")["holm_below_alpha"]) == (get(table, "AA7", "P1", "3")["p_holm"] < 0.05)


def test_strict_pairing_raises():
    must, q1, q3, q4, q6 = frames()
    q1 = q1.drop(q1.index[2])
    with pytest.raises(ValueError):
        build(must, q1, q3, q4, q6)
