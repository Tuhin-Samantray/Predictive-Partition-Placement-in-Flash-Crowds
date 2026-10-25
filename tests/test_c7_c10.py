import numpy as np
import pandas as pd
import pytest

from src.crosslayer import (CONST_M, COUPLED, attribution_share, c7_table, c8_table, c9_table, pair_multi, select)
from src.lagtax import KEYS
from src.p4a import c10_table, rerun_consistency
from src.stats_tools import INCONCLUSIVE, NOT_SUPPORTED, SUPPORTED

TIERS = [1.3, 2.0, 4.0]
N_BOOT = 999
LEVEL = 0.95
SEED = 7
M = 5
HEADROOM = 1.6
CLASSES = ("abrupt", "ramped")
DATASETS = ("D1", "D2")
LOW_PHI = 0.01
H9_SHARE = 0.05
H12_NONINF = 0.02
ALPHA = 0.05
MIN_EVENTS = 24
MIN_SHARE = 0.05
MARGIN = 0.9
METRICS = ("damage", "moves", "wasted_moves")


def q3_frame(n, spec):
    """spec(variant, mode, phi, j) -> (damage, violating steps, net share)."""
    rows = []
    for dataset in DATASETS:
        for cls in CLASSES:
            for j in range(n):
                for variant in ("P1", "P3", "P5", "P6"):
                    for tier in TIERS:
                        for mode, phi in ((COUPLED, 1.0), (COUPLED, LOW_PHI), (CONST_M, None)):
                            damage, steps, share = spec(variant, mode, phi, j)
                            rows.append(dict(dataset=dataset, onset_class=cls, m=M, headroom=HEADROOM, replicate=j,
                                             event_id="%s-%s-%d" % (dataset, cls, j), variant=variant, mode=mode,
                                             h_net=tier, phi=phi, damage=damage, attr_violating_steps=steps,
                                             attr_net_share=share))
    return pd.DataFrame(rows)


def c7_spec(share_of):
    def spec(variant, mode, phi, j):
        base = 10 + 0.1 * j
        damage = {"P1": base + 6, "P3": base, "P5": base - 4, "P6": base - 1 - 0.01 * j}[variant]
        return damage, 20 + j % 5, share_of(j)
    return spec


def run_c7(share_of, n=30):
    return c7_table(q3_frame(n, c7_spec(share_of)), M, HEADROOM, TIERS, 1.0, H9_SHARE, ALPHA, MIN_EVENTS, N_BOOT,
                    LEVEL, SEED)


def r_spec(coupled_p3, const_p3, defined_every=None):
    def spec(variant, mode, phi, j):
        if variant == "P1":
            damage = 10.0
        elif variant == "P5":
            damage = 9.9 if (defined_every is not None and j % 2 == 0) else 4.0
        elif variant == "P3":
            damage = coupled_p3 + 0.02 * j if mode == COUPLED else const_p3 + 0.01 * j
        else:
            damage = 5.0
        return damage, 10, 0.2
    return spec


def run_c8(spec, n=30):
    return c8_table(q3_frame(n, spec), M, HEADROOM, TIERS, 1.0, MIN_SHARE, ALPHA, MIN_EVENTS, N_BOOT, LEVEL, SEED)


def c9_spec(variant, mode, phi, j):
    low = phi == LOW_PHI
    if variant == "P1":
        damage = 8 + 0.02 * j if low else 10 + 0.01 * j
    elif variant == "P5":
        damage = 4.0
    elif variant == "P3":
        damage = 6.5 + 0.01 * j if low else 6 + 0.01 * j
    else:
        damage = 5.0
    return damage, 10, 0.2


def grid(n, variant, wasted, damage, extra=None, m=M):
    rows = []
    for dataset in DATASETS:
        for cls in CLASSES:
            for j in range(n):
                row = dict(dataset=dataset, onset_class=cls, m=m, headroom=HEADROOM, replicate=j,
                           event_id="%s-%s-%d" % (dataset, cls, j), variant=variant, damage=damage(j),
                           wasted_moves=wasted(j), moves=3 + j % 4)
                row.update(extra or {})
                rows.append(row)
    return pd.DataFrame(rows)


AA6 = dict(factor="AA6", param="margin_quantile", applied=0.9)


def c10_inputs(wasted_shift, damage_shift, n=30):
    must = pd.concat([grid(n, "P4", lambda j: 10 + j % 3, lambda j: 5 + 0.1 * j),
                      grid(n, "P1", lambda j: 1, lambda j: 1.0)], ignore_index=True)
    q1 = pd.concat([
        grid(n, "P4a", lambda j: 10 + j % 3 + wasted_shift(j), lambda j: 5 + 0.1 * j + damage_shift(j), AA6),
        grid(n, "P4a", lambda j: 999, lambda j: 999.0, dict(AA6, applied=0.8)),
        grid(n, "P4a", lambda j: 999, lambda j: 999.0, AA6, m=10)], ignore_index=True)
    return must, q1


def run_c10(wasted_shift, damage_shift):
    must, q1 = c10_inputs(wasted_shift, damage_shift)
    return c10_table(must, q1, M, HEADROOM, MARGIN, H12_NONINF, ALPHA, MIN_EVENTS, N_BOOT, LEVEL, SEED)


def small_frame():
    a = pd.DataFrame({"dataset": ["D1"] * 3, "onset_class": ["abrupt"] * 3, "m": [M] * 3, "headroom": [HEADROOM] * 3,
                      "replicate": [0, 1, 2], "event_id": ["e0", "e1", "e2"], "damage": [1.0, 2.0, 3.0]})
    b = a.copy()
    b["damage"] = [2.0, 3.0, 4.0]
    return a, b


def test_pair_multi_pairs_strictly():
    a, b = small_frame()
    out = pair_multi(a, b, KEYS, ["damage"])
    assert list(out["damage_b"] - out["damage_a"]) == [1.0, 1.0, 1.0]
    with pytest.raises(ValueError, match="duplicate"):
        pair_multi(pd.concat([a, a.iloc[:1]]), b, KEYS, ["damage"])
    with pytest.raises(ValueError, match="no partner"):
        pair_multi(a, b.iloc[:2], KEYS, ["damage"])
    other = b.copy()
    other.loc[0, "event_id"] = "x"
    with pytest.raises(ValueError, match="event_id"):
        pair_multi(a, other, KEYS, ["damage"])


def test_attribution_share_is_a_ratio_of_sums_not_a_mean_of_shares():
    rows = pd.DataFrame({"dataset": ["D1"] * 4, "event_id": ["a", "b", "c", "d"], "attr_net_share": [0.5, 0.1, 0.5, 0.1],
                         "attr_violating_steps": [10, 30, 10, 30]})
    out = attribution_share(rows, N_BOOT, LEVEL, SEED)
    assert out["share"] == pytest.approx(16 / 80) and out["n_violating_steps"] == 80
    assert out["share_ci_low"] < out["share"] < out["share_ci_high"]
    none = attribution_share(rows.assign(attr_violating_steps=0), N_BOOT, LEVEL, SEED)
    assert np.isnan(none["share"]) and np.isnan(none["share_ci_low"]) and none["n_violating_steps"] == 0


def test_select_keeps_one_mode_tier_and_phi():
    q3 = q3_frame(2, c7_spec(lambda j: 0.1))
    sel = select(q3, COUPLED, 2.0, 1.0, M, HEADROOM)
    assert set(sel["mode"]) == {COUPLED} and set(sel["h_net"]) == {2.0} and set(sel["phi"]) == {1.0}
    const = select(q3, CONST_M, 2.0, None, M, HEADROOM)
    assert set(const["mode"]) == {CONST_M} and const["phi"].isna().all()


def test_c7_supported_when_the_network_share_is_high_and_p6_is_lower():
    table = run_c7(lambda j: 0.4 + 0.01 * (j % 3))
    assert len(table) == 6 and set(table["h_net"]) == set(TIERS) and table["tight"].sum() == 2
    tight = table[table["tight"]]
    assert (tight["h_net"] == 1.3).all() and (tight["share"] > 0.39).all() and (tight["share"] < 0.43).all()
    assert (tight["share_ci_low"] > H9_SHARE).all() and (tight["mean_diff"] > 0).all()
    assert (table["p_holm"] >= table["p_raw"]).all() and (table["p_holm"] <= 1).all()
    assert (table["h9_label"] == SUPPORTED).all()


def test_c7_not_supported_when_the_network_share_interval_is_below_the_threshold():
    table = run_c7(lambda j: 0.01 + 0.002 * (j % 3))
    tight = table[table["tight"]]
    assert (tight["share_ci_high"] < H9_SHARE).all()
    assert (table["h9_label"] == NOT_SUPPORTED).all()


def test_c7_inconclusive_when_the_share_interval_straddles_the_threshold():
    table = run_c7(lambda j: 0.1 * (j % 2))
    tight = table[table["tight"]]
    assert np.allclose(tight["share"], H9_SHARE)
    assert (tight["share_ci_low"] < H9_SHARE).all() and (tight["share_ci_high"] > H9_SHARE).all()
    assert (table["h9_label"] == INCONCLUSIVE).all()


def test_c8_supported_when_coupled_recovery_is_lower():
    table = run_c8(r_spec(8.0, 6.0))
    tight = table[table["tight"]]
    assert len(table) == 6 and (tight["median_diff"] < 0).all() and (tight["p_holm"] < ALPHA).all()
    assert (tight["median_const"] > tight["median_coupled"]).all()
    assert (table["h10_label"] == SUPPORTED).all()


def test_c8_not_supported_when_coupled_recovery_is_higher():
    table = run_c8(r_spec(4.0, 6.0))
    assert (table[table["tight"]]["ci_low"] > 0).all()
    assert (table["h10_label"] == NOT_SUPPORTED).all()


def test_c8_drops_events_without_a_defined_recovery_and_needs_min_events():
    table = run_c8(r_spec(8.0, 6.0, defined_every=2), n=10)
    row = table[(table["dataset"] == "D1") & (table["h_net"] == 1.3)].iloc[0]
    assert row["n_events"] == 10 and row["n_events_coupled"] == 10 and row["share_defined_coupled"] == pytest.approx(0.5)
    assert row["median_diff"] < 0
    assert (table["h10_label"] == INCONCLUSIVE).all()


def test_c9_supported_when_a_small_phi_lowers_the_tax_and_the_recovery():
    q3 = q3_frame(30, c9_spec)
    table = c9_table(q3, M, HEADROOM, 2.0, 1.0, LOW_PHI, MIN_SHARE, ALPHA, MIN_EVENTS, N_BOOT, LEVEL, SEED)
    assert len(table) == 4 and set(table["metric"]) == {"lag_tax", "R"}
    assert (table["estimate"] < 0).all() and (table["p_holm"] < ALPHA).all()
    tax = table[(table["dataset"] == "D1") & (table["metric"] == "lag_tax")].iloc[0]
    assert tax["value_phi_high"] > tax["value_phi_low"]
    assert (table["h11_label"] == SUPPORTED).all()
    with pytest.raises(ValueError, match="phi_low"):
        c9_table(q3, M, HEADROOM, 2.0, 1.0, 1.0, MIN_SHARE, ALPHA, MIN_EVENTS, N_BOOT, LEVEL, SEED)


def test_c10_supported_with_fewer_wasted_moves_and_non_inferior_damage():
    table = run_c10(lambda j: -2 + 0.5 * (j % 2), lambda j: 0.01 + 0.001 * (j % 3))
    assert len(table) == 2 and (table["n_events"] == 60).all() and (table["n_tests_counted"] == 4).all()
    assert (table["wasted_diff"] < 0).all() and (table["mean_wasted_p4a"] < table["mean_wasted_p4"]).all()
    assert (table["mean_wasted_p4a"] < 20).all()
    assert (table["p_holm"] < ALPHA).all() and table["noninf_met"].all()
    assert (table["h12_label"] == SUPPORTED).all()


def test_c10_not_supported_when_p4a_wastes_more_moves():
    table = run_c10(lambda j: 1 + 0.5 * (j % 2), lambda j: 0.01 + 0.001 * (j % 3))
    assert (table["wasted_ci_low"] > 0).all()
    assert (table["h12_label"] == NOT_SUPPORTED).all()


def test_c10_inconclusive_when_the_damage_is_not_non_inferior():
    table = run_c10(lambda j: -2 + 0.5 * (j % 2), lambda j: 1 + 0.001 * (j % 3))
    assert (table["wasted_diff"] < 0).all() and not table["noninf_met"].any()
    assert (table["h12_label"] == INCONCLUSIVE).all()


def test_rerun_consistency_counts_exact_mismatches():
    must, q1 = c10_inputs(lambda j: -2 + 0.5 * (j % 2), lambda j: 0.01 + 0.001 * (j % 3))
    rerun_p4 = must[must["variant"] == "P4"]
    rerun_p4a = q1[(q1["applied"] == 0.9) & (q1["m"] == M)]
    q11 = pd.concat([rerun_p4, rerun_p4a], ignore_index=True)
    table = rerun_consistency(must, q1, q11, M, HEADROOM, MARGIN, METRICS)
    assert len(table) == 4 and (table["n_pairs"] == 60).all()
    assert (table[["mismatch_damage", "mismatch_moves", "mismatch_wasted_moves"]] == 0).all().all()
    spot = q11[(q11["variant"] == "P4") & (q11["dataset"] == "D1")].index[0]
    q11.loc[spot, "moves"] += 1
    changed = rerun_consistency(must, q1, q11, M, HEADROOM, MARGIN, METRICS)
    row = changed[(changed["comparison"].str.contains("P4 rerun")) & (changed["dataset"] == "D1")].iloc[0]
    assert row["mismatch_moves"] == 1 and row["mismatch_damage"] == 0 and row["max_abs_diff_damage"] == 0
    assert changed["mismatch_moves"].sum() == 1
