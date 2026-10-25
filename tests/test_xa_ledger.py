"""Planted tests for the XA tables, the ledger and TA3 to TA9 (A80)."""
import numpy as np
import pandas as pd

from src import ledger, tatables, xatables


def cross_frame():
    rows = []
    mase = {("fixed", "D1", "D1"): 1.0, ("fixed", "D2", "D1"): 3.0, ("fixed", "D2", "D2"): 0.5, ("fixed", "D1", "D2"): 0.75}
    for (protocol, src, tgt), base in mase.items():
        for group, factor, cells in (("all", 1.0, 100), ("event", 4.0, 20), ("nonevent", 0.5, 80), ("day0", 2.0, 50),
                                     ("day1", 1.0, 40)):
            rows.append({"protocol": protocol, "source": src, "target": tgt, "forecaster": "ridge", "hp": 0.1,
                         "horizon": 5, "group": group, "n_cells": cells, "n_mase_cells": cells, "mase": base * factor,
                         "smape": base})
    for tgt, base in (("D1", 1.2), ("D2", 0.6)):
        for group, factor, cells in (("all", 1.0, 100), ("event", 5.0, 20), ("nonevent", 0.5, 80), ("day0", 1.0, 60)):
            rows.append({"protocol": "rolling", "source": tgt, "target": tgt, "forecaster": "ridge", "hp": np.nan,
                         "horizon": 5, "group": group, "n_cells": cells, "n_mase_cells": cells, "mase": base * factor,
                         "smape": base})
    return pd.DataFrame(rows)


def test_xa1_ratio_and_rolling_reference():
    out = xatables.xa1_table(cross_frame())
    cell = out[(out["source"] == "D2") & (out["target"] == "D1") & (out["group"] == "all")].iloc[0]
    assert cell["cell"] == "cross-dataset"
    assert np.isclose(cell["ratio"], 3.0) and np.isclose(cell["mase_in"], 1.0)
    assert np.isclose(cell["mase_rolling_in"], 1.2)
    own = out[(out["source"] == "D1") & (out["target"] == "D1") & (out["group"] == "event")].iloc[0]
    assert own["cell"] == "in-dataset" and np.isclose(own["ratio"], 1.0)
    event = out[(out["source"] == "D1") & (out["target"] == "D2") & (out["group"] == "event")].iloc[0]
    assert np.isclose(event["ratio"], 0.75 * 4.0 / (0.5 * 4.0))
    assert len(out) == 4 * 5 and not out["protocol"].eq("rolling").any()


def test_xa1_zero_in_dataset_gives_nan():
    cross = cross_frame()
    cross.loc[(cross["protocol"] == "fixed") & (cross["source"] == "D1") & (cross["target"] == "D1"), "mase"] = 0.0
    out = xatables.xa1_table(cross)
    assert out.loc[(out["source"] == "D2") & (out["target"] == "D1"), "ratio"].isna().all()


def test_xa5_days_partial_and_ratio():
    cross = cross_frame()
    out = xatables.xa5_table(cross)
    assert set(out["group"]) == {"all", "day0", "day1"}
    fixed = out[(out["protocol"] == "fixed") & (out["source"] == "D1") & (out["target"] == "D1")]
    day1 = fixed[fixed["group"] == "day1"].iloc[0]
    assert bool(day1["partial_day"]) and day1["max_day_cells"] == 50
    day0 = fixed[fixed["group"] == "day0"].iloc[0]
    assert not bool(day0["partial_day"]) and np.isclose(day0["ratio_to_all_days"], 2.0)
    assert fixed[fixed["group"] == "all"].iloc[0]["day"] == -1
    roll = out[out["protocol"] == "rolling"]
    assert not roll["partial_day"].any()


def test_xa7_mase_columns_and_chronos_kept_apart():
    chronos = pd.DataFrame({"dataset": ["D1", "D1"], "forecaster": ["chronos_bolt_small", "ridge"], "split": ["eval"] * 2,
                            "horizon": [5, 5], "mase_all": [0.9, 1.1], "mase_event": [10.0, 11.0], "n_origins": [1440, 1440]})
    out = xatables.xa7_mase_table(cross_frame(), chronos)
    row = out[(out["origin_subset"] == "all_origins") & (out["dataset"] == "D1")].iloc[0]
    assert np.isclose(row["mase_event"], 6.0) and np.isclose(row["mase_nonevent"], 0.6)
    assert np.isclose(row["ratio_event_all"], 5.0) and np.isclose(row["ratio_event_nonevent"], 10.0)
    sub = out[out["origin_subset"] == "stride_subset"]
    assert len(sub) == 2 and sub["mase_nonevent"].isna().all() and sub["ratio_event_nonevent"].isna().all()
    assert sub["n_origins"].eq(1440).all() and out[out["origin_subset"] == "all_origins"]["n_origins"].isna().all()


def warning(forecasters, value):
    rows = [{"dataset": "D1", "forecaster": f, "split": "eval", "onset_class": "abrupt", "n_events": 5,
             "n_excluded_events": 0, "recall": value, "lead_median": 3.0, "lead_mean": 3.0, "n_alarm_cells": 9,
             "precision": 0.1} for f in forecasters]
    return pd.DataFrame(rows)


def test_xa7_warning_dedupes_and_counts_mismatch():
    main = warning(["naive", "ridge"], 0.5)
    base = warning(["naive", "ewma"], 0.5)
    assert xatables.baseline_overlap_mismatch(main, base) == 0
    base.loc[base["forecaster"] == "naive", "recall"] = 0.9
    assert xatables.baseline_overlap_mismatch(main, base) == 1
    chronos = warning(["chronos_bolt_small", "naive"], 0.2).assign(n_origins=1440)
    out = xatables.xa7_warning_table(main, base, chronos)
    assert len(out) == 3 + 2
    keep = out[(out["origin_subset"] == "all_origins") & (out["forecaster"] == "naive")]
    assert len(keep) == 1 and np.isclose(keep.iloc[0]["recall"], 0.5)
    assert out[out["origin_subset"] == "stride_subset"]["n_origins"].eq(1440).all()


CFG = {"alpha_adj": 0.05, "h2_split": 0.5, "min_events": 24, "h3_ub": 0.2, "h4_rho": 0.5, "h4_ub": 0.7, "h5_ratio": 1.25,
       "h6_tau": 0.6, "h9_share": 0.05, "h12_noninf": 0.02, "interaction_effect_frac": 0.05, "interaction_min_active": 2}


def tables():
    c1 = pd.DataFrame({"dataset": ["D1", "D1", "D2", "D2"], "test": ["wilcoxon_tax", "spearman_tax_m"] * 2,
                       "estimate": [-0.5, 0.2, -0.1, 0.3], "ci_low": [-1.0, 0.1, -0.2, 0.2], "ci_high": [0.5, 0.3, 0.1, 0.4],
                       "p_holm": [0.9, 0.001, 0.04, 0.001], "h1_label": ["inconclusive"] * 4})
    c2 = pd.DataFrame({"dataset": ["D1", "D2"], "n_events_abrupt": [8, 19], "n_events_ramped": [9, 15],
                       "median_abrupt": [0.06, 0.09], "median_ramped": [0.59, -0.05], "p_holm": [0.27, 0.62],
                       "h2_label": ["inconclusive", "not supported"]})
    c3 = pd.DataFrame({"dataset": ["D1", "D2"], "m_low": [1, 1], "median_m_low": [0.78, -0.7], "ci_low": [0.4, -3.7],
                       "ci_high": [1.2, 0.0], "n_events_m_low": [10, 31], "p_holm": [0.76, 0.009],
                       "p_wilcoxon_exploratory": [0.7, 0.00001], "h3_label": ["inconclusive"] * 2})
    c4 = pd.DataFrame({"rho": [-0.43], "ci_low": [-0.78], "ci_high": [0.08], "n_cells_used": [20], "h4_label": ["inconclusive"]})
    c5 = pd.DataFrame({"part": ["forecast_ratio", "forecast_ratio", "xa2_ratio", "overall"],
                       "unit": ["lightgbm", "ridge", "all policies", "H5"], "median_ratio": [2.09, 1.2, 1.02, np.nan],
                       "ci_low": [1.05, 1.07, 0.98, np.nan], "ci_high": [3.3, 1.3, 1.13, np.nan], "h5_ratio": [1.25] * 4,
                       "met": [True, False, False, False], "below": [False, False, True, False],
                       "h5_label": [np.nan, np.nan, np.nan, "not supported"]})
    c6 = pd.DataFrame({"n_policies": [6], "tau": [0.33], "ci_low": [0.07], "ci_high": [0.73], "h6_tau": [0.6],
                       "met": [False], "h6_label": ["inconclusive"]})
    c7 = pd.DataFrame({"dataset": ["D1", "D1", "D2", "D2"], "h_net": [1.3, 2.0, 1.3, 2.0], "tight": [True, False, True, False],
                       "share": [0.26, 0.2, 0.7, 0.005], "share_ci_low": [0.24, 0.19, 0.6, 0.0], "share_ci_high": [0.27, 0.2, 0.77, 0.01],
                       "mean_diff": [30.0, 41.0, 0.14, 0.03], "diff_ci_low": [23.0, 31.0, -0.08, 0.01], "diff_ci_high": [38.0, 51.0, 0.46, 0.06],
                       "p_holm": [1e-10, 1e-8, 0.03, 0.02], "h9_label": ["supported", "supported", "supported", "supported"]})
    c8 = pd.DataFrame({"dataset": ["D1", "D2"], "tight": [True, True], "n_events": [0, 15], "median_diff": [np.nan, -0.15],
                       "p_holm": [1.0, 1.0], "h10_label": ["inconclusive", "inconclusive"]})
    c9 = pd.DataFrame({"dataset": ["D1"], "metric": ["lag_tax"], "n_events": [60], "estimate": [-0.27], "ci_low": [-10.0],
                       "ci_high": [9.9], "p_holm": [1.0], "h11_label": ["inconclusive"]})
    c10 = pd.DataFrame({"dataset": ["D1", "D2"], "wasted_diff": [31.0, -1.2], "wasted_ci_low": [18.0, -4.0], "wasted_ci_high": [46.0, 1.0],
                        "noninf_met": [True, True], "p_holm": [0.99, 0.63], "n_tests_counted": [4, 4],
                        "h12_label": ["not supported"] * 2})
    f9 = pd.DataFrame({"family": ["F9"] * 3 + ["exploratory"], "policy": ["P0", "P2", "P5", "P3_ewma"], "reference": ["P1"] * 4,
                       "direction": ["higher damage", "lower damage", "lower damage", "lower damage"],
                       "holm_below_alpha": [True, True, False, True]})
    f10 = pd.DataFrame({"family": ["F10"] * 2, "comparison": ["a", "b"], "direction": ["lower damage", "higher damage"],
                        "holm_below_alpha": [True, False]})
    aa = pd.DataFrame({"ablation": ["AA2", "AA10", "AA10", "AA1"], "n_tests": [4, 24, 24, 8],
                       "holm_below_alpha": [True, True, False, False], "p_holm": [0.01, 0.0, 0.5, 0.9],
                       "p_wilcoxon_replicate_exploratory": [0.1] * 4})
    trig = pd.DataFrame({"fires": [True, True, False]})
    return {"c1_lag_tax": c1, "c2_recovery_class": c2, "c3_cheap_migration": c3, "c4_accuracy_benefit": c4,
            "c5_transfer_summary": c5, "c6_ranking_tau": c6, "c7_h9": c7, "c8_h10": c8, "c9_h11": c9, "c10_h12": c10,
            "c11_main_comparison": f9, "c13_p4a_pairs": f10, "aa_ablations": aa, "interaction_trigger": trig}


def test_ledger_reads_labels_and_counts():
    t = tables()
    assert sorted(t) == sorted(ledger.NEEDED)
    out = ledger.ledger_table(CFG, t).set_index("id")
    assert out.loc["C1", "label"] == "inconclusive" and out.loc["C1", "n_tests"] == 4 and out.loc["C1", "holm_family"] == "F1"
    assert out.loc["C2", "label"].startswith(ledger.MIXED) and "not supported" in out.loc["C2", "label"]
    assert out.loc["C3", "exploratory_columns"] == "p_wilcoxon_exploratory"
    assert out.loc["C5", "label"] == "not supported" and out.loc["C5", "n_tests"] == 0
    assert "parts disagree" in out.loc["C5", "note"] and "forecast_ratio:lightgbm" in out.loc["C5", "note"]
    assert out.loc["C7", "label"] == "supported" and out.loc["C7", "n_tests"] == 4
    assert "tight" in out.loc["C7", "result"] and out.loc["C7", "result"].count("tight tier") == 2
    assert out.loc["C8", "n_tests"] == 2 and out.loc["C10", "n_tests"] == 4
    assert out.loc["C11", "n_tests"] == 3 and out.loc["C11", "result"].startswith("3 comparisons, 2 with Holm p below alpha, 1 of")
    assert out.loc["C13", "n_tests"] == 2
    assert out.loc["C12:AA10", "n_tests"] == 24 and out.loc["C12:AA10", "result"].startswith("1 of 2 levels")
    assert list(out.index[out.index.str.startswith("C12")]) == ["C12:AA1", "C12:AA2", "C12:AA10"]
    assert out.loc["INT", "label"] == "does not fire for every pair"
    for rid in ("AA13", "XA6", "FA16"):
        assert out.loc[rid, "label"] == "not completed" and out.loc[rid, "n_tests"] == 0
    assert abs(out.loc["C1", "p_holm_min"] - 0.001) < 1e-12


def test_ledger_label_changes_with_table():
    t = tables()
    t["c4_accuracy_benefit"]["h4_label"] = "supported"
    out = ledger.ledger_table(CFG, t).set_index("id")
    assert out.loc["C4", "label"] == "supported"
    t["c1_lag_tax"] = t["c1_lag_tax"].drop(columns="h1_label").assign(h1_label=np.nan)
    assert ledger.ledger_table(CFG, t).set_index("id").loc["C1", "label"] == ledger.NO_LABEL


def test_ta9_totals():
    out = ledger.ledger_table(CFG, tables())
    ta = tatables.ta9(out).set_index("id")
    conf = out[out["status"].str.startswith("confirmatory")]["n_tests"].sum()
    abl = out[out["status"].str.startswith("ablation")]["n_tests"].sum()
    assert ta.loc["TOTAL confirmatory", "n_tests"] == conf
    assert ta.loc["TOTAL ablation families", "n_tests"] == abl == 8 + 4 + 24
    assert ta.loc["TOTAL all rows", "n_tests"] == out["n_tests"].sum()


def test_ta_assembly():
    c11 = pd.DataFrame({"policy": ["P0"], "reference": ["P1"], "family": ["F9"], "dataset": ["D1"], "onset_class": ["abrupt"],
                        "n_events": [30], "mean_reference": [1.0], "mean_policy": [2.0], "mean_diff": [1.0], "ci_low": [0.5],
                        "ci_high": [1.5], "direction": ["higher damage"], "p_holm": [0.01], "holm_below_alpha": [True]})
    c13 = pd.DataFrame({"comparison": ["P4a vs P1"], "family": ["F10"], "dataset": ["D1"], "n_events": [60]})
    out = tatables.ta4(c11, c13)
    assert list(out["block"]) == ["C11", "C13"] and out.loc[0, "comparison"] == "P0 vs P1" and np.isnan(out.loc[1, "mean_diff"])
    main = pd.DataFrame({"dataset": ["D1"] * 2, "forecaster": ["naive", "ridge"], "split": ["eval", "tune"], "horizon": [1, 1],
                         "mase_all": [1.0, 9.0]})
    base = pd.DataFrame({"dataset": ["D1"] * 2, "forecaster": ["naive", "ewma"], "split": ["eval", "eval"], "horizon": [1, 1],
                         "mase_all": [7.0, 2.0]})
    chron = pd.DataFrame({"dataset": ["D1"], "forecaster": ["chronos_bolt_small"], "split": ["eval"], "horizon": [1],
                          "mase_all": [0.9], "n_origins": [1440]})
    a = tatables.ta3(main, base, chron)
    assert len(a) == 3 and a[(a["forecaster"] == "naive")].iloc[0]["mase_all"] == 1.0
    assert set(a["origin_subset"]) == {"all_origins", "stride_subset"} and "ridge" not in set(a["forecaster"])
    cl3 = pd.DataFrame({"mode": ["coupled", "const_m"], "phi": [1.0, np.nan], "net_share": [0.2, 0.3]})
    t8 = tatables.ta8([("C7", pd.DataFrame({"x": [1]})), ("CL3", cl3)], cl3, "coupled", 1.0)
    assert list(t8["block"]) == ["C7", "CL3"] and t8["block"].is_unique
    x = xatables.xa1_table(cross_frame())
    c5 = tables()["c5_transfer_summary"]
    c6 = tables()["c6_ranking_tau"]
    t7 = tatables.ta7(c5, c6, x)
    assert set(t7["block"]) == {"C5", "C6", "XA1"} and (t7[t7["block"] == "XA1"]["key"].str.contains("->")).all()
