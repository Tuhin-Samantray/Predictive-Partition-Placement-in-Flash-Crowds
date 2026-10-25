"""Hypothesis ledger (A80): one row per pre-registered comparison, read from the recorded tables.

Labels, estimates, intervals and test counts are READ from the tables, never typed; a table that holds several different
labels for one hypothesis yields a row that says so (prefix mixed:) instead of picking one. Comparisons that were not
run (A52) are listed as not completed. Holm p values and exploratory columns come from the same tables.
"""
import numpy as np
import pandas as pd

NEEDED = ["c1_lag_tax", "c2_recovery_class", "c3_cheap_migration", "c4_accuracy_benefit", "c5_transfer_summary",
          "c6_ranking_tau", "c7_h9", "c8_h10", "c9_h11", "c10_h12", "c11_main_comparison", "c13_p4a_pairs",
          "aa_ablations", "interaction_trigger"]
PREFIX = "AA"
NO_LABEL = "no label in table"
MIXED = "mixed:"
CONFIRMATORY = "confirmatory"
RULE_ONLY = "confirmatory (no label rule)"
ABLATION = "ablation family (descriptive)"
RULE = "pre-registered rule"
NOT_RUN = "not completed"
NOT_COMPLETED = [("AA13", "scale ablation, P and N", "not run, scope decision A52"),
                 ("XA6", "cross-resolution transfer", "not run, scope decision A52"),
                 ("FA16", "sensitivity tornado (detector, SLO, bandwidth)", "not run, scope decision A52")]
COLUMNS = ["id", "hypothesis", "quantity", "threshold", "result", "label", "n_tests", "holm_family", "p_holm_min",
           "status", "source_table", "exploratory_columns", "note"]


def num(value):
    return format(float(value), ".4g")


def is_true(series):
    return series.astype(str) == "True"


def label_of(frame, column, mask=None):
    values = frame[column] if mask is None else frame.loc[mask, column]
    found = sorted(set(values.dropna().astype(str)))
    if not found:
        return NO_LABEL
    return found[0] if len(found) == 1 else MIXED + ";".join(found)


def p_min(frame, mask=None):
    if "p_holm" not in frame.columns:
        return np.nan
    values = (frame["p_holm"] if mask is None else frame.loc[mask, "p_holm"]).dropna()
    return float(values.min()) if len(values) else np.nan


def exploratory(frame):
    return ";".join(c for c in frame.columns if "exploratory" in c)


def thr(cfg, *keys):
    return "; ".join("%s=%s" % (k, cfg[k]) for k in keys)


def entry(rid, hyp, quantity, threshold, result, label, n_tests, family, p, status, source, frame, note=""):
    return {"id": rid, "hypothesis": hyp, "quantity": quantity, "threshold": threshold, "result": result,
            "label": label, "n_tests": int(n_tests), "holm_family": family, "p_holm_min": p, "status": status,
            "source_table": source, "exploratory_columns": exploratory(frame), "note": note}


def family_rows(frame, family, mask_family=True):
    return frame[frame["family"] == family] if mask_family else frame


def comparison_result(frame):
    lower = frame["direction"].astype(str).str.contains("lower")
    below = is_true(frame["holm_below_alpha"])
    return "%d comparisons, %d with Holm p below alpha, %d of those with lower damage than the reference" % (
        len(frame), int(below.sum()), int((below & lower).sum()))


def ledger_table(cfg, t):
    """t maps each name of NEEDED to its recorded table; returns the ledger frame."""
    rows = []
    c1 = t["c1_lag_tax"]
    result = "; ".join("%s %s %s [%s, %s], Holm p %s" % (r.dataset, r.test, num(r.estimate), num(r.ci_low),
                                                        num(r.ci_high), num(r.p_holm)) for r in c1.itertuples())
    rows.append(entry("C1", "H1", "mean per-event lag tax above 0 on both datasets, and its growth with m",
                      thr(cfg, "alpha_adj"), result, label_of(c1, "h1_label"), len(c1), "F1", p_min(c1), CONFIRMATORY,
                      "c1_lag_tax.csv", c1))
    c2 = t["c2_recovery_class"]
    result = "; ".join("%s median R abrupt %s, ramped %s, events %d and %d, Holm p %s" % (
        r.dataset, num(r.median_abrupt), num(r.median_ramped), r.n_events_abrupt, r.n_events_ramped, num(r.p_holm))
        for r in c2.itertuples())
    rows.append(entry("C2", "H2", "recovery ratio R higher for ramped than for abrupt onsets",
                      thr(cfg, "h2_split", "alpha_adj", "min_events"), result, label_of(c2, "h2_label"), len(c2), "F2",
                      p_min(c2), CONFIRMATORY, "c2_recovery_class.csv", c2))
    c3 = t["c3_cheap_migration"]
    result = "; ".join("%s median R at m %s %s [%s, %s], events %d, Holm p %s" % (
        r.dataset, r.m_low, num(r.median_m_low), num(r.ci_low), num(r.ci_high), r.n_events_m_low, num(r.p_holm))
        for r in c3.itertuples())
    rows.append(entry("C3", "H3", "median R at the smallest m below the upper bound",
                      thr(cfg, "h3_ub", "alpha_adj", "min_events"), result, label_of(c3, "h3_label"), len(c3), "F3",
                      p_min(c3), CONFIRMATORY, "c3_cheap_migration.csv", c3))
    c4 = t["c4_accuracy_benefit"]
    r = c4.iloc[0]
    rows.append(entry("C4", "H4", "Spearman rho of event-class MASE with placement benefit",
                      thr(cfg, "h4_rho", "h4_ub"), "rho %s [%s, %s] over %d cells" % (
                          num(r["rho"]), num(r["ci_low"]), num(r["ci_high"]), r["n_cells_used"]),
                      label_of(c4, "h4_label"), 1, "single test, no Holm", np.nan, CONFIRMATORY,
                      "c4_accuracy_benefit.csv", c4))
    c5 = t["c5_transfer_summary"]
    used = c5[c5["median_ratio"].notna()]
    result = "; ".join("%s %s median ratio %s [%s, %s]" % (r.part, r.unit, num(r.median_ratio), num(r.ci_low),
                                                          num(r.ci_high)) for r in used.itertuples())
    met = used[is_true(used["met"])]
    below = used[is_true(used["below"])]
    note = ""
    if len(met) and len(below):
        note = "parts disagree: threshold met by %s; below the threshold for %s" % (
            ", ".join(met["part"] + ":" + met["unit"]), ", ".join(below["part"] + ":" + below["unit"]))
    rows.append(entry("C5", "H5", "cross-dataset over in-dataset ratio for forecast error and tuning transfer",
                      thr(cfg, "h5_ratio"), result, label_of(c5, "h5_label"), 0, "no p value, no Holm", np.nan,
                      CONFIRMATORY, "c5_transfer_summary.csv", c5, note))
    c6 = t["c6_ranking_tau"]
    r = c6.iloc[0]
    rows.append(entry("C6", "H6", "Kendall tau of the D1 and D2 policy rankings", thr(cfg, "h6_tau"),
                      "tau %s [%s, %s] over %d policies" % (num(r["tau"]), num(r["ci_low"]), num(r["ci_high"]),
                                                            r["n_policies"]),
                      label_of(c6, "h6_label"), 1, "single test, no Holm", np.nan, CONFIRMATORY, "c6_ranking_tau.csv",
                      c6))
    c7 = t["c7_h9"]
    tight = is_true(c7["tight"])
    result = "; ".join("%s tight tier h_net %s: network share %s [%s, %s], mean damage P3 minus P6 %s [%s, %s], Holm p %s" % (
        r.dataset, r.h_net, num(r.share), num(r.share_ci_low), num(r.share_ci_high), num(r.mean_diff),
        num(r.diff_ci_low), num(r.diff_ci_high), num(r.p_holm)) for r in c7[tight].itertuples())
    rows.append(entry("C7", "H9", "network share of violating node-steps under P3 and P6 advantage, tight tier",
                      thr(cfg, "h9_share", "alpha_adj"), result, label_of(c7, "h9_label", tight), len(c7), "F5",
                      p_min(c7), CONFIRMATORY, "c7_h9.csv", c7,
                      "label read from the tight-tier rows; other tiers are reported in the table"))
    c8 = t["c8_h10"]
    tight = is_true(c8["tight"])
    result = "; ".join("%s tight tier: events with R in both modes %d, median difference %s" % (
        r.dataset, r.n_events, num(r.median_diff)) for r in c8[tight].itertuples())
    rows.append(entry("C8", "H10", "R of P3 coupled minus constant-m, tight tier", thr(cfg, "alpha_adj", "min_events"),
                      result, label_of(c8, "h10_label", tight), len(c8), "F6", p_min(c8), CONFIRMATORY, "c8_h10.csv",
                      c8, "label read from the tight-tier rows; R is defined for few events under the cross-layer model"))
    c9 = t["c9_h11"]
    result = "; ".join("%s %s difference %s [%s, %s] on %d events" % (r.dataset, r.metric, num(r.estimate),
                                                                      num(r.ci_low), num(r.ci_high), r.n_events)
                       for r in c9.itertuples())
    rows.append(entry("C9", "H11", "lag tax and R at the smallest phi against phi 1, moderate tier",
                      thr(cfg, "alpha_adj", "min_events"), result, label_of(c9, "h11_label"), len(c9), "F7", p_min(c9),
                      CONFIRMATORY, "c9_h11.csv", c9))
    c10 = t["c10_h12"]
    result = "; ".join("%s wasted moves P4a minus P4 %s [%s, %s], non-inferiority met %s" % (
        r.dataset, num(r.wasted_diff), num(r.wasted_ci_low), num(r.wasted_ci_high), r.noninf_met)
        for r in c10.itertuples())
    count = int(c10["n_tests_counted"].max()) if "n_tests_counted" in c10.columns else len(c10)
    rows.append(entry("C10", "H12", "P4a wastes fewer moves than P4 with damage not worse than the margin",
                      thr(cfg, "h12_noninf", "alpha_adj"), result, label_of(c10, "h12_label"), count, "F8",
                      p_min(c10), CONFIRMATORY, "c10_h12.csv", c10))
    c11 = family_rows(t["c11_main_comparison"], "F9")
    rows.append(entry("C11", "none", "each policy against the best tuned reactive policy, default m and headroom",
                      thr(cfg, "alpha_adj"), comparison_result(c11), "family reported, no support label", len(c11),
                      "F9", p_min(c11), RULE_ONLY, "c11_main_comparison.csv", t["c11_main_comparison"],
                      "rows outside family F9 are exploratory and not counted"))
    c13 = family_rows(t["c13_p4a_pairs"], "F10")
    rows.append(entry("C13", "none", "P4a against each of P1, P0, P3, P4 and P5, default m and headroom",
                      thr(cfg, "alpha_adj"), comparison_result(c13), "family reported, no support label", len(c13),
                      "F10", p_min(c13), RULE_ONLY, "c13_p4a_pairs.csv", t["c13_p4a_pairs"]))
    aa = t["aa_ablations"]
    for name in sorted(aa["ablation"].unique(), key=lambda s: int(s[len(PREFIX):])):
        part = aa[aa["ablation"] == name]
        below = is_true(part["holm_below_alpha"])
        rows.append(entry("C12:" + name, "none", "ablation levels against the default level, one Holm family",
                          thr(cfg, "alpha_adj"), "%d of %d levels with Holm p below alpha" % (int(below.sum()),
                                                                                              len(part)),
                          "not labelled (descriptive)", int(part["n_tests"].max()), "C12:" + name, p_min(part),
                          ABLATION, "aa_ablations.csv", aa))
    trig = t["interaction_trigger"]
    fires = is_true(trig["fires"])
    rows.append(entry("INT", "none", "interaction trigger, pairs of active factors", thr(cfg, "interaction_effect_frac",
                                                                                         "interaction_min_active"),
                      "fires for %d of %d (dataset, policy) pairs" % (int(fires.sum()), len(trig)),
                      "fires" if fires.all() else "does not fire for every pair", 0, "none", np.nan, RULE,
                      "interaction_trigger.csv", trig, "pairwise interaction queue not run (A61)"))
    for rid, what, why in NOT_COMPLETED:
        rows.append({"id": rid, "hypothesis": "none", "quantity": what, "threshold": "", "result": why,
                     "label": NOT_RUN, "n_tests": 0, "holm_family": "none", "p_holm_min": np.nan, "status": NOT_RUN,
                     "source_table": "", "exploratory_columns": "", "note": "listed in the limitations"})
    return pd.DataFrame(rows, columns=COLUMNS)
