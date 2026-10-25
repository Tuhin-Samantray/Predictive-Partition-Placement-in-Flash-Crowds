"""C10 (H12): P4a against P4, and the rerun consistency table (Phase 12, A72). Pure functions; every parameter is an argument.

Pairs (A57 item 6): the Must-grid P4 rows and the Q1 AA6 level margin_quantile P4a rows at the default m and headroom,
strictly paired on dataset, onset class, m, headroom and replicate (event ids must agree), onset classes pooled; the
unit is the event (per-event value = mean over its replicates) and the difference is P4a minus P4. Wasted moves: BCa
interval of the mean difference and a one-sided Wilcoxon (less), Holm over the datasets. Damage: non-inferiority
when the BCa upper end of the mean difference is at most h12_noninf times the mean damage of P4 (relative margin
because the dataset scales differ); no p value. Tests counted: the wasted-moves test and the non-inferiority
criterion on each dataset.
"""
import pandas as pd

from src.crosslayer import UNIT, pair_multi
from src.lagtax import KEYS
from src.maincomp import default_rows, p4a_rows, plain
from src.seeds import derive_seed
from src.stats_tools import bca_interval, holm_adjust, mean_stat, paired_wilcoxon, support_label

NAN = float("nan")
CONSISTENCY_METRICS = ("damage", "moves", "wasted_moves")


def p4_pairs(must, q1, m, headroom, margin_quantile):
    """Must-grid P4 rows and Q1 AA6 P4a rows (default margin level) at the default m and headroom."""
    p4 = plain(default_rows(must, m, headroom), "P4")
    p4a = p4a_rows(q1, m, headroom, margin_quantile)
    if p4.empty or p4a.empty:
        raise ValueError("no P4 or P4a rows at the default m and headroom")
    return p4, p4a


def event_diffs(p4, p4a, metric):
    """Per-event P4 value and difference (P4a minus P4) of one metric; rows strictly paired."""
    paired = pair_multi(p4, p4a, KEYS, [metric])
    paired["diff"] = paired[metric + "_b"] - paired[metric + "_a"]
    return paired.groupby(UNIT, sort=True)[["diff", metric + "_a"]].mean()


def c10_table(must, q1, m, headroom, margin_quantile, h12_noninf, alpha, min_events, n_boot, level, master_seed):
    """C10 (H12). Supported when on every dataset the wasted-moves mean difference is below 0 with Holm-adjusted p
    below alpha and the damage non-inferiority holds; not supported when on any dataset the wasted-moves interval
    lies entirely above 0; inconclusive otherwise or below min_events events."""
    p4, p4a = p4_pairs(must, q1, m, headroom, margin_quantile)
    rows = []
    for dataset in sorted(p4["dataset"].unique()):
        wasted = event_diffs(p4[p4["dataset"] == dataset], p4a[p4a["dataset"] == dataset], "wasted_moves")
        damage = event_diffs(p4[p4["dataset"] == dataset], p4a[p4a["dataset"] == dataset], "damage")
        dw = wasted["diff"].to_numpy()
        dd = damage["diff"].to_numpy()
        seed = derive_seed("c10", master_seed, dataset)
        w_lo, w_hi = bca_interval(dw, mean_stat, n_boot, level, derive_seed(seed, "wasted"))
        d_lo, d_hi = bca_interval(dd, mean_stat, n_boot, level, derive_seed(seed, "damage"))
        _stat, p = paired_wilcoxon(dw, "less")
        margin = h12_noninf * float(damage["damage_a"].mean())
        rows.append(dict(dataset=dataset, n_events=int(dw.size), mean_wasted_p4=float(wasted["wasted_moves_a"].mean()),
                         mean_wasted_p4a=float(wasted["wasted_moves_a"].mean() + dw.mean()),
                         wasted_diff=float(dw.mean()), wasted_ci_low=w_lo, wasted_ci_high=w_hi, p_raw=float(p),
                         mean_damage_p4=float(damage["damage_a"].mean()), damage_diff=float(dd.mean()),
                         damage_ci_low=d_lo, damage_ci_high=d_hi, noninf_margin=margin, noninf_met=bool(d_hi <= margin)))
    table = pd.DataFrame(rows)
    table["p_holm"] = holm_adjust(table["p_raw"].to_numpy())
    table["n_tests_counted"] = 2 * len(table)
    met = bool(((table["wasted_diff"] < 0) & (table["p_holm"] < alpha) & table["noninf_met"]).all())
    contradicted = bool((table["wasted_ci_low"] > 0).any())
    table["h12_label"] = support_label(met, contradicted, int(table["n_events"].min()), min_events)
    return table


def rerun_consistency(must, q1, q11, m, headroom, margin_quantile, metrics=CONSISTENCY_METRICS):
    """Counts of exact mismatches between the Q11 logged reruns (RV4) and the rows they repeat: P4 against the
    Must-grid P4 rows and P4a against the Q1 AA6 P4a rows, on the datasets the Q11 rows share with them."""
    p4, p4a = p4_pairs(must, q1, m, headroom, margin_quantile)
    rerun_rows = default_rows(q11, m, headroom)
    rows = []
    for name, ref, label in (("P4", p4, "Q11 P4 rerun against Must P4"),
                             ("P4a", p4a, "Q11 P4a rerun against Q1 AA6 P4a")):
        for dataset in sorted(ref["dataset"].unique()):
            rerun = rerun_rows[(rerun_rows["variant"] == name) & (rerun_rows["dataset"] == dataset)]
            if rerun.empty:
                continue
            paired = pair_multi(ref[ref["dataset"] == dataset], rerun, KEYS, list(metrics))
            row = dict(comparison=label, dataset=dataset, n_pairs=int(len(paired)))
            for metric in metrics:
                row["mismatch_" + metric] = int((paired[metric + "_a"] != paired[metric + "_b"]).sum())
            row["max_abs_diff_damage"] = float((paired["damage_a"] - paired["damage_b"]).abs().max())
            rows.append(row)
    return pd.DataFrame(rows)
