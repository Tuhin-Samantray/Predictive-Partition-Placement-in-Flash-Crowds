"""Cross-layer comparisons C7 to C9 (Phase 12, A72). Pure functions; every parameter is an argument.

Rows are the Q3 rows of the grid runner: P1, P3, P5 and P6 under the cross-layer model (A30, A31), one row per
(dataset, onset class, bandwidth tier h_net, mode, phi, variant, replicate). Every comparison is strictly paired on
dataset, onset class, m, headroom and replicate (event ids must agree) and the unit is the event.
C7 (H9, family F5 = 2 datasets x 3 tiers): share of violating node-steps attributed to the network layer under P3
(ratio of sums over events, A57 item 5) and the paired damage difference P3 minus P6 (above 0 = P6 lower).
C8 (H10, family F6 = 2 datasets x 3 tiers): per-event R of P3 in coupled mode minus R in constant-m mode.
C9 (H11, family F7 = 2 datasets x 2 metrics): lag tax and R at the smallest phi against phi 1.0, moderate tier.
The label of each hypothesis follows PREREG section 7 through support_label (rules in amendment A72).
"""
import numpy as np
import pandas as pd

from src.lagtax import KEYS
from src.recovery import EVENT, event_recovery, median_interval, recovery_frame
from src.seeds import derive_seed
from src.stats_tools import bca_interval, bca_paired, holm_adjust, mean_stat, paired_wilcoxon, support_label

NAN = float("nan")
COUPLED = "coupled"
CONST_M = "const_m"
REACTIVE = "P1"
HEADLINE = "P3"
HYBRID = "P6"
UNIT = ["dataset", "event_id"]


def tier_levels(h_net_levels):
    """Bandwidth tiers in increasing order: the first is the tight tier."""
    return sorted(h_net_levels)


def select(q3, mode, h_net, phi, m, headroom):
    """Q3 rows of one mode and tier at the default m and headroom; phi None keeps every phi (constant-m rows carry none)."""
    sel = q3[(q3["mode"] == mode) & np.isclose(q3["h_net"].astype(float), h_net)
             & (q3["m"] == m) & (q3["headroom"] == headroom)]
    if phi is not None:
        sel = sel[np.isclose(sel["phi"].astype(float), phi)]
    return sel


def pair_multi(rows_a, rows_b, keys, columns):
    """One-to-one join of two row sets on keys carrying several value columns (suffixes _a and _b). Raises on
    duplicate keys, on rows without a partner and on partners with different event ids."""
    keys = list(keys)
    columns = list(columns)
    for name, rows in (("a", rows_a), ("b", rows_b)):
        if rows.duplicated(keys).any():
            raise ValueError("duplicate pairing keys in set " + name)
    left = rows_a[keys + ["event_id"] + columns].rename(
        columns=dict({"event_id": "event_id_a"}, **{c: c + "_a" for c in columns}))
    right = rows_b[keys + ["event_id"] + columns].rename(
        columns=dict({"event_id": "event_id_b"}, **{c: c + "_b" for c in columns}))
    merged = left.merge(right, on=keys, how="outer", indicator=True)
    if (merged["_merge"] != "both").any():
        raise ValueError("%d rows have no partner (a %d, b %d)" % (
            int((merged["_merge"] != "both").sum()), len(rows_a), len(rows_b)))
    if (merged["event_id_a"] != merged["event_id_b"]).any():
        raise ValueError("paired rows carry different event_id")
    merged = merged.drop(columns=["_merge", "event_id_b"]).rename(columns={"event_id_a": "event_id"})
    return merged.sort_values(keys).reset_index(drop=True)


def ratio_stat(net, total):
    """Sum of the network-attributed violating steps over the sum of the violating steps; nan without violations."""
    denominator = float(np.sum(total))
    return float(np.sum(net) / denominator) if denominator > 0 else NAN


def attribution_share(rows, n_boot, level, seed):
    """Share of violating node-steps attributed to the network layer (A57 item 5): ratio of sums over events with a
    BCa interval from resampling events (the two sums are resampled together)."""
    per = rows.assign(net_steps=rows["attr_net_share"] * rows["attr_violating_steps"]).groupby(
        UNIT, sort=True)[["net_steps", "attr_violating_steps"]].sum()
    net = per["net_steps"].to_numpy()
    total = per["attr_violating_steps"].to_numpy()
    if total.sum() > 0:
        lo, hi = bca_paired(net, total, ratio_stat, n_boot, level, seed)
    else:
        lo, hi = NAN, NAN
    return {"n_violating_steps": int(total.sum()), "share": ratio_stat(net, total), "share_ci_low": lo,
            "share_ci_high": hi}


def c7_table(q3, m, headroom, h_net_levels, phi, h9_share, alpha, min_events, n_boot, level, master_seed):
    """C7 (H9): per dataset and tier, P3 and P6 in coupled mode at phi. Supported at the tight tier when on every
    dataset the share interval lower bound is above h9_share, the mean damage difference (P3 minus P6) is above 0 and
    the Holm-adjusted one-sided Wilcoxon p (over the 6 tests) is below alpha; not supported when on any dataset the
    tight-tier share interval upper bound is below h9_share; inconclusive otherwise or below min_events events."""
    tiers = tier_levels(h_net_levels)
    rows = []
    for dataset in sorted(q3["dataset"].unique()):
        data = q3[q3["dataset"] == dataset]
        for tier in tiers:
            sel = select(data, COUPLED, tier, phi, m, headroom)
            p3 = sel[sel["variant"] == HEADLINE]
            p6 = sel[sel["variant"] == HYBRID]
            if p3.empty or p6.empty:
                raise ValueError("no Q3 rows of P3 and P6 for %s at tier %s" % (dataset, tier))
            paired = pair_multi(p3, p6, KEYS, ["damage"])
            paired["diff"] = paired["damage_a"] - paired["damage_b"]
            d = paired.groupby(UNIT, sort=True)["diff"].mean().to_numpy()
            seed = derive_seed("c7", master_seed, dataset, tier)
            lo, hi = bca_interval(d, mean_stat, n_boot, level, derive_seed(seed, "diff"))
            _stat, p = paired_wilcoxon(d, "greater")
            share = attribution_share(p3, n_boot, level, derive_seed(seed, "share"))
            rows.append(dict(dataset=dataset, h_net=tier, tight=tier == tiers[0], n_events=int(d.size), **share,
                             mean_diff=float(d.mean()), diff_ci_low=lo, diff_ci_high=hi, p_raw=float(p)))
    table = pd.DataFrame(rows)
    table["p_holm"] = holm_adjust(table["p_raw"].to_numpy())
    tight = table[table["tight"]]
    met = bool(((tight["share_ci_low"] > h9_share) & (tight["mean_diff"] > 0) & (tight["p_holm"] < alpha)).all())
    contradicted = bool((tight["share_ci_high"] < h9_share).any())
    table["h9_label"] = support_label(met, contradicted, int(tight["n_events"].min()), min_events)
    return table


def recovery_by_mode(q3, dataset, mode, h_net, phi, m, headroom, min_share):
    """Per-replicate reactive (P1), headline (P3) and oracle (P5) damage with the lag tax and R of one model setting."""
    sel = select(q3[q3["dataset"] == dataset], mode, h_net, phi, m, headroom)
    if sel.empty:
        raise ValueError("no Q3 rows for %s, mode %s, tier %s, phi %s" % (dataset, mode, h_net, phi))
    return recovery_frame(sel, {dataset: REACTIVE}, {dataset: HEADLINE}, min_share)


def c8_table(q3, m, headroom, h_net_levels, phi, min_share, alpha, min_events, n_boot, level, master_seed):
    """C8 (H10): per event R(coupled, phi) minus R(constant m) of P3 at each tier, events with R defined in both
    modes; median difference with BCa and a one-sided Wilcoxon (less). Supported at the tight tier when on every
    dataset the median difference is below 0 and the Holm-adjusted p (over 6) is below alpha; not supported when on
    any dataset the tight-tier interval lies entirely above 0; inconclusive otherwise or below min_events events."""
    tiers = tier_levels(h_net_levels)
    rows = []
    for dataset in sorted(q3["dataset"].unique()):
        for tier in tiers:
            coupled = recovery_by_mode(q3, dataset, COUPLED, tier, phi, m, headroom, min_share)
            const = recovery_by_mode(q3, dataset, CONST_M, tier, None, m, headroom, min_share)
            ec = event_recovery(coupled).rename(columns={"R": "R_coupled"})
            ek = event_recovery(const).rename(columns={"R": "R_const"})
            both = ec.merge(ek, on=EVENT, how="inner", validate="one_to_one")
            d = (both["R_coupled"] - both["R_const"]).to_numpy()
            med, lo, hi = median_interval(d, n_boot, level, derive_seed("c8", master_seed, dataset, tier))
            _stat, p = paired_wilcoxon(d, "less") if d.size else (NAN, 1.0)
            rows.append(dict(dataset=dataset, h_net=tier, tight=tier == tiers[0], n_events=int(d.size),
                             n_events_coupled=int(len(ec)), n_events_const=int(len(ek)),
                             share_defined_coupled=float(coupled["defined"].mean()),
                             share_defined_const=float(const["defined"].mean()),
                             median_coupled=float(both["R_coupled"].median()) if d.size else NAN,
                             median_const=float(both["R_const"].median()) if d.size else NAN,
                             median_diff=med, ci_low=lo, ci_high=hi, p_raw=float(p)))
    table = pd.DataFrame(rows)
    table["p_holm"] = holm_adjust(table["p_raw"].to_numpy())
    tight = table[table["tight"]]
    met = bool(((tight["median_diff"] < 0) & (tight["p_holm"] < alpha)).all())
    contradicted = bool((tight["ci_low"] > 0).any())
    table["h10_label"] = support_label(met, contradicted, int(tight["n_events"].min()), min_events)
    return table


def c9_table(q3, m, headroom, h_net, phi_high, phi_low, min_share, alpha, min_events, n_boot, level, master_seed):
    """C9 (H11): at one tier in coupled mode, per event lag tax (P1 minus P5, mean over replicates) and R of P3 at
    phi_low minus the same at phi_high; mean difference with BCa for the tax, median difference with BCa for R,
    one-sided Wilcoxon (less); Holm over the 4 tests. Supported when all four estimates are below 0 and every
    Holm-adjusted p is below alpha; not supported when any interval lies entirely above 0; inconclusive otherwise
    or below min_events events."""
    if phi_low >= phi_high:
        raise ValueError("phi_low must be smaller than phi_high")
    rows = []
    for dataset in sorted(q3["dataset"].unique()):
        high = recovery_by_mode(q3, dataset, COUPLED, h_net, phi_high, m, headroom, min_share)
        low = recovery_by_mode(q3, dataset, COUPLED, h_net, phi_low, m, headroom, min_share)
        seed = derive_seed("c9", master_seed, dataset)
        tax = pd.concat([high.groupby(UNIT, sort=True)["tax"].mean().rename("tax_high"),
                         low.groupby(UNIT, sort=True)["tax"].mean().rename("tax_low")], axis=1, join="inner")
        d_tax = (tax["tax_low"] - tax["tax_high"]).to_numpy()
        lo, hi = bca_interval(d_tax, mean_stat, n_boot, level, derive_seed(seed, "tax"))
        _stat, p = paired_wilcoxon(d_tax, "less") if d_tax.size else (NAN, 1.0)
        rows.append(dict(dataset=dataset, metric="lag_tax", n_events=int(d_tax.size),
                         value_phi_high=float(tax["tax_high"].mean()), value_phi_low=float(tax["tax_low"].mean()),
                         estimate=float(d_tax.mean()) if d_tax.size else NAN, ci_low=lo, ci_high=hi, p_raw=float(p)))
        r_low = event_recovery(low).rename(columns={"R": "R_low"})
        r_high = event_recovery(high).rename(columns={"R": "R_high"})
        both = r_low.merge(r_high, on=EVENT, how="inner", validate="one_to_one")
        d_r = (both["R_low"] - both["R_high"]).to_numpy()
        med, lo, hi = median_interval(d_r, n_boot, level, derive_seed(seed, "R"))
        _stat, p = paired_wilcoxon(d_r, "less") if d_r.size else (NAN, 1.0)
        rows.append(dict(dataset=dataset, metric="R", n_events=int(d_r.size),
                         value_phi_high=float(both["R_high"].median()) if d_r.size else NAN,
                         value_phi_low=float(both["R_low"].median()) if d_r.size else NAN,
                         estimate=med, ci_low=lo, ci_high=hi, p_raw=float(p)))
    table = pd.DataFrame(rows)
    table["p_holm"] = holm_adjust(table["p_raw"].to_numpy())
    met = bool(((table["estimate"] < 0) & (table["p_holm"] < alpha)).all())
    contradicted = bool((table["ci_low"] > 0).any())
    table["h11_label"] = support_label(met, contradicted, int(table["n_events"].min()), min_events)
    return table
