"""Main comparisons C11 and C13 (Phase 12, A68). Pure functions; every parameter is an argument.

C11 (family F9 = policies x datasets x onset class): each of P0, P2, P3 (headline), P4, P4a, P5 and P6 against the best
tuned reactive policy, at the default m and headroom. C13 (family F10 = 5 pairs x datasets): P4a against each of P1, P0,
P3 (headline), P4 and P5, onset classes pooled. Both are strictly paired on dataset, onset class, m, headroom and
replicate (event ids must agree). The unit is the event: per-event value = mean over its replicates; the mean paired
difference (policy minus reference, below 0 = the policy has lower damage) has a BCa interval over events; the
two-sided Wilcoxon signed-rank test runs on the per-event differences (zero differences dropped); Cliff's delta
compares the per-event damage of the policy with that of the reference (below 0 = the policy tends lower). A
replicate-level Wilcoxon p value is added as an exploratory column. Results are reported in full (no support label).
"""
import numpy as np
import pandas as pd

from src.analysis_data import pair_rows, select_variant
from src.lagtax import KEYS
from src.seeds import derive_seed
from src.stats_tools import bca_interval, cliffs_delta, holm_adjust, mean_stat, paired_wilcoxon

NAN = float("nan")
F9 = "F9"
F10 = "F10"
EXPLORATORY = "exploratory"
LOWER = "lower damage"
HIGHER = "higher damage"
EQUAL = "equal"
LAYERS_OFF = "layers off"
LAYERS_ON = "cross-layer model, coupled"
GROUP11 = ["dataset", "onset_class"]
GROUP13 = ["dataset"]
P3_PREFIX = "P3_"


def default_rows(frame, m, headroom):
    return frame[(frame["m"] == m) & (frame["headroom"] == headroom)]


def p4a_rows(q1, m, headroom, margin_quantile):
    """P4a rows of Q1 AA6 at the default margin level (the P4a default rows, A34), default m and headroom."""
    sel = q1[(q1["factor"] == "AA6") & (q1["param"] == "margin_quantile") & (q1["variant"] == "P4a")
             & np.isclose(q1["applied"].astype(float), margin_quantile)]
    return default_rows(sel, m, headroom)


def p6_rows(q3, m, headroom, h_net, phi):
    """Q3 rows of the coupled mode at one bandwidth tier and one phi, default m and headroom."""
    sel = q3[(q3["mode"] == "coupled") & np.isclose(q3["h_net"].astype(float), h_net)
             & np.isclose(q3["phi"].astype(float), phi)]
    return default_rows(sel, m, headroom)


def plain(frame, name):
    return frame[frame["variant"] == name]


def paired_frame(policy_rows, reference_rows):
    """Strict one-to-one pairing on KEYS with an event id check; diff = policy damage minus reference damage."""
    paired = pair_rows(reference_rows, policy_rows, KEYS, "damage")
    paired = paired.rename(columns={"damage_a": "damage_reference", "damage_b": "damage_policy"})
    paired["diff"] = paired["damage_policy"] - paired["damage_reference"]
    return paired


def direction(mean_diff):
    if mean_diff < 0:
        return LOWER
    if mean_diff > 0:
        return HIGHER
    return EQUAL


def compare_table(paired, group, n_boot, level, master_seed, label):
    """One row per group (for example dataset and onset class) of a paired frame."""
    rows = []
    for key, sub in paired.groupby(group, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        per = sub.groupby(group + ["event_id"], sort=True)[["diff", "damage_reference", "damage_policy"]].mean()
        d = per["diff"].to_numpy()
        seed = derive_seed(label, master_seed, *[str(k) for k in key])
        lo, hi = bca_interval(d, mean_stat, n_boot, level, seed)
        _stat, p = paired_wilcoxon(d, "two-sided")
        _stat_rep, p_rep = paired_wilcoxon(sub["diff"].to_numpy(), "two-sided")
        ref_mean = float(per["damage_reference"].mean())
        row = dict(zip(group, key))
        row.update(n_events=int(d.size), n_replicates=int(len(sub)), n_nonzero=int((d != 0).sum()),
                   mean_reference=ref_mean, mean_policy=float(per["damage_policy"].mean()),
                   mean_diff=float(d.mean()), median_diff=float(np.median(d)), ci_low=lo, ci_high=hi,
                   relative_diff=float(d.mean() / ref_mean) if ref_mean > 0 else NAN,
                   direction=direction(float(d.mean())),
                   cliffs_delta=cliffs_delta(per["damage_policy"].to_numpy(), per["damage_reference"].to_numpy()),
                   p_raw=float(p), p_wilcoxon_replicate_exploratory=float(p_rep))
        rows.append(row)
    return pd.DataFrame(rows)


def c11_table(must, q1, q3, best_reactive, headline, m, headroom, margin_quantile, h_net, phi, alpha, n_boot, level,
              master_seed):
    base = default_rows(must, m, headroom)
    reference = select_variant(base, best_reactive)
    specs = [
        ("P0", F9, LAYERS_OFF, "must:P0", plain(base, "P0"), reference),
        ("P2", F9, LAYERS_OFF, "must:P2", plain(base, "P2"), reference),
        ("P3", F9, LAYERS_OFF, "must:" + "+".join(sorted(set(headline.values()))), select_variant(base, headline),
         reference),
        ("P4", F9, LAYERS_OFF, "must:P4", plain(base, "P4"), reference),
        ("P4a", F9, LAYERS_OFF, "q1:AA6 P4a margin_quantile=%s" % margin_quantile,
         p4a_rows(q1, m, headroom, margin_quantile), reference),
        ("P5", F9, LAYERS_OFF, "must:P5", plain(base, "P5"), reference)]
    q3_sel = p6_rows(q3, m, headroom, h_net, phi)
    specs.append(("P6", F9, LAYERS_ON, "q3:coupled h_net=%s phi=%s, reference q3 %s" % (
        h_net, phi, "+".join(sorted(set(best_reactive.values())))),
        plain(q3_sel, "P6"), select_variant(q3_sel, best_reactive)))
    heads = set(headline.values())
    for name in sorted(v for v in base["variant"].unique() if v.startswith(P3_PREFIX) and v not in heads):
        specs.append((name, EXPLORATORY, LAYERS_OFF, "must:" + name, plain(base, name), reference))
    parts = []
    for policy, family, model, source, rows, ref in specs:
        table = compare_table(paired_frame(rows, ref), GROUP11, n_boot, level, master_seed, "c11:" + policy)
        table.insert(0, "policy", policy)
        table.insert(1, "family", family)
        table.insert(2, "model", model)
        table.insert(3, "source", source)
        parts.append(table)
    out = pd.concat(parts, ignore_index=True)
    out["reference"] = out["dataset"].map(best_reactive)
    mask = (out["family"] == F9).to_numpy()
    out["p_holm"] = NAN
    out.loc[mask, "p_holm"] = holm_adjust(out.loc[mask, "p_raw"].to_numpy())
    out["holm_below_alpha"] = [bool(p < alpha) if flag else None for p, flag in zip(out["p_holm"], mask)]
    return out


def c13_table(must, q1, headline, m, headroom, margin_quantile, alpha, n_boot, level, master_seed):
    base = default_rows(must, m, headroom)
    policy = p4a_rows(q1, m, headroom, margin_quantile)
    others = [("P1", plain(base, "P1")), ("P0", plain(base, "P0")), ("P3", select_variant(base, headline)),
              ("P4", plain(base, "P4")), ("P5", plain(base, "P5"))]
    parts = []
    for name, ref in others:
        table = compare_table(paired_frame(policy, ref), GROUP13, n_boot, level, master_seed, "c13:" + name)
        table.insert(0, "comparison", "P4a vs " + name)
        table.insert(1, "family", F10)
        parts.append(table)
    out = pd.concat(parts, ignore_index=True)
    out["p_holm"] = holm_adjust(out["p_raw"].to_numpy())
    out["holm_below_alpha"] = out["p_holm"] < alpha
    return out
