"""Interaction trigger (A38, A57, A60): which Q1 factors are active and whether the pre-registered trigger fires.

A level is compared with the default-level row of the same replicate (strict pairing); the effect is the mean over events
of the paired damage difference with a BCa interval. A factor is active when at least one level has an interval that
excludes zero and a mean difference of at least effect_frac times the mean damage of the default rows.
"""
import pandas as pd

from src.analysis_data import event_frame, pair_rows
from src.seeds import derive_seed
from src.stats_tools import bca_interval, mean_stat

KEYS = ["dataset", "onset_class", "m", "headroom", "replicate"]
LEVEL_KEYS = ["dataset", "variant", "factor", "param", "level_key"]


def is_default_p4a(q1, margin_quantile):
    return (q1["variant"] == "P4a") & (q1["factor"] == "AA6") & (q1["level"] == margin_quantile)


def reference_rows(q1, must, default_m, default_headroom, margin_quantile):
    """Default-level rows: the Must rows at the default m and headroom, and for P4a the Q1 AA6 rows at the default margin."""
    base = must[(must["m"] == default_m) & (must["headroom"] == default_headroom)]
    return pd.concat([base, q1[is_default_p4a(q1, margin_quantile)]], ignore_index=True)


def interaction_effects(q1, must, default_m, default_headroom, margin_quantile, n_boot, level, effect_frac,
                        master_seed):
    ref = reference_rows(q1, must, default_m, default_headroom, margin_quantile)
    tested = q1[~is_default_p4a(q1, margin_quantile)].copy()
    tested["level_key"] = tested["level"].astype(str)
    rows = []
    for key, group in tested.groupby(LEVEL_KEYS, sort=True):
        dataset, variant, factor, param, level_key = key
        refs = ref[(ref["dataset"] == dataset) & (ref["variant"] == variant)]
        paired = pair_rows(refs, group, KEYS, "damage")
        paired["diff"] = paired["damage_b"] - paired["damage_a"]
        events = event_frame(paired, ["dataset", "event_id"], ["damage_a", "damage_b", "diff"])
        seed = derive_seed("interaction", master_seed, dataset, variant, factor, param, level_key)
        lo, hi = bca_interval(events["diff"].to_numpy(), mean_stat, n_boot, level, seed)
        mean_diff = float(events["diff"].mean())
        ref_mean = float(events["damage_a"].mean())
        excludes = bool(lo > 0 or hi < 0)
        large = bool(ref_mean > 0 and abs(mean_diff) >= effect_frac * ref_mean)
        rows.append({"dataset": dataset, "variant": variant, "factor": factor, "param": param, "level": level_key,
                     "n_events": len(events), "mean_diff": mean_diff, "ci_low": lo, "ci_high": hi,
                     "ref_mean": ref_mean, "rel_effect": mean_diff / ref_mean if ref_mean > 0 else float("nan"),
                     "excludes_zero": excludes, "large": large, "active": excludes and large})
    return pd.DataFrame(rows)


def interaction_trigger(effects, min_active):
    """Per dataset and base policy: the active factors and whether at least min_active of them fire the trigger."""
    rows = []
    for (dataset, variant), group in effects.groupby(["dataset", "variant"], sort=True):
        active = sorted(set(group.loc[group["active"], "factor"]))
        rows.append({"dataset": dataset, "variant": variant, "n_factors_tested": int(group["factor"].nunique()),
                     "n_active_factors": len(active), "active_factors": ",".join(active),
                     "fires": len(active) >= min_active})
    return pd.DataFrame(rows)
