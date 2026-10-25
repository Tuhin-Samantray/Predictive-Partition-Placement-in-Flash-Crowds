"""Recovery ratio, C2 and C3 (Phase 12, A62, A64). Pure functions; every parameter is an argument.

R = (reactive - X) / (reactive - oracle) per replicate, defined only when the lag tax is at least min_share of the
reactive damage (A57 item 3); per-event R is the median of the defined replicate values (A62). X is the headline
forecast policy (P3 with the forecaster chosen for P4). C2 (H2): median R by onset class at the default m and
headroom. C3 (H3): median R at the smallest m level against the m levels at least the default m. Events are the
cluster unit.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.analysis_data import pair_rows, select_variant
from src.lagtax import KEYS, ORACLE, lag_tax_frame
from src.seeds import derive_seed
from src.stats_tools import (bca_interval, cliffs_delta, holm_adjust, mann_whitney, median_stat, paired_wilcoxon,
                             recovery_ratio, support_label)

ABRUPT = "abrupt"
RAMPED = "ramped"
NAN = float("nan")
EVENT = ["dataset", "onset_class", "event_id"]


def load_headline(path):
    """Headline forecast policy per dataset: P3 with the forecaster chosen for P4 (A13), from configs/tuned.yaml."""
    tuned = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["tuned"]
    return {dataset: "P3_" + tuned[dataset]["P4"]["forecaster"] for dataset in tuned}


def recovery_frame(must, best_reactive, headline, min_share, oracle=ORACLE):
    """Per-replicate damage of the reactive policy, the headline policy and the oracle, strictly paired on KEYS
    (rows without a partner or with different event ids raise), with the lag tax, R and the defined flag."""
    paired = lag_tax_frame(must, best_reactive, oracle)
    react = select_variant(must, best_reactive)
    px = pair_rows(react, select_variant(must, headline), KEYS, "damage")
    x = px[KEYS + ["event_id", "damage_b"]].rename(columns={"event_id": "event_id_x", "damage_b": "damage_x"})
    merged = paired.merge(x, on=KEYS, how="outer", indicator=True, validate="one_to_one")
    if (merged["_merge"] != "both").any():
        raise ValueError("reactive, headline and oracle rows are not one-to-one on " + ", ".join(KEYS))
    if (merged["event_id"] != merged["event_id_x"]).any():
        raise ValueError("paired rows carry different event_id")
    tax, ratio, defined = recovery_ratio(merged["damage_a"].to_numpy(), merged["damage_x"].to_numpy(),
                                         merged["damage_b"].to_numpy(), min_share)
    out = merged.drop(columns=["_merge", "event_id_x"]).rename(
        columns={"damage_a": "damage_reactive", "damage_b": "damage_oracle"})
    out["tax"] = tax
    out["R"] = ratio
    out["defined"] = defined
    return out


def event_recovery(frame):
    """Per-event R: the median of the defined replicate values of each event (events without one drop out)."""
    return frame[frame["defined"]].groupby(EVENT, sort=True)["R"].median().reset_index()


def median_interval(values, n_boot, level, seed):
    if values.size == 0:
        return NAN, NAN, NAN
    lo, hi = bca_interval(values, median_stat, n_boot, level, seed)
    return float(np.median(values)), lo, hi


def c2_table(rec, default_m, headroom, h2_split, n_boot, level, alpha, min_events, master_seed):
    """C2 (H2, family F2 = 2 datasets): median R of abrupt and ramped events with BCa, one-sided Mann-Whitney U
    (ramped greater), Cliff's delta (ramped against abrupt). Supported when the abrupt median is below h2_split,
    the ramped median above it, both intervals exclude it and the Holm-adjusted U p is below alpha on every
    dataset; not supported when on any dataset the abrupt interval lies entirely above h2_split and the ramped
    interval entirely below it (A64); inconclusive otherwise or below min_events events in a class."""
    rows = []
    for dataset in sorted(rec["dataset"].unique()):
        sub = rec[(rec["dataset"] == dataset) & (rec["m"] == default_m) & (rec["headroom"] == headroom)]
        if sub.empty:
            raise ValueError("no rows for %s at m %s and headroom %s" % (dataset, default_m, headroom))
        per = event_recovery(sub)
        abrupt = per.loc[per["onset_class"] == ABRUPT, "R"].to_numpy()
        ramped = per.loc[per["onset_class"] == RAMPED, "R"].to_numpy()
        m_a, lo_a, hi_a = median_interval(abrupt, n_boot, level, derive_seed("c2", master_seed, dataset, ABRUPT))
        m_r, lo_r, hi_r = median_interval(ramped, n_boot, level, derive_seed("c2", master_seed, dataset, RAMPED))
        both = abrupt.size > 0 and ramped.size > 0
        u, p = mann_whitney(ramped, abrupt, "greater") if both else (NAN, 1.0)
        delta = cliffs_delta(ramped, abrupt) if both else NAN
        rows.append(dict(dataset=dataset, n_events_abrupt=int(abrupt.size), n_events_ramped=int(ramped.size),
                         share_defined=float(sub["defined"].mean()), median_abrupt=m_a, abrupt_ci_low=lo_a,
                         abrupt_ci_high=hi_a, median_ramped=m_r, ramped_ci_low=lo_r, ramped_ci_high=hi_r,
                         u_stat=u, p_raw=p, cliffs_delta=delta))
    table = pd.DataFrame(rows)
    table["p_holm"] = holm_adjust(table["p_raw"].to_numpy())
    met = ((table["median_abrupt"] < h2_split) & (table["median_ramped"] > h2_split)
           & (table["abrupt_ci_high"] < h2_split) & (table["ramped_ci_low"] > h2_split) & (table["p_holm"] < alpha))
    flipped = (table["abrupt_ci_low"] > h2_split) & (table["ramped_ci_high"] < h2_split)
    table["met"] = met
    n_obs = int(table[["n_events_abrupt", "n_events_ramped"]].to_numpy().min())
    table["h2_label"] = support_label(bool(met.all()), bool(flipped.any()), n_obs, min_events)
    return table


def c3_table(rec, m_levels, default_m, headroom, h3_ub, n_boot, level, alpha, min_events, master_seed):
    """C3 (H3, family F3 = 2 datasets): median R at the smallest m level (classes pooled) with BCa; supported when
    its upper bound is below h3_ub on every dataset, not supported when the lower bound is above h3_ub on any
    dataset (A64), inconclusive otherwise or below min_events events. Mann-Whitney U (one-sided, low m smaller) of
    per-event R at the low m against per-event R at the m levels at least default_m (median over those levels),
    on the events with R defined in both; the same events appear at every m, so the groups are not independent
    (limitation); a paired Wilcoxon on the same pairs is an exploratory sensitivity column outside the family."""
    m_low = min(m_levels)
    high = [m for m in m_levels if m >= default_m]
    rows = []
    for dataset in sorted(rec["dataset"].unique()):
        at = rec[(rec["dataset"] == dataset) & (rec["headroom"] == headroom)]
        low = at[at["m"] == m_low]
        if low.empty or not high:
            raise ValueError("no rows at m %s in %s" % (m_low, dataset))
        per_low = event_recovery(low)[["dataset", "event_id", "R"]]
        kept = at[at["m"].isin(high) & at["defined"]]
        by_m = kept.groupby(["dataset", "event_id", "m"])["R"].median().reset_index()
        per_high = by_m.groupby(["dataset", "event_id"])["R"].median().reset_index()
        pairs = per_low.merge(per_high, on=["dataset", "event_id"], suffixes=("_low", "_high"))
        values = per_low["R"].to_numpy()
        med, lo, hi = median_interval(values, n_boot, level, derive_seed("c3", master_seed, dataset))
        if len(pairs):
            u, p = mann_whitney(pairs["R_low"].to_numpy(), pairs["R_high"].to_numpy(), "less")
            _stat, p_w = paired_wilcoxon(pairs["R_low"].to_numpy() - pairs["R_high"].to_numpy(), "less")
        else:
            u, p, p_w = NAN, 1.0, 1.0
        rows.append(dict(dataset=dataset, m_low=m_low, n_events_m_low=int(values.size),
                         share_defined_m_low=float(low["defined"].mean()), median_m_low=med, ci_low=lo, ci_high=hi,
                         n_pairs=int(len(pairs)), median_pairs_m_high=float(pairs["R_high"].median()) if len(pairs) else NAN,
                         u_stat=u, p_raw=p, p_wilcoxon_exploratory=float(p_w)))
    table = pd.DataFrame(rows)
    table["p_holm"] = holm_adjust(table["p_raw"].to_numpy())
    met = table["ci_high"] < h3_ub
    table["met"] = met
    n_obs = int(table["n_events_m_low"].min())
    table["h3_label"] = support_label(bool(met.all()), bool((table["ci_low"] > h3_ub).any()), n_obs, min_events)
    return table
