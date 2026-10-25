"""Forecast-only cross-data tables XA1, XA5 and XA7 (A80). Descriptive: no new test, no hypothesis label.

Rows come from tables/forecast_cross.csv (Q5, A37), the Phase 6 early-warning tables and the Chronos tables (A53).
Nothing is recomputed from forecasts and nothing is invented: a table that lacks a breakdown stays without it.
"""
import numpy as np
import pandas as pd

FIXED = "fixed"
ROLLING = "rolling"
ALL = "all"
EVENT = "event"
NONEVENT = "nonevent"
DAY = "day"
GROUPS = [ALL, EVENT, NONEVENT]
FULL = "all_origins"
SUBSET = "stride_subset"
EVAL = "eval"


def safe_ratio(num, den):
    """num over den, NaN where the denominator is not above 0."""
    den = pd.Series(den, dtype="float64")
    return pd.Series(num, dtype="float64") / den.where(den > 0)


def xa1_table(cross):
    """XA1 transfer matrix: fixed-protocol MASE of every (source, target) pair against the in-dataset cell of the same
    forecaster, horizon and group (ratio above 1 = transfer is worse), plus the Phase 6 rolling in-dataset MASE."""
    keys = ["forecaster", "horizon", "group"]
    fixed = cross[cross["protocol"] == FIXED]
    own = fixed[fixed["source"] == fixed["target"]][["target"] + keys + ["mase", "smape"]]
    own = own.rename(columns={"mase": "mase_in", "smape": "smape_in"})
    out = fixed.merge(own, on=["target"] + keys, how="left", validate="many_to_one")
    rolling = cross[(cross["protocol"] == ROLLING) & (cross["source"] == cross["target"])]
    rolling = rolling[["target"] + keys + ["mase"]].rename(columns={"mase": "mase_rolling_in"})
    out = out.merge(rolling, on=["target"] + keys, how="left", validate="many_to_one")
    out["cell"] = np.where(out["source"] == out["target"], "in-dataset", "cross-dataset")
    out["ratio"] = safe_ratio(out["mase"], out["mase_in"])
    out["ratio_smape"] = safe_ratio(out["smape"], out["smape_in"])
    cols = ["protocol", "source", "target", "forecaster", "hp", "horizon", "group", "cell", "n_cells", "n_mase_cells",
            "mase", "smape", "mase_in", "ratio", "ratio_smape", "mase_rolling_in"]
    order = ["forecaster", "horizon", "group", "source", "target"]
    return out[cols].sort_values(order).reset_index(drop=True)


def day_index(group):
    return int(group[len(DAY):]) if group.startswith(DAY) else -1


def xa5_table(cross):
    """XA5 accuracy by evaluation day (groups day0, day1, ... of forecast_cross.csv, both protocols) next to the whole
    evaluation span; a day with fewer scored cells than the largest day of the same series is flagged partial."""
    keys = ["protocol", "source", "target", "forecaster", "horizon"]
    sel = cross[cross["group"].str.startswith(DAY) | (cross["group"] == ALL)].copy()
    sel["day"] = sel["group"].map(day_index)
    days = sel[sel["day"] >= 0]
    biggest = days.groupby(keys)["n_cells"].max().rename("max_day_cells").reset_index()
    whole = sel[sel["day"] < 0][keys + ["mase"]].rename(columns={"mase": "mase_all_days"})
    out = sel.merge(biggest, on=keys, how="left").merge(whole, on=keys, how="left", validate="many_to_one")
    out["partial_day"] = (out["day"] >= 0) & (out["n_cells"] < out["max_day_cells"])
    out["ratio_to_all_days"] = safe_ratio(out["mase"], out["mase_all_days"])
    cols = keys + ["hp", "group", "day", "n_cells", "n_mase_cells", "max_day_cells", "partial_day", "mase", "smape",
                   "mase_all_days", "ratio_to_all_days"]
    return out[cols].sort_values(keys + ["day"]).reset_index(drop=True)


def xa7_mase_table(cross, chronos):
    """XA7 MASE overall, inside event windows and outside them. All-origin rows: rolling in-dataset rows of
    forecast_cross.csv. Chronos subset rows: the stride-4 origin subset of the Chronos table (all six forecasters on the
    identical origins), which has no outside-event column; the two origin sets are never mixed in one row."""
    roll = cross[(cross["protocol"] == ROLLING) & (cross["source"] == cross["target"]) & cross["group"].isin(GROUPS)]
    wide = roll.pivot(index=["target", "forecaster", "horizon"], columns="group", values="mase")
    wide = wide.reindex(columns=GROUPS).reset_index()
    wide.columns = ["dataset", "forecaster", "horizon", "mase_all", "mase_event", "mase_nonevent"]
    wide["origin_subset"] = FULL
    wide["n_origins"] = np.nan
    sub = chronos[chronos["split"] == EVAL][["dataset", "forecaster", "horizon", "mase_all", "mase_event", "n_origins"]].copy()
    sub["mase_nonevent"] = np.nan
    sub["origin_subset"] = SUBSET
    out = pd.concat([wide, sub], ignore_index=True)
    out["ratio_event_all"] = safe_ratio(out["mase_event"], out["mase_all"])
    out["ratio_event_nonevent"] = safe_ratio(out["mase_event"], out["mase_nonevent"])
    cols = ["origin_subset", "dataset", "forecaster", "horizon", "n_origins", "mase_all", "mase_event", "mase_nonevent",
            "ratio_event_all", "ratio_event_nonevent"]
    return out[cols].sort_values(["origin_subset", "dataset", "forecaster", "horizon"]).reset_index(drop=True)


WARN_KEY = ["dataset", "forecaster", "split", "onset_class"]
WARN_VALUES = ["recall", "lead_median", "precision"]


def baseline_overlap_mismatch(main, baselines):
    """Number of rows present in both early-warning tables whose recall, lead or precision differ."""
    both = main.merge(baselines, on=WARN_KEY, suffixes=("", "_b"))
    bad = np.zeros(len(both), dtype=bool)
    for col in WARN_VALUES:
        bad |= ~np.isclose(both[col].astype(float), both[col + "_b"].astype(float), equal_nan=True)
    return int(bad.sum())


def xa7_warning_table(main, baselines, chronos):
    """XA7 early warning (recall, lead, precision) per forecaster, dataset, split and onset class. Main table first, the
    baseline table only for forecasters the main table lacks; Chronos rows are the stride-4 subset and stay separate."""
    full = pd.concat([main, baselines], ignore_index=True).drop_duplicates(subset=WARN_KEY, keep="first")
    full = full.assign(origin_subset=FULL, n_origins=np.nan)
    sub = chronos.assign(origin_subset=SUBSET)
    out = pd.concat([full, sub], ignore_index=True)
    cols = ["origin_subset", "dataset", "forecaster", "split", "onset_class", "n_origins", "n_events",
            "n_excluded_events", "recall", "lead_median", "lead_mean", "n_alarm_cells", "precision"]
    return out[cols].sort_values(["origin_subset", "dataset", "forecaster", "split", "onset_class"]).reset_index(drop=True)
