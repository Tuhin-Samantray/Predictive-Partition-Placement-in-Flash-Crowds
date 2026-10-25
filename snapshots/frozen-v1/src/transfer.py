"""C5 (H5): transfer of forecasters and of tuned parameters across D1 and D2 (Phase 12, A70). Pure functions; every
parameter is an argument.

Forecaster transfer: the fixed-protocol MASE of a model fitted on the other dataset divided by the in-dataset MASE, per
forecaster, target dataset and stored horizon. Tuning transfer (XA2): per policy and target dataset the mean per-event
damage with the other dataset's tuned parameters divided by the mean with the dataset's own parameters, strictly paired
on dataset, onset class, m, headroom, replicate and policy. H5 rule (PREREG): median ratio at least h5_ratio. Supported
when the median ratio of every forecaster and of the XA2 ratios is at least h5_ratio; not supported when the BCa
interval of any of them lies entirely below h5_ratio; inconclusive otherwise. The number of ratios is reported (low power).
"""
import numpy as np
import pandas as pd

from src.analysis_data import pair_rows
from src.lagtax import KEYS
from src.seeds import derive_seed
from src.stats_tools import bca_interval, mean_stat, median_stat, paired_wilcoxon, support_label

NAN = float("nan")
FIXED = "fixed"
POOLED = "all"
FORECAST = "forecast_ratio"
XA2 = "xa2_ratio"
OVERALL = "overall"
COLUMNS = ["part", "unit", "target", "source", "horizon", "value_in", "value_cross", "ratio", "n_events",
           "regret_mean", "regret_ci_low", "regret_ci_high", "p_wilcoxon_exploratory"]


def forecast_ratios(cross, forecasters, horizons, group):
    fixed = cross[(cross["protocol"] == FIXED) & (cross["group"] == group) & cross["forecaster"].isin(forecasters)]
    rows = []
    for name in forecasters:
        for target in sorted(fixed["target"].unique()):
            for h in horizons:
                sel = fixed[(fixed["forecaster"] == name) & (fixed["target"] == target) & (fixed["horizon"] == h)]
                own = sel[sel["source"] == target]
                foreign = sel[sel["source"] != target]
                if len(own) != 1 or len(foreign) != 1:
                    raise ValueError("need one in-dataset and one cross row for %s %s horizon %s" % (name, target, h))
                m_in = float(own["mase"].iloc[0])
                m_cross = float(foreign["mase"].iloc[0])
                rows.append({"part": FORECAST, "unit": name, "target": target, "source": str(foreign["source"].iloc[0]),
                             "horizon": int(h), "value_in": m_in, "value_cross": m_cross,
                             "ratio": m_cross / m_in if m_in > 0 else NAN})
    return pd.DataFrame(rows)


def xa2_ratios(q8, n_boot, level, master_seed):
    own = q8[q8["level"] == q8["dataset"]]
    foreign = q8[q8["level"] != q8["dataset"]]
    paired = pair_rows(own, foreign, KEYS + ["variant"], "damage")
    source_of = foreign.groupby("dataset")["level"].first()
    per = paired.groupby(["dataset", "variant", "event_id"], sort=True)[["damage_a", "damage_b"]].mean().reset_index()
    rows = []
    for (dataset, variant), sub in per.groupby(["dataset", "variant"], sort=True):
        m_own = float(sub["damage_a"].mean())
        m_for = float(sub["damage_b"].mean())
        diff = (sub["damage_b"] - sub["damage_a"]).to_numpy()
        lo, hi = bca_interval(diff, mean_stat, n_boot, level, derive_seed("c5", master_seed, "xa2", dataset, variant))
        _stat, p = paired_wilcoxon(diff, "two-sided")
        rows.append({"part": XA2, "unit": variant, "target": dataset, "source": str(source_of[dataset]),
                     "value_in": m_own, "value_cross": m_for, "ratio": m_for / m_own if m_own > 0 else NAN,
                     "n_events": int(len(sub)), "regret_mean": float(diff.mean()), "regret_ci_low": lo,
                     "regret_ci_high": hi, "p_wilcoxon_exploratory": float(p)})
    return pd.DataFrame(rows)


def c5_tables(cross, q8, forecasters, horizons, group, h5_ratio, n_boot, level, master_seed):
    """Returns (detail, summary): one detail row per ratio, one summary row per forecaster, one for XA2 and an overall row."""
    detail = pd.concat([forecast_ratios(cross, forecasters, horizons, group), xa2_ratios(q8, n_boot, level, master_seed)],
                       ignore_index=True)
    detail = detail.reindex(columns=COLUMNS)
    detail["horizon"] = detail["horizon"].astype("Int64")
    units = [(FORECAST, name) for name in forecasters] + [(XA2, None)]
    rows = []
    for part, unit in units:
        sub = detail[detail["part"] == part]
        if unit is not None:
            sub = sub[sub["unit"] == unit]
        values = sub["ratio"].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        lo, hi = bca_interval(values, median_stat, n_boot, level, derive_seed("c5", master_seed, part, str(unit)))
        row = {"part": part, "unit": unit if unit is not None else "all policies", "n_ratios": int(values.size),
               "median_ratio": float(np.median(values)) if values.size else NAN, "ci_low": lo, "ci_high": hi,
               "h5_ratio": h5_ratio}
        for target, tsub in sub.groupby("target", sort=True):
            tv = tsub["ratio"].to_numpy(dtype=np.float64)
            tv = tv[np.isfinite(tv)]
            row["median_target_" + target] = float(np.median(tv)) if tv.size else NAN
        row["met"] = bool(values.size > 0 and row["median_ratio"] >= h5_ratio)
        row["below"] = bool(np.isfinite(hi) and hi < h5_ratio)
        rows.append(row)
    summary = pd.DataFrame(rows)
    label = support_label(bool(summary["met"].all()), bool(summary["below"].any()), int(summary["n_ratios"].min()), 1)
    summary = pd.concat([summary, pd.DataFrame([{"part": OVERALL, "unit": "H5", "n_ratios": int(summary["n_ratios"].min()),
                                                 "h5_ratio": h5_ratio, "met": bool(summary["met"].all()),
                                                 "below": bool(summary["below"].any()), "h5_label": label}])],
                        ignore_index=True)
    return detail, summary
