"""Descriptive policy and contention tables for the Phase 14 report (A89). Pure functions; every parameter is an argument.

Post-freeze, descriptive, not pre-registered: no hypothesis label, no Holm family, no pooling across datasets (the
damage scales differ by orders of magnitude). The unit is the event: the per-event value of a metric is the mean
over the replicates of that event, the interval is BCa over events. Policy table: for each dataset and policy at the
default m and headroom, the mean, median and standard deviation across events and the BCa interval of damage,
slo_violation_frac, peak_backlog_steps and moves, plus the ratio to the reference policy P1 (sum over events of the
policy metric over the sum of the P1 metric, events resampled in pairs; nan where the P1 sum is not positive).
Sources are those of RV0: Must rows (P0, P1, P2, the headline P3, P4, P5), Q1 AA6 rows at the default margin (P4a),
and the D3 variants from q11_rv0. Contention table: the same four metrics for the RV1 rows (policy x level).
"""
import numpy as np
import pandas as pd

from src.analysis_data import select_variant
from src.maincomp import default_rows, p4a_rows, plain
from src.rvtables import (D3_SUFFIX, LEVELS, P3_PREFIX, RV1_FACTOR, RV1_POLICIES, aligned, need, per_event,
                          ratio_of_sums, safe_paired)
from src.seeds import derive_seed
from src.stats_tools import bca_interval, mean_stat

NAN = float("nan")
METRICS = ["damage", "slo_violation_frac", "peak_backlog_steps", "moves"]
REFERENCE = "P1"
POLICIES = ["P0", "P1", "P2", "P3", "P4", "P4a", "P5"]


def column_stats(per, n_boot, level, seed):
    """Mean, median, standard deviation (n minus 1) and BCa interval of the mean of per-event values (nulls skipped)."""
    values = per.dropna().to_numpy(dtype=np.float64)
    if values.size == 0:
        return {"n_events": 0, "mean": NAN, "median": NAN, "sd": NAN, "ci_low": NAN, "ci_high": NAN}
    lo, hi = bca_interval(values, mean_stat, n_boot, level, seed)
    return {"n_events": int(values.size), "mean": float(values.mean()), "median": float(np.median(values)),
            "sd": float(np.std(values, ddof=1)) if values.size > 1 else NAN, "ci_low": lo, "ci_high": hi}


def ratio_fields(per, ref, policy, n_boot, level, seed):
    """Ratio of the policy to the reference over events present in both (events must be the same)."""
    aligned(ref, per, "the ratio to " + REFERENCE)
    keep = per.notna().to_numpy() & ref.notna().to_numpy()
    x = per.to_numpy(dtype=np.float64)[keep]
    y = ref.to_numpy(dtype=np.float64)[keep]
    ratio = ratio_of_sums(x, y)
    if policy == REFERENCE:
        lo, hi = ratio, ratio
    else:
        lo, hi = safe_paired(x, y, n_boot, level, seed)
    return {"ratio_to_p1": ratio, "ratio_ci_low": lo, "ratio_ci_high": hi, "n_events_paired": int(x.size)}


def policy_frames(must, q1, rv0, headline, m, headroom, margin_quantile):
    """{dataset: {policy: (rows, source)}} with the RV0 sources at the default m and headroom."""
    base = default_rows(must, m, headroom)
    specs = [(name, plain(base, name), "must:" + name) for name in ("P0", "P1", "P2")]
    specs.append(("P3", select_variant(base, headline), "must:" + "+".join(sorted(set(headline.values())))))
    specs.append(("P4", plain(base, "P4"), "must:P4"))
    specs.append(("P4a", p4a_rows(q1, m, headroom, margin_quantile), "q1:AA6 P4a margin_quantile=%s" % margin_quantile))
    specs.append(("P5", plain(base, "P5"), "must:P5"))
    out = {}
    for policy, frame, source in specs:
        need(frame, "policy metrics " + policy)
        for dataset, sub in frame.groupby("dataset", sort=True):
            out.setdefault(dataset, {})[policy] = (sub, source)
    third = default_rows(rv0, m, headroom)
    third = need(third[third["dataset"].str.endswith(D3_SUFFIX)], "policy metrics D3")
    for (dataset, variant), sub in third.groupby(["dataset", "variant"], sort=True):
        policy = "P3" if variant.startswith(P3_PREFIX) else variant
        out.setdefault(dataset, {})[policy] = (sub, "q11_rv0:" + variant)
    return out


def policy_metrics_table(must, q1, rv0, headline, m, headroom, margin_quantile, n_boot, level, master_seed):
    frames = policy_frames(must, q1, rv0, headline, m, headroom, margin_quantile)
    rows = []
    for dataset in sorted(frames):
        if REFERENCE not in frames[dataset]:
            raise ValueError("no %s rows for %s" % (REFERENCE, dataset))
        reference = frames[dataset][REFERENCE][0]
        for metric in METRICS:
            ref = per_event(reference, metric)
            for policy in POLICIES:
                if policy not in frames[dataset]:
                    continue
                frame, source = frames[dataset][policy]
                per = per_event(frame, metric)
                row = dict(dataset=dataset, policy=policy, metric=metric, source=source, n_replicates=int(len(frame)))
                row.update(column_stats(per, n_boot, level, derive_seed("rt", master_seed, dataset, policy, metric)))
                row.update(ratio_fields(per, ref, policy, n_boot, level,
                                        derive_seed("rtratio", master_seed, dataset, policy, metric)))
                rows.append(row)
    return pd.DataFrame(rows)


def rv1_metrics_table(rv123, m, headroom, n_boot, level, master_seed):
    """The four metrics of the RV1 rows per dataset, policy and contention level."""
    base = default_rows(rv123, m, headroom)
    base = need(base[base["factor"] == RV1_FACTOR], "RV1 metrics")
    rows = []
    for dataset in sorted(base["dataset"].unique()):
        for policy in RV1_POLICIES:
            for lv in LEVELS:
                sub = need(base[(base["dataset"] == dataset) & (base["variant"] == policy) & (base["level"] == lv)],
                           "RV1 metrics %s %s %s" % (dataset, policy, lv))
                for metric in METRICS:
                    row = dict(dataset=dataset, policy=policy, level=lv, mode=str(sub["mode"].iloc[0]), metric=metric,
                               n_replicates=int(len(sub)))
                    row.update(column_stats(per_event(sub, metric), n_boot, level,
                                            derive_seed("rtrv1", master_seed, dataset, policy, lv, metric)))
                    rows.append(row)
    return pd.DataFrame(rows)
