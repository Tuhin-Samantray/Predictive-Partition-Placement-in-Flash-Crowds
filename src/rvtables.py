"""Review tables RV0 to RV4 (Phase 12, A76). Pure functions; every parameter is an argument.

Descriptive tables: no hypothesis label, no Holm family. The unit is the event (per-event value = mean over its
replicates), intervals are BCa over events. RV0 gives the mean per-event damage of the seven policies (D1 and D2 from
the Must rows and the Q1 AA6 P4a rows, D3 from q11_rv0). RV1 gives, per policy and contention row, the mean damage,
the paired difference to the full row and the recovery fraction (full minus off over full minus both off), which is
nan where the denominator is not positive; censored migration durations are counted, not imputed. RV2 and RV3 are
paired differences of the Q11 rows (skewed placement, throttle budget 1) against the same events at balanced
placement and the default budget. RV4 reports the logged forecast-interval fields; the interval is one-sided and no
coverage guarantee is claimed (item 13a of the caveats).
"""
import numpy as np
import pandas as pd

from src.analysis_data import select_variant
from src.maincomp import compare_table, default_rows, p4a_rows, paired_frame, plain
from src.seeds import derive_seed
from src.stats_tools import bca_interval, bca_paired, mean_stat

NAN = float("nan")
FULL = "full"
CPU_OFF = "cpu_off"
NIC_OFF = "nic_off"
BOTH_OFF = "both_off"
LEVELS = [FULL, CPU_OFF, NIC_OFF, BOTH_OFF]
OFF_ROWS = [CPU_OFF, NIC_OFF]
RV1_POLICIES = ["P1", "P3", "P4a", "P5"]
RV23_POLICIES = ["P0", "P1", "P4a"]
RV3_POLICIES = ["P1", "P4a"]
RV4_POLICIES = ["P4", "P4a"]
RV4_FACTOR = "RV4"
RV1_FACTOR = "RV1"
RV2_FACTOR = "RV2"
RV3_FACTOR = "RV3"
BALANCED = "balanced"
SKEWED = "skewed"
D3_SUFFIX = "_D3"
P3_PREFIX = "P3_"
RV4_METRICS = ["rv4_point_mae", "rv4_point_bias", "rv4_fix_cov", "rv4_fix_width", "rv4_ad_cov", "rv4_ad_width",
               "rv4_ad_alpha_mean", "rv4_ad_alpha_last", "trig_decisions", "trig_no_slot_steps", "trig_proposed",
               "trig_proposed_steps"]
RV4_POOLED = [("rv4_fix_pooled_coverage", "rv4_fix_hits"), ("rv4_ad_pooled_coverage", "rv4_ad_hits")]
TRIALS = "rv4_trials"
DURATION = "mig_duration_mean"
CENSORED = "mig_censored"


def cluster_cols(frame):
    return [c for c in ("dataset", "onset_class") if c in frame.columns] + ["event_id"]


def need(frame, what):
    if len(frame) == 0:
        raise ValueError("no rows for " + what)
    return frame


def per_event(frame, column):
    """Per-event mean of one column (nulls skipped), indexed by the cluster columns."""
    return frame.groupby(cluster_cols(frame), sort=True)[column].mean()


def aligned(first, second, what):
    if not first.index.equals(second.index):
        raise ValueError("events differ between the rows of " + what)


def ratio_of_sums(x, y):
    d = np.sum(y)
    return float(np.sum(x) / d) if d > 0 else NAN


def safe_paired(x, y, n_boot, level, seed):
    try:
        return bca_paired(x, y, ratio_of_sums, n_boot, level, seed)
    except (ValueError, FloatingPointError, ZeroDivisionError):
        return (NAN, NAN)


def damage_row(frame, n_boot, level, seed):
    per = per_event(frame, "damage").to_numpy(dtype=np.float64)
    lo, hi = bca_interval(per, mean_stat, n_boot, level, seed)
    return {"n_events": int(per.size), "n_replicates": int(len(frame)), "mean_damage": float(per.mean()),
            "median_damage": float(np.median(per)), "ci_low": lo, "ci_high": hi}


def rv0_table(must, q1, rv0, headline, m, headroom, margin_quantile, n_boot, level, master_seed):
    base = default_rows(must, m, headroom)
    specs = [(name, plain(base, name), "must:" + name) for name in ("P0", "P1", "P2")]
    specs.append(("P3", select_variant(base, headline), "must:" + "+".join(sorted(set(headline.values())))))
    specs.append(("P4", plain(base, "P4"), "must:P4"))
    specs.append(("P4a", p4a_rows(q1, m, headroom, margin_quantile), "q1:AA6 P4a margin_quantile=%s" % margin_quantile))
    specs.append(("P5", plain(base, "P5"), "must:P5"))
    rows = []
    for policy, frame, source in specs:
        need(frame, "RV0 " + policy)
        for dataset, sub in frame.groupby("dataset", sort=True):
            seed = derive_seed("rv0", master_seed, dataset, policy)
            rows.append(dict(dataset=dataset, policy=policy, source=source, **damage_row(sub, n_boot, level, seed)))
    third = default_rows(rv0, m, headroom)
    third = need(third[third["dataset"].str.endswith(D3_SUFFIX)], "RV0 D3")
    for (dataset, variant), sub in third.groupby(["dataset", "variant"], sort=True):
        policy = "P3" if variant.startswith(P3_PREFIX) else variant
        seed = derive_seed("rv0", master_seed, dataset, policy)
        rows.append(dict(dataset=dataset, policy=policy, source="q11_rv0:" + variant,
                         **damage_row(sub, n_boot, level, seed)))
    return pd.DataFrame(rows)


def censor_fields(frame):
    duration = pd.to_numeric(frame[DURATION], errors="coerce")
    censored = pd.to_numeric(frame[CENSORED], errors="coerce")
    return {"censored_moves": float(censored.sum()), "rows_without_duration": int(duration.isna().sum()),
            "mig_duration_mean": float(duration.mean())}


def rv1_table(rv123, m, headroom, n_boot, level, master_seed):
    base = default_rows(rv123, m, headroom)
    base = need(base[base["factor"] == RV1_FACTOR], "RV1")
    rows = []
    for dataset in sorted(base["dataset"].unique()):
        for policy in RV1_POLICIES:
            sub = need(base[(base["dataset"] == dataset) & (base["variant"] == policy)], "RV1 %s %s" % (dataset, policy))
            by = {lv: need(sub[sub["level"] == lv], "RV1 %s %s %s" % (dataset, policy, lv)) for lv in LEVELS}
            pe = {lv: per_event(by[lv], "damage") for lv in LEVELS}
            for lv in LEVELS:
                aligned(pe[FULL], pe[lv], "RV1 %s %s %s" % (dataset, policy, lv))
            denominator = pe[FULL] - pe[BOTH_OFF]
            for lv in LEVELS:
                seed = derive_seed("rv1", master_seed, dataset, policy, lv)
                row = dict(dataset=dataset, policy=policy, level=lv, mode=str(by[lv]["mode"].iloc[0]))
                row.update(damage_row(by[lv], n_boot, level, seed))
                diff = (pe[lv] - pe[FULL]).to_numpy(dtype=np.float64)
                lo, hi = bca_interval(diff, mean_stat, n_boot, level, derive_seed("rv1diff", master_seed, dataset, policy, lv))
                row.update(mean_diff_to_full=float(diff.mean()), diff_ci_low=lo, diff_ci_high=hi)
                frac, flo, fhi, defined = NAN, NAN, NAN, False
                if lv in OFF_ROWS and float(denominator.mean()) > 0:
                    x = (pe[FULL] - pe[lv]).to_numpy(dtype=np.float64)
                    y = denominator.to_numpy(dtype=np.float64)
                    frac = ratio_of_sums(x, y)
                    flo, fhi = safe_paired(x, y, n_boot, level, derive_seed("rv1frac", master_seed, dataset, policy, lv))
                    defined = True
                row.update(recovery_fraction=frac, recovery_ci_low=flo, recovery_ci_high=fhi, recovery_defined=defined)
                row.update(censor_fields(by[lv]))
                rows.append(row)
    return pd.DataFrame(rows)


def contrast(policy_rows, reference_rows, factor, comparison, policy, source, n_boot, level, master_seed):
    table = compare_table(paired_frame(policy_rows, reference_rows), ["dataset"], n_boot, level, master_seed,
                          "%s:%s:%s" % (factor, comparison, policy))
    head = pd.DataFrame({"factor": factor, "comparison": comparison, "policy": policy, "reference_source": source},
                        index=table.index)
    return pd.concat([head, table], axis=1)


def default_balanced(base_must, q1, policy, m, headroom, margin_quantile):
    if policy == "P4a":
        return need(p4a_rows(q1, m, headroom, margin_quantile), "balanced P4a"), "q1:AA6 P4a margin_quantile=%s" % margin_quantile
    return need(plain(base_must, policy), "balanced " + policy), "must:" + policy


def rv23_tables(must, q1, rv123, m, headroom, margin_quantile, n_boot, level, master_seed):
    base_must = default_rows(must, m, headroom)
    base = default_rows(rv123, m, headroom)
    rv2_rows = base[base["factor"] == RV2_FACTOR]
    rv3_rows = base[base["factor"] == RV3_FACTOR]
    rv2, rv3 = [], []
    for policy in RV23_POLICIES:
        reference, source = default_balanced(base_must, q1, policy, m, headroom, margin_quantile)
        skewed = need(rv2_rows[rv2_rows["variant"] == policy], "RV2 " + policy)
        rv2.append(contrast(skewed, reference, RV2_FACTOR, "skewed minus balanced", policy, source, n_boot, level,
                            master_seed))
    for policy in RV3_POLICIES:
        reference, source = default_balanced(base_must, q1, policy, m, headroom, margin_quantile)
        mine = rv3_rows[rv3_rows["variant"] == policy]
        balanced = need(mine[mine["placement"] == BALANCED], "RV3 balanced " + policy)
        rv3.append(contrast(balanced, reference, RV3_FACTOR, "budget 1 minus default, balanced", policy, source, n_boot,
                            level, master_seed))
        skew_budget = need(mine[mine["placement"] == SKEWED], "RV3 skewed " + policy)
        skew_default = need(rv2_rows[rv2_rows["variant"] == policy], "RV2 " + policy)
        rv3.append(contrast(skew_budget, skew_default, RV3_FACTOR, "budget 1 minus default, skewed", policy,
                            "q11_rv123:RV2 " + policy, n_boot, level, master_seed))
    return pd.concat(rv2, ignore_index=True), pd.concat(rv3, ignore_index=True)


def rv4_table(rv0, m, headroom, n_boot, level, master_seed):
    base = default_rows(rv0, m, headroom)
    base = need(base[(base["factor"] == RV4_FACTOR) & base["variant"].isin(RV4_POLICIES)], "RV4")
    rows = []
    for (dataset, variant), sub in base.groupby(["dataset", "variant"], sort=True):
        for metric in RV4_METRICS:
            per = per_event(sub, metric).dropna().to_numpy(dtype=np.float64)
            seed = derive_seed("rv4", master_seed, dataset, variant, metric)
            lo, hi = bca_interval(per, mean_stat, n_boot, level, seed) if per.size else (NAN, NAN)
            rows.append(dict(dataset=dataset, variant=variant, metric=metric, n_events=int(per.size),
                             n_null_rows=int(sub[metric].isna().sum()),
                             mean=float(per.mean()) if per.size else NAN, ci_low=lo, ci_high=hi))
        sums = sub.groupby(cluster_cols(sub), sort=True)[[hits for _n, hits in RV4_POOLED] + [TRIALS]].sum()
        for name, hits in RV4_POOLED:
            x = sums[hits].to_numpy(dtype=np.float64)
            y = sums[TRIALS].to_numpy(dtype=np.float64)
            lo, hi = safe_paired(x, y, n_boot, level, derive_seed("rv4", master_seed, dataset, variant, name))
            rows.append(dict(dataset=dataset, variant=variant, metric=name, n_events=int(x.size),
                             n_null_rows=int(sub[hits].isna().sum()), mean=ratio_of_sums(x, y), ci_low=lo, ci_high=hi))
    return pd.DataFrame(rows)
