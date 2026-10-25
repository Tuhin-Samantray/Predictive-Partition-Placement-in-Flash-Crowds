"""C6 (H6): stability of the policy ranking across D1 and D2 (Phase 12, A70). Pure functions; every parameter is an argument.

The policies are ranked by the median over events of the per-event mean damage (classes pooled, default m and headroom)
on each dataset; P3 stands for the headline forecaster variant of that dataset. The statistic is the Kendall tau
between the two datasets' median damages (so between the rankings); supported when tau is at least h6_tau, not
supported when the upper end of a percentile bootstrap interval (events resampled within each dataset) is below
h6_tau, inconclusive otherwise. With few policies the test has little power; the number of policies is reported.
"""
import numpy as np
import pandas as pd
from scipy import stats

from src.seeds import derive_seed
from src.stats_tools import kendall_tau, support_label

NAN = float("nan")
P3 = "P3"
POLICIES = ("P0", "P1", "P2", "P3", "P4", "P5")


def damage_matrix(must, dataset, policies, headline, default_m, headroom):
    """Events x policies matrix of per-event mean damage (mean over replicates) at the default m and headroom."""
    sub = must[(must["dataset"] == dataset) & (must["m"] == default_m) & (must["headroom"] == headroom)]
    wanted = {name: (headline[dataset] if name == P3 else name) for name in policies}
    sub = sub[sub["variant"].isin(wanted.values())]
    label_of = {variant: name for name, variant in wanted.items()}
    sub = sub.assign(policy=sub["variant"].map(label_of))
    wide = sub.groupby(["event_id", "policy"])["damage"].mean().unstack("policy")
    wide = wide.reindex(columns=list(policies))
    if wide.isna().any().any() or wide.empty:
        raise ValueError("every event needs every policy in " + dataset)
    return wide


def c6_tables(must, headline, policies, default_m, headroom, h6_tau, n_boot, level, master_seed):
    """Returns (detail, summary): one detail row per policy, one summary row with tau and its interval."""
    datasets = sorted(must["dataset"].unique())
    mats = {ds: damage_matrix(must, ds, policies, headline, default_m, headroom) for ds in datasets}
    meds = {ds: mats[ds].median(axis=0).to_numpy() for ds in datasets}
    first, second = datasets
    tau = kendall_tau(meds[first], meds[second])
    rng = np.random.default_rng(derive_seed("c6", master_seed))
    boot = {}
    for ds in datasets:
        values = mats[ds].to_numpy()
        idx = rng.integers(0, values.shape[0], size=(n_boot, values.shape[0]))
        boot[ds] = np.median(values[idx], axis=1)
    taus = np.array([kendall_tau(boot[first][b], boot[second][b]) for b in range(n_boot)])
    half = (1 - level) / (1 + 1)
    lo, hi = np.nanquantile(taus, [half, 1 - half]) if np.isfinite(taus).any() else (NAN, NAN)
    detail = pd.DataFrame({"policy": list(policies)})
    for ds in datasets:
        detail["n_events_" + ds] = int(mats[ds].shape[0])
        detail["median_damage_" + ds] = meds[ds]
        detail["rank_" + ds] = stats.rankdata(meds[ds], method="min").astype(int)
    met = bool(np.isfinite(tau) and tau >= h6_tau)
    contradicted = bool(np.isfinite(hi) and hi < h6_tau)
    summary = pd.DataFrame([{"n_policies": len(policies), "tau": tau, "ci_low": float(lo), "ci_high": float(hi),
                             "h6_tau": h6_tau, "met": met, "h6_label": support_label(met, contradicted, len(policies), 1)}])
    return detail, summary
