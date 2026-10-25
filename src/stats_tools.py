"""Statistics helpers for scripts/05_analyze.py (Phase 12, A57). Pure functions; every parameter is an argument.

Events are the cluster unit: a per-event value is the mean over the rows of that event (A57 item 2), and the
bootstrap resamples events. Wilcoxon tests drop zero differences (all zero gives p 1). Labels follow PREREG section 7.
"""
import warnings

import numpy as np
from scipy import stats

SUPPORTED = "supported"
NOT_SUPPORTED = "not supported"
INCONCLUSIVE = "inconclusive"


def event_means(frame, cluster, value):
    """Per-event value: the mean of `value` over the rows of each cluster (cluster is a column name or a list)."""
    return frame.groupby(cluster, sort=True)[value].mean()


def mean_stat(x, axis=-1):
    return np.mean(x, axis=axis)


def median_stat(x, axis=-1):
    return np.median(x, axis=axis)


def bca_interval(values, statistic, n_boot, level, seed):
    """BCa interval of statistic(values, axis) over events. A constant sample gives (value, value);
    fewer than two events or an undefined interval gives (nan, nan)."""
    v = np.asarray(values, dtype=np.float64)
    if v.size < 2 or not np.isfinite(v).all():
        return (float("nan"), float("nan"))
    if np.ptp(v) == 0:
        point = float(statistic(v, axis=-1))
        return (point, point)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = stats.bootstrap((v,), statistic, n_resamples=n_boot, confidence_level=level, method="BCa",
                              vectorized=True, random_state=np.random.default_rng(seed))
    return (float(res.confidence_interval.low), float(res.confidence_interval.high))


def bca_paired(x, y, statistic, n_boot, level, seed):
    """BCa interval of statistic(x, y) over paired clusters (for example a Spearman correlation over cells)."""
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.size != b.size or a.size < 2:
        return (float("nan"), float("nan"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = stats.bootstrap((a, b), statistic, n_resamples=n_boot, confidence_level=level, method="BCa",
                              paired=True, vectorized=False, random_state=np.random.default_rng(seed))
    return (float(res.confidence_interval.low), float(res.confidence_interval.high))


def spearman(x, y):
    return float(stats.spearmanr(x, y).statistic)


def kendall_tau(x, y):
    return float(stats.kendalltau(x, y).statistic)


def paired_wilcoxon(diff, alternative):
    """Signed-rank test on paired differences; zero differences are dropped, nothing left gives p 1."""
    d = np.asarray(diff, dtype=np.float64)
    d = d[d != 0]
    if d.size == 0:
        return float("nan"), 1
    res = stats.wilcoxon(d, alternative=alternative, zero_method="wilcox")
    return float(res.statistic), float(res.pvalue)


def mann_whitney(a, b, alternative):
    res = stats.mannwhitneyu(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64),
                             alternative=alternative)
    return float(res.statistic), float(res.pvalue)


def cliffs_delta(a, b):
    """P(a > b) - P(a < b), from the Mann-Whitney U statistic (ties count one half)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(2 * u / (a.size * b.size) - 1)


def holm_adjust(pvalues):
    """Holm step-down adjusted p values, in the order given, capped at 1."""
    p = np.asarray(pvalues, dtype=np.float64)
    m = p.size
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1)
    return adj


def recovery_ratio(d_reactive, d_x, d_oracle, min_share):
    """Lag tax, recovery ratio R and the defined mask (A57 item 3): R is defined only when the reactive damage is
    positive and the lag tax is at least min_share times the reactive damage; elsewhere R is nan."""
    reactive = np.asarray(d_reactive, dtype=np.float64)
    tax = reactive - np.asarray(d_oracle, dtype=np.float64)
    defined = (reactive > 0) & (tax >= min_share * reactive)
    ratio = np.full(tax.shape, np.nan)
    np.divide(reactive - np.asarray(d_x, dtype=np.float64), tax, out=ratio, where=defined)
    return tax, ratio, defined


def support_label(met, contradicted, n_obs, n_min):
    """PREREG section 7: inconclusive when the sample is below the minimum or the evidence is neither;
    supported when the support rule is met; not supported when the evidence points the other way."""
    if n_obs < n_min:
        return INCONCLUSIVE
    if met:
        return SUPPORTED
    if contradicted:
        return NOT_SUPPORTED
    return INCONCLUSIVE
