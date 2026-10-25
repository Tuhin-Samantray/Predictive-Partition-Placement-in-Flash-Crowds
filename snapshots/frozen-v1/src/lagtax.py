"""Lag tax and the C1 statistics (Phase 12, A62). Pure functions; every parameter is an argument.

Lag tax = damage of the best tuned reactive policy minus damage of the oracle P5, per replicate, strictly paired
(A57 item 3). C1 (H1), per dataset: a one-sided Wilcoxon test of the per-event tax at the default m and headroom
against 0, and the Spearman correlation of tax with m over the m levels at the default headroom. Events are the
cluster unit: the interval of rho resamples events and its p value permutes the m labels within each event (A62).
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from src.analysis_data import event_frame, pair_rows, select_variant
from src.seeds import derive_seed
from src.stats_tools import bca_interval, holm_adjust, mean_stat, paired_wilcoxon, spearman, support_label

KEYS = ["dataset", "onset_class", "m", "headroom", "replicate"]
ORACLE = "P5"
WILCOXON = "wilcoxon_tax"
SPEARMAN = "spearman_tax_m"
NAN = float("nan")


def load_best_reactive(path):
    """best_reactive of configs/tuned.yaml: the tuned reactive policy per dataset (A57 item 1)."""
    return dict(yaml.safe_load(Path(path).read_text(encoding="utf-8"))["best_reactive"])


def lag_tax_frame(must, best_reactive, oracle=ORACLE):
    """Per-replicate lag tax (reactive minus oracle damage), strictly paired on KEYS; columns tax, damage_a (reactive)."""
    react = select_variant(must, best_reactive)
    orac = must[must["variant"] == oracle]
    paired = pair_rows(react, orac, KEYS, "damage")
    paired["tax"] = paired["damage_a"] - paired["damage_b"]
    return paired


def default_events(paired, dataset, m, headroom):
    """Per-event mean tax of one dataset at the default m and headroom, classes pooled."""
    sub = paired[(paired["dataset"] == dataset) & (paired["m"] == m) & (paired["headroom"] == headroom)]
    if sub.empty:
        raise ValueError("no rows for %s at m %s and headroom %s" % (dataset, m, headroom))
    return event_frame(sub, ["dataset", "event_id"], ["tax", "damage_a"])


def tax_test(events, n_boot, level, seed):
    """Mean per-event tax with a BCa interval and the one-sided Wilcoxon p of tax above 0."""
    tax = events["tax"].to_numpy()
    lo, hi = bca_interval(tax, mean_stat, n_boot, level, seed)
    _stat, p = paired_wilcoxon(tax, "greater")
    return {"n_events": int(tax.size), "n_nonzero": int((tax != 0).sum()), "estimate": float(tax.mean()),
            "ci_low": lo, "ci_high": hi, "p_raw": float(p)}


def tax_matrix(paired, dataset, headroom, m_levels):
    """Events x m levels matrix of the per-event mean tax at one headroom; every event needs every level."""
    levels = sorted(m_levels)
    sub = paired[(paired["dataset"] == dataset) & (paired["headroom"] == headroom) & paired["m"].isin(levels)]
    wide = sub.groupby(["event_id", "m"])["tax"].mean().unstack("m")
    if list(wide.columns) != levels or wide.isna().any().any():
        raise ValueError("every event needs a tax at every m level in " + dataset)
    return wide.to_numpy(), np.asarray(levels, dtype=np.float64)


def pooled_rho(tax, m_values):
    """Spearman correlation of tax with m over all (event, m) points."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return spearman(tax.ravel(), np.tile(m_values, tax.shape[0]))


def rho_interval(tax, m_values, n_boot, level, seed):
    """BCa interval of the pooled rho, resampling events (rows of the matrix)."""
    if tax.shape[0] < 2 or np.ptp(tax) == 0:
        return (NAN, NAN)

    def statistic(idx):
        return pooled_rho(tax[np.asarray(idx, dtype=np.int64)], m_values)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = stats.bootstrap((np.arange(tax.shape[0]),), statistic, n_resamples=n_boot, confidence_level=level,
                              method="BCa", vectorized=False, random_state=np.random.default_rng(seed))
    return (float(res.confidence_interval.low), float(res.confidence_interval.high))


def permutation_p(tax, m_values, n_perm, seed):
    """One-sided p of rho above 0: the m labels are permuted within each event (null: no dependence of tax on m
    inside an event), the pooled rho is recomputed, p = (1 + count(rho_perm >= rho)) / (1 + n_perm)."""
    obs = pooled_rho(tax, m_values)
    if not np.isfinite(obs):
        return 1.0
    n, k = tax.shape
    r_tax = stats.rankdata(tax.ravel()).reshape(n, k)
    r_m = stats.rankdata(m_values)
    a = r_tax - r_tax.mean()
    order = np.argsort(np.random.default_rng(seed).random((n_perm, n, k)), axis=-1)
    b = r_m[order] - r_m.mean()
    rho = (a[None] * b).sum(axis=(1, 2)) / np.sqrt((a ** 2).sum() * (b ** 2).sum(axis=(1, 2)))
    tol = 8 * np.finfo(np.float64).eps
    return float((1 + np.sum(rho >= obs - tol)) / (1 + n_perm))


def spearman_test(tax, m_values, n_boot, level, seed_ci, seed_perm):
    lo, hi = rho_interval(tax, m_values, n_boot, level, seed_ci)
    return {"n_events": int(tax.shape[0]), "n_nonzero": None, "estimate": pooled_rho(tax, m_values),
            "ci_low": lo, "ci_high": hi, "p_raw": permutation_p(tax, m_values, n_boot, seed_perm)}


def c1_label(table, alpha, min_events):
    """H1 support rule (guide section 5): tax and rho above 0 and both Holm-adjusted p below alpha on every dataset;
    contradicted when an interval lies entirely below 0; inconclusive otherwise or below min_events events."""
    met = bool(((table["estimate"] > 0) & (table["p_holm"] < alpha)).all())
    contradicted = bool((table["ci_high"] < 0).any())
    return support_label(met, contradicted, int(table["n_events"].min()), min_events)


def c1_table(must, best_reactive, default_m, headroom, m_levels, n_boot, level, alpha, min_events, master_seed):
    """C1 table: per dataset the Wilcoxon row and the Spearman row, Holm over the family F1 (2 datasets x 2 tests)."""
    paired = lag_tax_frame(must, best_reactive)
    rows = []
    for dataset in sorted(paired["dataset"].unique()):
        events = default_events(paired, dataset, default_m, headroom)
        seed = derive_seed("c1", master_seed, dataset)
        rows.append(dict(dataset=dataset, test=WILCOXON, **tax_test(events, n_boot, level, derive_seed(seed, WILCOXON))))
        tax, m_values = tax_matrix(paired, dataset, headroom, m_levels)
        rows.append(dict(dataset=dataset, test=SPEARMAN,
                         **spearman_test(tax, m_values, n_boot, level, derive_seed(seed, SPEARMAN, "ci"),
                                         derive_seed(seed, SPEARMAN, "perm"))))
    table = pd.DataFrame(rows)
    table["p_holm"] = holm_adjust(table["p_raw"].to_numpy())
    table["h1_label"] = c1_label(table, alpha, min_events)
    return table
