"""Regime maps (Phase 12, A66). Pure functions; every parameter is an argument.

A regime map is the median recovery ratio R per cell with its BCa interval, next to the event count, the share of
replicates with R defined and the median lag tax (A57, A62, A66). Q2 map: D3 rise bucket x m at the default headroom
for the Q2 forecast variant of A36. Must-grid map: onset class x m x headroom for the headline policy and for P4.
Events are the cluster unit (per-event R is the median of its defined replicate values). Cells with fewer than
min_events events with R defined are marked and kept; an interval needs at least two events with spread.
"""
import numpy as np
import pandas as pd

from src.recovery import event_recovery, recovery_frame
from src.seeds import derive_seed
from src.stats_tools import bca_interval, median_stat

D3 = "_D3"
CELL = ["dataset", "onset_class", "m", "headroom"]
NAN = float("nan")
NO_R = "no R defined"
FEW = "few events"
NO_CI = "no interval"
CONSTANT = "R constant"


def cell_interval(values, n_boot, level, seed):
    """BCa interval of the median of the per-event R values; NaN with fewer than two events or without spread."""
    if values.size < 2 or np.ptp(values) == 0:
        return NAN, NAN
    return bca_interval(values, median_stat, n_boot, level, seed)


def cell_note(values, min_events):
    notes = []
    if values.size == 0:
        notes.append(NO_R)
    elif values.size < min_events:
        notes.append(FEW)
    if values.size < 2:
        notes.append(NO_CI)
    elif np.ptp(values) == 0:
        notes.append(CONSTANT)
    return "; ".join(notes)


def regime_table(rec, policy_by_dataset, n_boot, level, min_events, master_seed, label):
    """One row per (dataset, onset_class, m, headroom) cell of a recovery frame."""
    rows = []
    for (dataset, cls, m, headroom), sub in rec.groupby(CELL, sort=True):
        policy = policy_by_dataset[dataset]
        values = event_recovery(sub)["R"].to_numpy()
        seed = derive_seed(label, master_seed, dataset, cls, str(m), str(headroom), policy)
        lo, hi = cell_interval(values, n_boot, level, seed)
        rows.append(dict(
            dataset=dataset, onset_class=cls, m=m, headroom=headroom, policy=policy,
            n_replicates=int(len(sub)), n_events=int(sub["event_id"].nunique()), n_events_defined=int(values.size),
            share_defined=float(sub["defined"].mean()),
            median_tax=float(sub.groupby("event_id")["tax"].mean().median()),
            median_R=float(np.median(values)) if values.size else NAN, ci_low=lo, ci_high=hi,
            note=cell_note(values, min_events)))
    return pd.DataFrame(rows)


def q2_mappings(best_reactive, variants):
    """Reactive reference and Q2 forecast variant per D3 dataset name (A36: the variants come from the config)."""
    reactive = {dataset + D3: name for dataset, name in best_reactive.items()}
    headline = {dataset + D3: name for dataset, name in variants.items()}
    return reactive, headline


def regime_q2(q2, best_reactive, variants, min_share, n_boot, level, min_events, master_seed):
    reactive, headline = q2_mappings(best_reactive, variants)
    rec = recovery_frame(q2, reactive, headline, min_share)
    return regime_table(rec, headline, n_boot, level, min_events, master_seed, "regime_q2")


def regime_must(must, best_reactive, policies, min_share, n_boot, level, min_events, master_seed):
    """policies: list of {dataset: variant} mappings (the headline P3 and P4), one block of rows each."""
    parts = []
    for mapping in policies:
        rec = recovery_frame(must, best_reactive, mapping, min_share)
        parts.append(regime_table(rec, mapping, n_boot, level, min_events, master_seed, "regime_must"))
    return pd.concat(parts, ignore_index=True)
