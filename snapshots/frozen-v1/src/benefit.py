"""C4 (H4): forecast accuracy against placement benefit (Phase 12, A70). Pure functions; every parameter is an argument.

A cell is (forecaster, dataset, onset class). The benefit of a cell is the median over its events of the per-event R of
the policy P3 with that forecaster (A57 item 3, default m and headroom); the accuracy of a cell is the MASE inside the
evaluation-split events of that onset class at one stored horizon (the definition of accuracy_rows, A9, with the event
mask restricted to the class). H4 is tested with the Spearman correlation of MASE with benefit over the cells (below 0
means more accurate forecasters bring more benefit) with a BCa interval over cells.
"""
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from src.accuracy import error_sums, event_cells, ratio, scales
from src.recovery import event_recovery, recovery_frame
from src.seeds import derive_seed
from src.stats_tools import bca_paired, spearman, support_label

NAN = float("nan")
CELL = ["forecaster", "dataset", "onset_class"]
P3_PREFIX = "P3_"
POOLED = "all"


def class_event_mase(table, load, events, train_end, tune_end, horizons, horizon, smape_eps):
    """MASE of one forecast table inside the evaluation-split events of each onset class and of all events (key POOLED)
    at one stored horizon. Returns {class: (mase, n_mase_cells)}."""
    n_steps, n_part = load.shape
    j = list(horizons).index(horizon)
    scale = scales(load, train_end, horizons)
    lo, hi = tune_end, n_steps
    pred = np.asarray(table[lo:hi - horizon, j, :], dtype=np.float64)
    truth = np.asarray(load[lo + horizon:hi], dtype=np.float64)
    ev = events[events["split"] == "eval"]
    groups = {POOLED: ev}
    for cls in sorted(ev["onset_class"].unique()):
        groups[cls] = ev[ev["onset_class"] == cls]
    out = {}
    for name, sub in groups.items():
        mask = event_cells(sub, n_steps, n_part)[lo + horizon:hi]
        sums = error_sums(pred, truth, scale[j], smape_eps, mask)
        out[name] = (ratio(sums["mase_sum"], sums["n_mase_cells"]), sums["n_mase_cells"])
    return out


def benefit_cells(must, best_reactive, forecasters, default_m, headroom, min_share):
    """Per cell: the number of events with R defined and the median of their per-event R (nan when there is none)."""
    rows = []
    for name in forecasters:
        headline = {dataset: P3_PREFIX + name for dataset in sorted(must["dataset"].unique())}
        rec = recovery_frame(must, best_reactive, headline, min_share)
        sub = rec[(rec["m"] == default_m) & (rec["headroom"] == headroom)]
        per = event_recovery(sub)
        got = per.groupby(["dataset", "onset_class"])["R"].agg(["median", "size"]).reset_index()
        grid = sub[["dataset", "onset_class"]].drop_duplicates()
        merged = grid.merge(got, on=["dataset", "onset_class"], how="left")
        for _, r in merged.iterrows():
            rows.append({"forecaster": name, "dataset": r["dataset"], "onset_class": r["onset_class"],
                         "n_events_r": int(r["size"]) if pd.notna(r["size"]) else 0,
                         "benefit": float(r["median"]) if pd.notna(r["median"]) else NAN})
    return pd.DataFrame(rows).sort_values(CELL).reset_index(drop=True)


def c4_table(cells, mase, h4_rho, h4_ub, n_boot, level, master_seed):
    """Returns (detail, summary). Support: |rho| below h4_rho and the larger absolute end of the BCa interval below
    h4_ub; not supported when |rho| is at least h4_rho and the interval excludes 0; otherwise inconclusive (low power:
    the number of cells is reported, PREREG). Cells without a defined benefit or MASE drop out and are counted."""
    detail = cells.merge(mase, on=CELL, how="left", validate="one_to_one")
    used = detail[np.isfinite(detail["benefit"]) & np.isfinite(detail["mase_event_class"])]
    x = used["mase_event_class"].to_numpy()
    y = used["benefit"].to_numpy()
    rho, p_exploratory, lo, hi = NAN, NAN, NAN, NAN
    if used.shape[0] > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho = spearman(x, y)
            p_exploratory = float(stats.spearmanr(x, y).pvalue)
        lo, hi = bca_paired(x, y, spearman, n_boot, level, derive_seed("c4", master_seed))
    finite = bool(np.isfinite([rho, lo, hi]).all())
    bound = max(abs(lo), abs(hi)) if finite else NAN
    met = bool(finite and abs(rho) < h4_rho and bound < h4_ub)
    contradicted = bool(finite and abs(rho) >= h4_rho and (lo > 0 or hi < 0))
    summary = pd.DataFrame([{
        "n_cells_total": int(len(detail)), "n_cells_used": int(used.shape[0]),
        "n_cells_without_benefit": int(detail["benefit"].isna().sum()),
        "rho": rho, "ci_low": lo, "ci_high": hi, "abs_ci_bound": bound, "h4_rho": h4_rho, "h4_ub": h4_ub,
        "p_spearman_exploratory": p_exploratory, "met": met,
        "h4_label": support_label(met, contradicted, used.shape[0], 1)}])
    return detail, summary
