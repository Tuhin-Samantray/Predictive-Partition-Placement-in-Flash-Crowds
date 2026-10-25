"""Tables TA3 to TA9 (A80), assembled from tables that are already recorded. No new statistic is computed here."""
import numpy as np
import pandas as pd

from src.xatables import FULL, SUBSET

ACCURACY = ["dataset", "forecaster", "split", "horizon", "mase_all", "smape_all", "mase_event", "smape_event",
            "n_cells_all", "n_cells_event", "n_zero_scale"]
MAIN = ["block", "comparison", "family", "dataset", "onset_class", "n_events", "mean_reference", "mean_policy",
        "mean_diff", "ci_low", "ci_high", "direction", "p_holm", "holm_below_alpha"]
ABLATIONS = ["family", "ablation", "dataset", "variant", "param", "level", "n_events", "mean_reference", "mean_policy",
             "mean_diff", "ci_low", "ci_high", "direction", "p_holm", "holm_below_alpha"]
CROSS = ["block", "key", "estimate", "ci_low", "ci_high", "threshold", "met", "label"]
STATS = ["id", "hypothesis", "holm_family", "n_tests", "p_holm_min", "label", "status"]
CONFIRMATORY = "confirmatory"
ABLATION = "ablation"


def ta3(main, baselines, chronos):
    """Forecast accuracy, evaluation split: all-origin rows (main table first, baseline rows only where the main table
    lacks the forecaster) and the stride-4 subset rows of the Chronos table, marked by origin_subset."""
    key = ["dataset", "forecaster", "horizon"]
    full = pd.concat([main, baselines], ignore_index=True)
    full = full[full["split"] == "eval"].drop_duplicates(subset=key, keep="first")
    full = full.reindex(columns=ACCURACY).assign(origin_subset=FULL, n_origins=np.nan)
    sub = chronos[chronos["split"] == "eval"].reindex(columns=ACCURACY + ["n_origins"]).assign(origin_subset=SUBSET)
    out = pd.concat([full, sub], ignore_index=True)
    return out.sort_values(["origin_subset"] + key).reset_index(drop=True)


def ta4(c11, c13):
    first = c11.assign(comparison=c11["policy"].astype(str) + " vs " + c11["reference"].astype(str), block="C11")
    second = c13.assign(block="C13")
    return pd.concat([first.reindex(columns=MAIN), second.reindex(columns=MAIN)], ignore_index=True)


def ta5(q2, must):
    both = [q2.assign(block="Q2 D3 rise bucket"), must.assign(block="Must grid onset class")]
    out = pd.concat(both, ignore_index=True)
    return out[["block"] + [c for c in out.columns if c != "block"]]


def ta6(aa):
    return aa.reindex(columns=ABLATIONS)


def ta7(c5_summary, c6_tau, xa1):
    """Cross-data block: C5 summary rows, the C6 tau, and the XA1 ratios of the all-cells group."""
    c5 = pd.DataFrame({"block": "C5", "key": c5_summary["part"].astype(str) + ":" + c5_summary["unit"].astype(str),
                       "estimate": c5_summary["median_ratio"], "ci_low": c5_summary["ci_low"],
                       "ci_high": c5_summary["ci_high"], "threshold": c5_summary["h5_ratio"],
                       "met": c5_summary["met"], "label": c5_summary["h5_label"]})
    c6 = pd.DataFrame({"block": "C6", "key": "kendall_tau", "estimate": c6_tau["tau"], "ci_low": c6_tau["ci_low"],
                       "ci_high": c6_tau["ci_high"], "threshold": c6_tau["h6_tau"], "met": c6_tau["met"],
                       "label": c6_tau["h6_label"]})
    cross = xa1[(xa1["cell"] == "cross-dataset") & (xa1["group"] == "all")]
    keys = (cross["forecaster"].astype(str) + ":h" + cross["horizon"].astype(int).astype(str) + ":" +
            cross["source"].astype(str) + "->" + cross["target"].astype(str))
    x = pd.DataFrame({"block": "XA1", "key": keys, "estimate": cross["ratio"]})
    return pd.concat([c5, c6, x], ignore_index=True).reindex(columns=CROSS)


def ta8(parts, cl3, mode, phi):
    """Cross-layer block: C7, C8, C9, CL3 (coupled, default phi), CL4 and RV1 rows stacked with a block column."""
    cl3 = cl3[(cl3["mode"] == mode) & (cl3["phi"] == phi)]
    frames = [frame.assign(block=name) for name, frame in parts if name != "CL3"]
    frames.append(cl3.assign(block="CL3"))
    out = pd.concat(frames, ignore_index=True)
    return out[["block"] + [c for c in out.columns if c != "block"]]


def ta9(ledger):
    """Statistics: families, test counts, Holm p and labels from the ledger, with totals by status."""
    body = ledger.reindex(columns=STATS)
    counts = ledger["n_tests"].astype(int)
    status = ledger["status"].astype(str)
    totals = []
    for name, mask in (("TOTAL confirmatory", status.str.startswith(CONFIRMATORY)),
                       ("TOTAL ablation families", status.str.startswith(ABLATION)),
                       ("TOTAL all rows", status.str.len() > 0)):
        totals.append({"id": name, "hypothesis": "", "holm_family": "sum of n_tests", "n_tests": int(counts[mask].sum()),
                       "p_holm_min": np.nan, "label": "", "status": ""})
    return pd.concat([body, pd.DataFrame(totals)], ignore_index=True)
