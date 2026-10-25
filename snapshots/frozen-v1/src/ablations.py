"""Ablation tables AA1 to AA11 (Phase 12, A78). Pure functions; every parameter is an argument.

One table, one row per ablation level and dataset (and policy). Each level is compared with the default level of the
same dataset and policy: strict pairing on dataset, onset class, m, headroom and replicate with an event id check
(A68), per-event mean, BCa interval over events, two-sided Wilcoxon on the per-event differences, Cliff's delta.
The difference is level minus reference (below 0 = the level has lower damage). Reference rows: the Must rows at the
default m and headroom of the same dataset and policy; for P4a the Q1 AA6 rows at the default margin (A38). AA1 compares
the other P3 forecasters with the headline P3 forecaster; AA11 compares the layers-off rows of Q4 with the coupled Q3
rows at the configured bandwidth tier and phi (off minus on). The Holm family is one per ablation (C12), over
dataset x policy x level. Reported in full: no support label. Levels that were not run are not invented.
"""
import numpy as np
import pandas as pd

from src.analysis_data import select_variant
from src.interaction import is_default_p4a, reference_rows
from src.maincomp import compare_table, default_rows, p6_rows, paired_frame
from src.stats_tools import holm_adjust

NAN = float("nan")
AA1 = "AA1"
AA11 = "AA11"
P3_PREFIX = "P3_"
FORECASTER = "forecaster"
LAYERS = "layers"
GROUP = ["dataset"]
LEVEL_KEYS = ["dataset", "variant", "factor", "param", "level_key"]
META = ["family", "ablation", "dataset", "variant", "param", "level", "reference"]


def one_level(level_rows, ref_rows, label, n_boot, level, master_seed):
    if len(level_rows) == 0 or len(ref_rows) == 0:
        raise ValueError("no rows to compare for " + label)
    return compare_table(paired_frame(level_rows, ref_rows), GROUP, n_boot, level, master_seed, label).iloc[0].to_dict()


def finish(row, ablation, variant, param, level_key, reference):
    row.update(family="C12:" + ablation, ablation=ablation, variant=variant, param=param, level=level_key,
               reference=reference)
    return row


def reference_text(variant, margin_quantile):
    if variant == "P4a":
        return "q1:AA6 P4a margin_quantile=%s" % margin_quantile
    return "must:" + variant


def factor_rows(tested, ref, margin_quantile, n_boot, level, master_seed):
    """Every (dataset, policy, factor, level) group of a queue against the default rows of the same dataset and policy."""
    tested = tested.copy()
    tested["level_key"] = tested["level"].astype(str)
    rows = []
    for key, group in tested.groupby(LEVEL_KEYS, sort=True):
        dataset, variant, factor, param, level_key = key
        refs = ref[(ref["dataset"] == dataset) & (ref["variant"] == variant)]
        label = "aa:%s:%s:%s:%s:%s" % (factor, dataset, variant, param, level_key)
        row = one_level(group, refs, label, n_boot, level, master_seed)
        rows.append(finish(row, factor, variant, param, level_key, reference_text(variant, margin_quantile)))
    return rows


def forecaster_rows(base, headline, n_boot, level, master_seed):
    """AA1: each other P3 forecaster against the headline P3 forecaster, per dataset."""
    reference = select_variant(base, headline)
    rows = []
    for dataset in sorted(headline):
        head = headline[dataset]
        for name in sorted(v for v in base.loc[base["dataset"] == dataset, "variant"].unique()
                           if v.startswith(P3_PREFIX) and v != head):
            sel = base[(base["dataset"] == dataset) & (base["variant"] == name)]
            ref = reference[reference["dataset"] == dataset]
            label = "aa:AA1:%s:%s" % (dataset, name)
            rows.append(finish(one_level(sel, ref, label, n_boot, level, master_seed), AA1, "P3", FORECASTER, name,
                               "must:" + head))
    return rows


def layers_rows(q3, q4, m, headroom, h_net, phi, n_boot, level, master_seed):
    """AA11: layers off (Q4) against layers on (Q3, coupled, h_net and phi of the configuration), same policy."""
    off = q4[q4["factor"] == AA11]
    on = p6_rows(q3, m, headroom, h_net, phi)
    rows = []
    for (dataset, variant), group in off.groupby(["dataset", "variant"], sort=True):
        ref = on[(on["dataset"] == dataset) & (on["variant"] == variant)]
        label = "aa:AA11:%s:%s" % (dataset, variant)
        rows.append(finish(one_level(group, ref, label, n_boot, level, master_seed), AA11, variant, LAYERS, "off",
                           "q3:coupled h_net=%s phi=%s" % (h_net, phi)))
    return rows


def ablation_table(must, q1, q3, q4, q6, headline, m, headroom, margin_quantile, h_net, phi, alpha, n_boot, level,
                   master_seed):
    base = default_rows(must, m, headroom)
    ref = reference_rows(q1, must, m, headroom, margin_quantile)
    rows = []
    rows += factor_rows(q1[~is_default_p4a(q1, margin_quantile)], ref, margin_quantile, n_boot, level, master_seed)
    rows += factor_rows(q6, ref, margin_quantile, n_boot, level, master_seed)
    rows += factor_rows(q4[q4["factor"] != AA11], ref, margin_quantile, n_boot, level, master_seed)
    rows += forecaster_rows(base, headline, n_boot, level, master_seed)
    rows += layers_rows(q3, q4, m, headroom, h_net, phi, n_boot, level, master_seed)
    table = pd.DataFrame(rows)
    table["ab_num"] = table["ablation"].str[2:].astype(int)
    table = table.sort_values(["ab_num", "dataset", "variant", "param", "level"], kind="stable").reset_index(drop=True)
    table["n_tests"] = table.groupby("ablation")["p_raw"].transform("size")
    table["p_holm"] = NAN
    for name, idx in table.groupby("ablation").groups.items():
        table.loc[idx, "p_holm"] = holm_adjust(table.loc[idx, "p_raw"].to_numpy())
    table["holm_below_alpha"] = table["p_holm"] < alpha
    rest = [c for c in table.columns if c not in META and c != "ab_num"]
    return table[META + rest]
