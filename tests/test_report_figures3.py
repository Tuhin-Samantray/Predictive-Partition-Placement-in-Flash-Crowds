"""Planted tests for the group 3 figures (A101)."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src import report_figures3 as g3

STYLE = Path(__file__).resolve().parents[1] / "configs" / "report_style.json"
CFG = {"ci_level": 0.95, "n_boot": 9999, "master_seed": 7}
LEVELS = ["full", "cpu_off", "nic_off", "both_off"]
POLICIES = ["P1", "P3", "P4a", "P5"]


def rt_rv1():
    rows = []
    for ds in ["D1", "D2"]:
        for p in POLICIES:
            for li, lv in enumerate(LEVELS):
                for mode in ["coupled", "const_m"]:
                    for mi, metric in enumerate(["damage", "moves", "peak_backlog_steps", "slo_violation_frac"]):
                        mean = 1.0 + li + mi
                        rows.append(dict(dataset=ds, policy=p, level=lv, mode=mode, metric=metric, n_replicates=60,
                                         n_events=60, mean=mean, median=mean, sd=0.1, ci_low=mean - 0.5,
                                         ci_high=mean + 0.5))
    return pd.DataFrame(rows)


def rv1():
    rows = []
    for ds in ["D1", "D2"]:
        for p in POLICIES:
            for lv in LEVELS:
                for mode in ["coupled", "const_m"]:
                    ok = lv != "full"
                    rows.append(dict(dataset=ds, policy=p, level=lv, mode=mode, n_events=60,
                                     recovery_fraction=0.4 if ok else np.nan, recovery_ci_low=0.1 if ok else np.nan,
                                     recovery_ci_high=0.8 if ok else np.nan, recovery_defined=ok))
    return pd.DataFrame(rows)


def rv2():
    rows = [dict(factor="RV2", comparison="skewed minus balanced", policy=p, dataset=ds, n_events=60,
                 mean_diff=2.0 + k, ci_low=1.0, ci_high=3.0 + k)
            for ds in ["D1", "D2"] for k, p in enumerate(["P0", "P1", "P4a"])]
    return pd.DataFrame(rows)


def rv3():
    rows = [dict(factor="RV3", comparison="budget 1 minus default, %s" % pl, policy=p, dataset=ds, n_events=60,
                 mean_diff=1.0, ci_low=np.nan if pl == "skewed" else 0.5, ci_high=2.0)
            for ds in ["D1", "D2"] for p in ["P1", "P4a"] for pl in ["balanced", "skewed"]]
    return pd.DataFrame(rows)


def rv4():
    rows = []
    for ds in ["D1", "D1_D3", "D2", "D2_D3"]:
        for v in ["P4", "P4a"]:
            for m, val in [("rv4_fix_pooled_coverage", 0.9), ("rv4_ad_pooled_coverage", 0.8), ("rv4_fix_width", 3.0),
                           ("rv4_ad_width", 2.0), ("rv4_point_mae", 5.0)]:
                rows.append(dict(dataset=ds, variant=v, metric=m, n_events=60, n_null_rows=0, mean=val,
                                 ci_low=val * 0.9, ci_high=val * 1.1))
    return pd.DataFrame(rows)


def write_all(d):
    d.mkdir(parents=True, exist_ok=True)
    rt_rv1().to_csv(d / "rt_rv1_metrics.csv", index=False)
    rv1().to_csv(d / "rv1_contention.csv", index=False)
    rv2().to_csv(d / "rv2_placement.csv", index=False)
    rv3().to_csv(d / "rv3_budget.csv", index=False)
    rv4().to_csv(d / "rv4_logging.csv", index=False)


def test_bounds_fill_and_clip():
    assert g3.bounds(2.0, np.nan, np.nan) == (2.0, 2.0)
    assert g3.bounds(2.0, 2.5, 1.5) == (2.0, 2.0)
    assert g3.bounds(2.0, 1.0, 3.0) == (1.0, 3.0)


def test_offsets_are_centred():
    o = g3.offsets(4, 0.2)
    assert abs(sum(o)) < 10 ** -9 and len(o) == 4


def test_group3_files_captions_determinism(tmp_path):
    write_all(tmp_path / "t")
    a = g3.make_group3(CFG, tmp_path / "t", tmp_path / "o1", STYLE)
    b = g3.make_group3(CFG, tmp_path / "t", tmp_path / "o2", STYLE)
    assert len(a) == 3 and all(Path(p).stat().st_size > 0 for p in a)
    assert [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in a] == \
           [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in b]
    for p in a:
        assert "tables/" in Path(p).with_suffix(".caption.txt").read_text(encoding="utf-8")
