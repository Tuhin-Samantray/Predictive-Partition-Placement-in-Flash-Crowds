"""Planted tests for the group 1 figures (A94): data preparation, files, captions, determinism."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import report_figures as rf

STYLE = Path(__file__).resolve().parents[1] / "configs" / "report_style.json"
CFG = {"ci_level": 0.95, "n_boot": 9999, "master_seed": 7}
POLICIES = ["P0", "P1", "P2", "P3", "P4", "P4a", "P5"]
DATASETS = ["D1", "D2", "D1_D3", "D2_D3"]


def rt_table():
    rows = []
    for di, ds in enumerate(DATASETS):
        for pi, p in enumerate(POLICIES):
            for mi, metric in enumerate([rf.METRIC_DAMAGE, rf.METRIC_VIOLATION, rf.METRIC_BACKLOG, "moves"]):
                mean = 1.0 + di + pi + mi
                rows.append(dict(dataset=ds, policy=p, metric=metric, source="x", n_replicates=60, n_events=60,
                                 mean=mean, median=mean, sd=0.1, ci_low=mean - 0.5, ci_high=mean + 0.5))
    return pd.DataFrame(rows)


def acc_table():
    rows = []
    for ds in ["D1", "D2"]:
        for f in ["naive", "ewma", "seasonal_naive", "ridge", "lightgbm"]:
            for sp in ["eval", "tune"]:
                for h in [1, 2, 5, 10]:
                    rows.append(dict(dataset=ds, forecaster=f, split=sp, horizon=h, mase_all=1.0 + h / 10,
                                     mase_event=10.0 + h, smape_all=1, smape_event=1, n_cells_all=1,
                                     n_cells_event=1, n_zero_scale=0))
    return pd.DataFrame(rows)


def ew_table():
    rows = []
    for ds in ["D1", "D2"]:
        for f in ["naive", "ewma", "seasonal_naive", "ridge", "lightgbm"]:
            for sp in ["eval", "tune"]:
                for c in ["abrupt", "ramped", "all"]:
                    rows.append(dict(dataset=ds, forecaster=f, split=sp, onset_class=c, n_events=5,
                                     n_excluded_events=0, recall=0.2, lead_median=np.nan if f == "ridge" else 3.0,
                                     lead_mean=3.0, n_alarm_cells=1, precision=0.1))
    return pd.DataFrame(rows)


def c1_table():
    rows = []
    for ds in ["D1", "D2"]:
        for t in ["wilcoxon_tax", "spearman_tax_m"]:
            rows.append(dict(dataset=ds, test=t, n_events=60, n_nonzero=60, estimate=0.1, ci_low=-0.2,
                             ci_high=0.4, p_raw=0.1, p_holm=0.2, h1_label="inconclusive"))
    return pd.DataFrame(rows)


def write_tables(d):
    d.mkdir(parents=True, exist_ok=True)
    rt_table().to_csv(d / "rt_policy_metrics.csv", index=False)
    acc_table().to_csv(d / "forecast_accuracy.csv", index=False)
    ew_table().to_csv(d / "forecast_accuracy_early_warning.csv", index=False)
    c1_table().to_csv(d / "c1_lag_tax.csv", index=False)


def digest(paths):
    return [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths]


def test_policy_frame_order_and_values():
    style = rf.load_style(STYLE)
    d = rf.policy_frame(rt_table(), rf.METRIC_DAMAGE, style)
    assert list(d["policy"].astype(str).iloc[:7]) == style["policy_order"]
    assert list(d["dataset"].astype(str).unique()) == style["dataset_order"]
    r = d[(d["dataset"] == "D2") & (d["policy"] == "P4a")].iloc[0]
    assert r["mean"] == 1.0 + 1 + 5 + 0 and r["ci_low"] == r["mean"] - 0.5


def test_policy_frame_fills_missing_interval():
    style = rf.load_style(STYLE)
    t = rt_table()
    t.loc[0, "ci_low"] = np.nan
    d = rf.policy_frame(t, rf.METRIC_DAMAGE, style)
    assert not d["ci_low"].isna().any()


def test_fa3_uses_eval_only():
    d = rf.fa3_frame(acc_table())
    assert set(d["split"]) == {"eval"} and len(d) == 2 * 5 * 4


def test_group1_files_captions_and_determinism(tmp_path):
    write_tables(tmp_path / "t")
    out1, out2 = tmp_path / "o1", tmp_path / "o2"
    a = rf.make_group1(CFG, tmp_path / "t", out1, STYLE)
    b = rf.make_group1(CFG, tmp_path / "t", out2, STYLE)
    assert len(a) == 5 and all(Path(p).stat().st_size > 0 for p in a)
    assert digest(a) == digest(b)
    for p in a:
        cap = Path(p).with_suffix(".caption.txt").read_text(encoding="utf-8")
        assert "tables/" in cap
    cap = (out1 / "fa7_damage.caption.txt").read_text(encoding="utf-8")
    assert "95% BCa" in cap and "9999" in cap and "seed 7" in cap and "D1 60" in cap
    assert b"/CreationDate" not in Path(a[0]).read_bytes()


def test_style_file_is_valid_json():
    s = json.loads(STYLE.read_text(encoding="utf-8"))
    assert set(s["policy_order"]) == set(s["policy_labels"]) == set(s["policy_colors"]) == set(s["policy_markers"])
    assert len(s["palette"]) == 8
