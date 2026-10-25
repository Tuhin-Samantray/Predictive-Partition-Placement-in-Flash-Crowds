"""Planted tests for the group 2 figures (A97)."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src import report_figures2 as g2

STYLE = Path(__file__).resolve().parents[1] / "configs" / "report_style.json"
CFG = {"ci_level": 0.95, "n_boot": 9999, "master_seed": 7, "headroom": 1.6, "h5_ratio": 1.25, "h6_tau": 0.6}
POLICIES = ["P0", "P1", "P2", "P3", "P4", "P4a", "P5"]


def rt_table():
    rows = []
    for di, ds in enumerate(["D1", "D2", "D1_D3", "D2_D3"]):
        for pi, p in enumerate(POLICIES):
            for mi, metric in enumerate(["damage", "slo_violation_frac", "peak_backlog_steps", "moves"]):
                mean = 1.0 + di + pi + mi
                rows.append(dict(dataset=ds, policy=p, metric=metric, source="x", n_replicates=60, n_events=60,
                                 mean=mean, median=mean, sd=0.1, ci_low=mean - 0.5, ci_high=mean + 0.5))
    return pd.DataFrame(rows)


def regime(datasets, classes, policy, headrooms):
    rows = []
    for ds in datasets:
        for c in classes:
            for m in [1, 2, 5, 10]:
                for h in headrooms:
                    rows.append(dict(dataset=ds, onset_class=c, m=m, headroom=h, policy=policy, n_replicates=30,
                                     n_events=8, n_events_defined=0 if (c == classes[0] and m == 1) else 5,
                                     share_defined=0.3, median_tax=1.0,
                                     median_R=np.nan if (c == classes[0] and m == 1) else 0.5 * m / 10,
                                     ci_low=0.0, ci_high=1.0, note="few events"))
    return pd.DataFrame(rows)


def c5():
    rows = []
    for u in ["ridge", "lightgbm"]:
        for tg in ["D1", "D2"]:
            for h in [1.0, 2.0, 5.0, 10.0]:
                rows.append(dict(part="forecast_ratio", unit=u, target=tg, source="x", horizon=h, value_in=1,
                                 value_cross=2, ratio=1.5, n_events=np.nan, regret_mean=np.nan, regret_ci_low=np.nan,
                                 regret_ci_high=np.nan, p_wilcoxon_exploratory=np.nan))
    for u in ["P1", "P2", "P3_lightgbm", "P4", "P4a", "P5"]:
        for tg in ["D1", "D2"]:
            rows.append(dict(part="xa2_ratio", unit=u, target=tg, source="x", horizon=np.nan, value_in=1,
                             value_cross=1, ratio=1.02, n_events=60, regret_mean=0.1, regret_ci_low=-0.1,
                             regret_ci_high=0.3, p_wilcoxon_exploratory=0.5))
    return pd.DataFrame(rows)


def c6():
    pols = ["P0", "P1", "P2", "P3", "P4", "P5"]
    return pd.DataFrame(dict(policy=pols, n_events_D1=60, median_damage_D1=1.0, rank_D1=range(1, 7),
                             n_events_D2=60, median_damage_D2=1.0, rank_D2=list(range(6, 0, -1))))


def aa():
    rows = []
    for ds in ["D1", "D2"]:
        for a, par, lv in [("AA3", "hysteresis", ["0.1", "0.2", "0.5"]), ("AA5", "budget", ["1", "2", "8", "unlimited"]),
                           ("AA1", "forecaster", ["P3_naive"]), ("AA10", "kappa", ["0.2", "0.5"])]:
            for v in ["P1", "P3", "P4a"]:
                for k, l in enumerate(lv):
                    rows.append(dict(family="C12:" + a, ablation=a, dataset=ds, variant=v, param=par, level=l,
                                     n_events=60, mean_reference=2.0, mean_policy=3.0, mean_diff=1.0 + k,
                                     ci_low=0.5, ci_high=2.0 + k, direction="up", p_holm=0.01 * k,
                                     holm_below_alpha=bool(k % 2)))
    return pd.DataFrame(rows)


def c4_cells():
    rows = []
    for f in ["naive", "ewma", "seasonal_naive", "ridge", "lightgbm"]:
        for ds in ["D1", "D2"]:
            for c in ["abrupt", "ramped"]:
                rows.append(dict(forecaster=f, dataset=ds, onset_class=c, n_events_r=8, benefit=-1.0 if f == "ridge" else 0.5,
                                 horizon=5, mase_event_class=15.0, n_mase_cells=3))
    return pd.DataFrame(rows)


def write_all(d):
    d.mkdir(parents=True, exist_ok=True)
    rt_table().to_csv(d / "rt_policy_metrics.csv", index=False)
    regime(["D1_D3", "D2_D3"], ["rise1", "rise2", "rise3", "rise4", "rise5"], "P3_naive", [1.6]).to_csv(
        d / "regime_map_q2.csv", index=False)
    regime(["D1", "D2"], ["abrupt", "ramped"], "P3_lightgbm", [1.3, 1.6, 2.0]).to_csv(d / "regime_map_must.csv", index=False)
    c5().to_csv(d / "c5_transfer_detail.csv", index=False)
    c6().to_csv(d / "c6_ranking.csv", index=False)
    pd.DataFrame([dict(n_policies=6, tau=0.33, ci_low=0.07, ci_high=0.73, h6_tau=0.6, met=False,
                       h6_label="inconclusive")]).to_csv(d / "c6_ranking_tau.csv", index=False)
    aa().to_csv(d / "aa_ablations.csv", index=False)
    c4_cells().to_csv(d / "c4_accuracy_benefit_cells.csv", index=False)
    pd.DataFrame([dict(n_cells_total=20, n_cells_used=20, n_cells_without_benefit=0, rho=-0.43, ci_low=-0.79,
                       ci_high=0.09, abs_ci_bound=0.79, h4_rho=0.5, h4_ub=0.7, p_spearman_exploratory=0.059, met=False,
                       h4_label="inconclusive")]).to_csv(d / "c4_accuracy_benefit.csv", index=False)


def test_heat_grid_values_and_blank_cells():
    d = regime(["D1"], ["abrupt", "ramped"], "P3_lightgbm", [1.6])
    grid, count = g2.heat_grid(d, "onset_class", ["abrupt", "ramped"], [1, 2, 5, 10])
    assert np.isnan(grid[0, 0]) and count[0, 0] == 0
    assert grid[1, 2] == 0.5 * 5 / 10 and count[1, 2] == 5


def test_numeric_key_orders_numbers_before_words():
    assert sorted(["unlimited", "8", "1", "2"], key=g2.numeric_key) == ["1", "2", "8", "unlimited"]


def test_ablation_frame_relative_scaling():
    d = g2.ablation_frame(aa(), "D1")
    r = d.iloc[0]
    assert r["rel"] == r["mean_diff"] / 2.0 and r["rel_high"] == r["ci_high"] / 2.0


def test_group2_files_captions_determinism(tmp_path):
    write_all(tmp_path / "t")
    a = g2.make_group2(CFG, tmp_path / "t", tmp_path / "o1", STYLE)
    b = g2.make_group2(CFG, tmp_path / "t", tmp_path / "o2", STYLE)
    assert len(a) == 8 and all(Path(p).stat().st_size > 0 for p in a)
    assert [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in a] == \
           [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in b]
    for p in a:
        assert "tables/" in Path(p).with_suffix(".caption.txt").read_text(encoding="utf-8")


def test_place_labels_no_overlap():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = g2.load_style2(STYLE)
    fig, ax = plt.subplots(figsize=style["size_fa11"])
    marks = [(1.0, 1.0, "R1"), (1.01, 1.0, "R2"), (1.0, 1.01, "P4"), (1.02, 1.02, "P4a"), (1.01, 1.02, "P5")]
    ax.plot([m[0] for m in marks], [m[1] for m in marks], linestyle="none", marker="o")
    texts = g2.place_labels(ax, marks, style)
    renderer = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(renderer) for t in texts]
    area = ax.get_window_extent(renderer)
    for i, a in enumerate(boxes):
        assert a.x0 >= area.x0 and a.x1 <= area.x1 and a.y0 >= area.y0 and a.y1 <= area.y1
        for b in boxes[i + 1:]:
            assert not g2.boxes_overlap(a, b)
    plt.close(fig)
