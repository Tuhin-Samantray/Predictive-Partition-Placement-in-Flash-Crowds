"""Report figures, group 1 (A94): FA3, FA4, FA5, FA7.

Every drawn number is read from a recorded table in tables/ (no number is typed here).
Style constants live in configs/report_style.json. Output is vector PDF without a creation
date, so a rerun gives identical bytes. A caption text file sits beside every figure.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

EVAL = "eval"
ALL_CLASS = "all"
METRIC_DAMAGE = "damage"
METRIC_VIOLATION = "slo_violation_frac"
METRIC_BACKLOG = "peak_backlog_steps"
TEST_TAX = "wilcoxon_tax"
TEST_RHO = "spearman_tax_m"
HEADLINE_FORECASTER = "naive"


def load_style(path):
    style = json.loads(Path(path).read_text(encoding="utf-8"))
    matplotlib.rcParams.update(style["rcparams"])
    return style


def ci_text(cfg):
    return "%s BCa interval over events (%s resamples, master seed %s)" % (
        format(cfg["ci_level"], ".0%"), cfg["n_boot"], cfg["master_seed"])


def save(fig, out_dir, stem, caption, style):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (stem + ".pdf")
    fig.savefig(path, format="pdf", metadata=style["pdf_metadata"])
    plt.close(fig)
    (out_dir / (stem + ".caption.txt")).write_text(caption + "\n", encoding="utf-8", newline="\n")
    return path


# ---------------------------------------------------------------- data preparation

def fa3_frame(acc):
    """Evaluation split only, one row per dataset, forecaster and horizon."""
    d = acc[acc["split"] == EVAL].copy()
    return d.sort_values(["dataset", "forecaster", "horizon"]).reset_index(drop=True)


def fa4_frame(ew):
    d = ew[ew["split"] == EVAL].copy()
    return d.reset_index(drop=True)


def fa5_frame(c1):
    return c1.copy().reset_index(drop=True)


def policy_frame(metrics, metric, style):
    """Rows of one metric in the registered policy and dataset order."""
    d = metrics[metrics["metric"] == metric].copy()
    d["policy"] = pd.Categorical(d["policy"], style["policy_order"], ordered=True)
    d["dataset"] = pd.Categorical(d["dataset"], style["dataset_order"], ordered=True)
    d = d.sort_values(["dataset", "policy"]).reset_index(drop=True)
    d["ci_low"] = d["ci_low"].fillna(d["mean"])
    d["ci_high"] = d["ci_high"].fillna(d["mean"])
    return d


# ---------------------------------------------------------------- drawing helpers

def dot_panel(ax, frame, style, log=False):
    policies = style["policy_order"]
    for i, p in enumerate(policies):
        r = frame[frame["policy"] == p]
        if r.empty:
            continue
        r = r.iloc[0]
        ax.errorbar([i], [r["mean"]], yerr=[[r["mean"] - r["ci_low"]], [r["ci_high"] - r["mean"]]],
                    fmt=style["policy_markers"][p], color=style["policy_colors"][p],
                    capsize=style["capsize"], elinewidth=style["elinewidth"])
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels([style["policy_labels"][p] for p in policies], rotation=style["tick_rotation"])
    if log:
        ax.set_yscale("log")


def datasets_in(frame, style):
    present = set(frame["dataset"].astype(str))
    return [d for d in style["dataset_order"] if d in present]


def event_counts(frame, style):
    parts = []
    for d in datasets_in(frame, style):
        n = frame[frame["dataset"] == d]["n_events"]
        parts.append("%s %s" % (d, int(n.iloc[0])))
    return "; ".join(parts)


# ---------------------------------------------------------------- figures

def make_fa3(acc, cfg, out_dir, style):
    d = fa3_frame(acc)
    dsets = [x for x in style["dataset_order"] if x in set(d["dataset"])]
    fig, axes = plt.subplots(len(dsets), 2, figsize=style["size_fa3"], squeeze=False)
    cols = [("mase_all", "MASE, all cells"), ("mase_event", "MASE, inside events")]
    horizons = sorted(d["horizon"].unique())
    for i, ds in enumerate(dsets):
        for j, (col, title) in enumerate(cols):
            ax = axes[i][j]
            for f in style["forecasters"]:
                r = d[(d["dataset"] == ds) & (d["forecaster"] == f)].sort_values("horizon")
                ax.plot(range(len(r)), r[col], marker=style["forecaster_markers"][f],
                        color=style["forecaster_colors"][f], label=f)
            ax.set_xticks(range(len(horizons)))
            ax.set_xticklabels([str(h) for h in horizons])
            ax.set_title("%s: %s" % (ds, title))
            ax.set_ylabel("MASE")
            if i == len(dsets) - 1:
                ax.set_xlabel(style["horizon_label"])
    axes[0][0].legend(ncol=1)
    cap = ("FA3. Forecast accuracy on the evaluation split, MASE over all cells and inside events, by forecaster and "
           "horizon, per dataset (D1, D2). Values are pooled point estimates from tables/forecast_accuracy.csv; no "
           "interval is drawn because the table holds none. MASE is scaled by the training-prefix naive error, so 1 "
           "equals that scale. Source rows are rolling-refit forecasts; the stride-4 Chronos subset is not drawn "
           "(see tables/forecast_accuracy_chronos.csv). Seed: forecasts are deterministic given the stored tables "
           "(master seed %s)." % cfg["master_seed"])
    return save(fig, out_dir, "fa3_forecast_accuracy", cap, style)


def make_fa4(ew, cfg, out_dir, style):
    d = fa4_frame(ew)
    dsets = [x for x in style["dataset_order"] if x in set(d["dataset"])]
    metrics = [("recall", "recall"), ("lead_median", "median lead (steps)")]
    fig, axes = plt.subplots(len(metrics), len(dsets), figsize=style["size_fa4"], squeeze=False)
    classes = style["classes"]
    width = style["bar_total"] / len(classes)
    for i, (col, label) in enumerate(metrics):
        for j, ds in enumerate(dsets):
            ax = axes[i][j]
            for k, c in enumerate(classes):
                offs = (k - (len(classes) - 1) / 2) * width
                xs, ys = [], []
                for n, f in enumerate(style["forecasters"]):
                    r = d[(d["dataset"] == ds) & (d["forecaster"] == f) & (d["onset_class"] == c)]
                    if r.empty or pd.isna(r.iloc[0][col]):
                        continue
                    xs.append(n + offs)
                    ys.append(r.iloc[0][col])
                ax.bar(xs, ys, width=width, color=style["class_colors"][c], label=c)
            ax.set_xticks(range(len(style["forecasters"])))
            ax.set_xticklabels(style["forecasters"], rotation=style["tick_rotation"])
            ax.set_title("%s: %s" % (ds, label))
            ax.set_ylabel(label)
    axes[0][0].legend(title="onset class")
    ref = d[(d["forecaster"] == HEADLINE_FORECASTER)]
    counts = "; ".join("%s %s %s" % (r["dataset"], r["onset_class"], int(r["n_events"]))
                       for _, r in ref.iterrows())
    cap = ("FA4. Early warning on the evaluation split: recall and median lead time (steps) of each forecaster by "
           "onset class, per dataset. Pooled point values from tables/forecast_accuracy_early_warning.csv, no "
           "interval in the table. Events per class (naive row, after exclusions; other forecasters differ by the "
           "excluded events listed in the table): %s. A missing bar means the value is undefined (no alarm raised). "
           "Precision is in the table and is not drawn." % counts)
    return save(fig, out_dir, "fa4_early_warning", cap, style)


def make_fa5(c1, cfg, out_dir, style):
    d = fa5_frame(c1)
    fig, axes = plt.subplots(1, 3, figsize=style["size_fa5"], squeeze=False)
    axes = axes[0]
    tax = d[d["test"] == TEST_TAX]
    for ax, ds in zip(axes[:2], ["D1", "D2"]):
        r = tax[tax["dataset"] == ds].iloc[0]
        ax.errorbar([0], [r["estimate"]], yerr=[[r["estimate"] - r["ci_low"]], [r["ci_high"] - r["estimate"]]],
                    fmt="o", color=style["tax_color"], capsize=style["capsize"], elinewidth=style["elinewidth"])
        ax.axhline(0, color=style["zero_line_color"], linewidth=style["elinewidth"])
        ax.set_xticks([0])
        ax.set_xticklabels([ds])
        ax.set_ylabel("mean per-event lag tax (damage units)")
        ax.set_title("%s\nHolm p %s, %s events" % (ds, format(r["p_holm"], ".3g"),
                                                       int(r["n_events"])))
    rho = d[d["test"] == TEST_RHO]
    ax = axes[2]
    for i, ds in enumerate(["D1", "D2"]):
        r = rho[rho["dataset"] == ds].iloc[0]
        ax.errorbar([i], [r["estimate"]], yerr=[[r["estimate"] - r["ci_low"]], [r["ci_high"] - r["estimate"]]],
                    fmt="o", color=style["rho_color"], capsize=style["capsize"], elinewidth=style["elinewidth"])
    ax.axhline(0, color=style["zero_line_color"], linewidth=style["elinewidth"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["D1", "D2"])
    ax.set_xlim(-1, 2)
    ax.set_ylabel("Spearman rho, tax against m")
    ax.set_title("growth of the tax with m")
    n = int(d["n_events"].iloc[0])
    cap = ("FA5. Lag tax (reactive minus oracle damage) per event at the default m and headroom: mean with its %s "
           "(left, per dataset, own axis because D1 and D2 damage scales differ), and Spearman rho of tax against m "
           "(right). Zero line marks no tax. %s events per dataset; labels and Holm-adjusted p values are those of "
           "tables/c1_lag_tax.csv (family F1, 4 p values). Hypothesis labels are in the ledger." % (ci_text(cfg), n))
    return save(fig, out_dir, "fa5_lag_tax", cap, style)


def make_fa7_damage(metrics, cfg, out_dir, style):
    d = policy_frame(metrics, METRIC_DAMAGE, style)
    dsets = datasets_in(d, style)
    fig, axes = plt.subplots(1, len(dsets), figsize=style["size_fa7_damage"], squeeze=False)
    for ax, ds in zip(axes[0], dsets):
        dot_panel(ax, d[d["dataset"] == ds], style, log=True)
        ax.set_title(style["dataset_titles"][ds])
        ax.set_ylabel("normalised damage (log scale)")
    cap = ("FA7a. Overall performance of the seven placement policies: mean per-event normalised damage with its %s, "
           "per dataset (log y axis, each panel its own scale; datasets are not pooled because D1 damage is about "
           "20 and D2 about 0.02). R0, R1 and R2 are P0, P1 and P2. Events per dataset: %s. Default m and "
           "headroom, 30 replicates per event, tables/rt_policy_metrics.csv (descriptive, not pre-registered)."
           % (ci_text(cfg), event_counts(d, style)))
    return save(fig, out_dir, "fa7_damage", cap, style)


def make_fa7_second(metrics, cfg, out_dir, style):
    v = policy_frame(metrics, METRIC_VIOLATION, style)
    b = policy_frame(metrics, METRIC_BACKLOG, style)
    dsets = datasets_in(v, style)
    fig, axes = plt.subplots(2, len(dsets), figsize=style["size_fa7_second"], squeeze=False)
    for j, ds in enumerate(dsets):
        dot_panel(axes[0][j], v[v["dataset"] == ds], style)
        axes[0][j].set_title(style["dataset_titles"][ds])
        axes[0][j].set_ylabel("SLO violation fraction")
        dot_panel(axes[1][j], b[b["dataset"] == ds], style)
        axes[1][j].set_ylabel("peak backlog (steps)")
    cap = ("FA7b. SLO violation fraction (top) and peak backlog in steps (bottom) for the seven policies, per "
           "dataset: mean per-event value with its %s. R0, R1, R2 are P0, P1, P2. Events per dataset: %s. Default "
           "m and headroom; tables/rt_policy_metrics.csv (descriptive, not pre-registered)."
           % (ci_text(cfg), event_counts(v, style)))
    return save(fig, out_dir, "fa7_violation_backlog", cap, style)


def make_group1(cfg, tables_dir, out_dir, style_path):
    style = load_style(style_path)
    t = Path(tables_dir)
    acc = pd.read_csv(t / "forecast_accuracy.csv")
    ew = pd.read_csv(t / "forecast_accuracy_early_warning.csv")
    c1 = pd.read_csv(t / "c1_lag_tax.csv")
    rt = pd.read_csv(t / "rt_policy_metrics.csv")
    return [
        make_fa3(acc, cfg, out_dir, style),
        make_fa4(ew, cfg, out_dir, style),
        make_fa5(c1, cfg, out_dir, style),
        make_fa7_damage(rt, cfg, out_dir, style),
        make_fa7_second(rt, cfg, out_dir, style),
    ]
