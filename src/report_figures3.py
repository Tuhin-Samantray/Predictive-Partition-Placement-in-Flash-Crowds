"""Report figures, group 3 (A101): FA13 contention, FA14 conformal logging, FA15 placement and budget.

Same rules as groups 1 and 2: every drawn number is read from a recorded table, no number is typed here,
style constants are in the style files (configs/report_style_g3.json adds the group 3 keys), vector PDF
without creation date, caption file beside each figure, datasets never pooled, legends outside the axes.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src import report_figures as rf
from src import report_figures2 as g2

MODE = "coupled"
FIRST, SECOND = "D1", "D2"
RV1_ROWS = (("damage", "normalised damage (log scale)"),
            ("slo_violation_frac", "SLO violation fraction"),
            ("peak_backlog_steps", "peak backlog (steps)"))
RV4_PANELS = (("pooled coverage", "rv4_fix_pooled_coverage", "rv4_ad_pooled_coverage", False),
              ("interval width", "rv4_fix_width", "rv4_ad_width", True))
RV4_VARIANTS = ("P4", "P4a")
RV3_PLACEMENTS = ("balanced", "skewed")
RV3_POLICIES = ("P1", "P4a")
RV2_POLICIES = ("P0", "P1", "P4a")


def load_style3(path):
    style = g2.load_style2(path)
    extra = Path(path).with_name("report_style_g3.json")
    style.update(json.loads(extra.read_text(encoding="utf-8")))
    return style


def offsets(n, width):
    return [(k - (n - 1) / 2) * width for k in range(n)]


def bounds(mean, lo, hi):
    """Interval ends with a missing end replaced by the mean and never on the wrong side of the mean."""
    lo = mean if pd.isna(lo) else min(lo, mean)
    hi = mean if pd.isna(hi) else max(hi, mean)
    return lo, hi


def draw(ax, x, mean, lo, hi, policy, filled, style):
    lo, hi = bounds(mean, lo, hi)
    colour = style["policy_colors"][policy]
    ax.errorbar([x], [mean], yerr=[[mean - lo], [hi - mean]], fmt=style["policy_markers"][policy], color=colour,
                markerfacecolor=colour if filled else "none", capsize=style["capsize"] / 2,
                elinewidth=style["elinewidth"])


def policy_handles(policies, style):
    handles = [plt.Line2D([], [], linestyle="none", marker=style["policy_markers"][p],
                          color=style["policy_colors"][p]) for p in policies]
    return handles, [style["policy_labels"][p] for p in policies]


def pick(frame, **conds):
    x = frame
    for k, v in conds.items():
        x = x[x[k] == v]
    return x


# ---------------------------------------------------------------- figures

def make_fa13(rt, rv1, cfg, out_dir, style):
    d = rt[rt["mode"] == MODE]
    rec = rv1[(rv1["mode"] == MODE) & (rv1["recovery_defined"].astype(str) == "True")]
    levels, pols, dsets = style["level_order"], style["rv1_policies"], [FIRST, SECOND]
    nrow = len(RV1_ROWS) + 1
    fig, axes = plt.subplots(nrow, len(dsets), figsize=style["size_fa13"], squeeze=False)
    shift = offsets(len(pols), style["bar_total"] / len(pols))
    for j, ds in enumerate(dsets):
        for i, (metric, label) in enumerate(RV1_ROWS):
            ax = axes[i][j]
            for k, p in enumerate(pols):
                for li, lv in enumerate(levels):
                    r = pick(d, dataset=ds, policy=p, level=lv, metric=metric)
                    if not r.empty:
                        r = r.iloc[0]
                        draw(ax, li + shift[k], r["mean"], r["ci_low"], r["ci_high"], p, True, style)
            if metric == rf.METRIC_DAMAGE:
                ax.set_yscale("log")
            ax.set_ylabel(label)
        ax = axes[nrow - 1][j]
        for k, p in enumerate(pols):
            for li, lv in enumerate(levels):
                r = pick(rec, dataset=ds, policy=p, level=lv)
                if not r.empty:
                    r = r.iloc[0]
                    draw(ax, li + shift[k], r["recovery_fraction"], r["recovery_ci_low"], r["recovery_ci_high"], p,
                         True, style)
        ax.set_ylabel("recovery fraction")
        axes[0][j].set_title(style["dataset_titles"][ds])
        for i in range(nrow):
            axes[i][j].set_xticks(range(len(levels)))
            axes[i][j].set_xticklabels([style["level_labels"][lv] for lv in levels])
    handles, labels = policy_handles(pols, style)
    fig.legend(handles, labels, loc="outside lower center", ncol=len(pols))
    counts = rf.event_counts(d[d["metric"] == rf.METRIC_DAMAGE], style)
    cap = ("FA13. Contention ablation (RV1) in the coupled model, per dataset (each panel its own scale): mean per-event "
           "damage (log scale), SLO violation fraction and peak backlog with their %s, and the recovery fraction "
           "of tables/rv1_contention.csv (bottom row; blank where it is undefined, which includes the full model, "
           "the reference). Levels: full model, CPU contention off, NIC contention off, both off. The constant-m "
           "control mode is in the tables and not drawn. Events per dataset: %s. Tables: tables/rt_rv1_metrics.csv "
           "(descriptive, not pre-registered) and tables/rv1_contention.csv." % (rf.ci_text(cfg), counts))
    return rf.save(fig, out_dir, "fa13_contention", cap, style)


def make_fa14(rv4, cfg, out_dir, style):
    dsets = [x for x in style["dataset_order"] if x in set(rv4["dataset"])]
    series = [(v, kind) for v in RV4_VARIANTS for kind in ("fixed", "adaptive")]
    shift = offsets(len(series), style["bar_total"] / len(series))
    fig, axes = plt.subplots(1, len(RV4_PANELS), figsize=style["size_fa14"], squeeze=False)
    for ax, (title, fixed, adaptive, log) in zip(axes[0], RV4_PANELS):
        for di, ds in enumerate(dsets):
            for s, (v, kind) in enumerate(series):
                r = pick(rv4, dataset=ds, variant=v, metric=fixed if kind == "fixed" else adaptive)
                if not r.empty:
                    r = r.iloc[0]
                    draw(ax, di + shift[s], r["mean"], r["ci_low"], r["ci_high"], v, kind == "fixed", style)
        if log:
            ax.set_yscale("log")
        ax.set_xticks(range(len(dsets)))
        ax.set_xticklabels([style["dataset_titles"][x] for x in dsets], rotation=style["tick_rotation"])
        ax.set_ylabel(title)
    handles = [plt.Line2D([], [], linestyle="none", marker=style["policy_markers"][v], color=style["policy_colors"][v],
                          markerfacecolor=style["policy_colors"][v] if kind == "fixed" else "none")
               for v, kind in series]
    fig.legend(handles, ["%s, %s interval" % (v, kind) for v, kind in series], loc="outside lower center",
               ncol=len(series))
    counts = rf.event_counts(rv4[rv4["metric"] == RV4_PANELS[0][1]], style)
    cap = ("FA14. Conformal interval logging (RV4): pooled coverage (left) and interval width (right, log scale) of the "
           "fixed and the adaptive interval, for P4 and P4a, per dataset (widths are not comparable across "
           "datasets). Filled markers: fixed interval; open: adaptive. Mean with its %s. The table holds pooled "
           "values, so there is no time axis; the adaptive level (alpha), the point error and the trigger counts are "
           "in the table and not drawn. Events per dataset: %s. Source: tables/rv4_logging.csv."
           % (rf.ci_text(cfg), counts))
    return rf.save(fig, out_dir, "fa14_conformal_logging", cap, style)


def make_fa15(rv2, rv3, cfg, out_dir, style):
    dsets = [FIRST, SECOND]
    fig, axes = plt.subplots(len(dsets), 2, figsize=style["size_fa15"], squeeze=False)
    for i, ds in enumerate(dsets):
        ax = axes[i][0]
        for k, p in enumerate(RV2_POLICIES):
            r = pick(rv2, dataset=ds, policy=p)
            if not r.empty:
                r = r.iloc[0]
                draw(ax, k, r["mean_diff"], r["ci_low"], r["ci_high"], p, True, style)
        ax.set_xticks(range(len(RV2_POLICIES)))
        ax.set_xticklabels([style["policy_labels"][p] for p in RV2_POLICIES], rotation=style["tick_rotation"])
        ax.set_ylabel("damage difference")
        ax.set_title("%s: skewed minus balanced placement" % ds)
        ax = axes[i][1]
        pos = 0
        names = []
        for p in RV3_POLICIES:
            for placement in RV3_PLACEMENTS:
                r = rv3[(rv3["dataset"] == ds) & (rv3["policy"] == p) & rv3["comparison"].str.endswith(placement)]
                if not r.empty:
                    r = r.iloc[0]
                    draw(ax, pos, r["mean_diff"], r["ci_low"], r["ci_high"], p, placement == RV3_PLACEMENTS[0], style)
                names.append("%s, %s" % (style["policy_labels"][p], placement))
                pos += 1
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=style["tick_rotation"])
        ax.set_ylabel("damage difference")
        ax.set_title("%s: budget 1 minus default" % ds)
        for ax in axes[i]:
            ax.axhline(0, color=style["zero_line_color"], linewidth=style["elinewidth"])
    n = "; ".join("%s %s" % (ds, int(rv2[rv2["dataset"] == ds]["n_events"].iloc[0])) for ds in dsets)
    cap = ("FA15. Placement and migration budget (RV2, RV3), per dataset (each panel its own damage scale). Left: mean "
           "per-event damage under skewed placement minus balanced placement, for R0, R1 and P4a. Right: mean per-event "
           "damage with migration budget 1 minus the default budget, for R1 and P4a, under balanced placement (filled) "
           "and skewed placement (open). Mean difference with its %s; zero line marks no difference. R0, R1 are P0, "
           "P1. Events per dataset: %s. Sources: tables/rv2_placement.csv and tables/rv3_budget.csv. Review figure 10 "
           "asked for the throttled policy; the recorded RV3 rows hold the budget-1 contrast, which is what is drawn."
           % (rf.ci_text(cfg), n))
    return rf.save(fig, out_dir, "fa15_placement_budget", cap, style)


def make_group3(cfg, tables_dir, out_dir, style_path):
    style = load_style3(style_path)
    t = Path(tables_dir)
    rd = lambda n: pd.read_csv(t / n)  # noqa: E731
    return [
        make_fa13(rd("rt_rv1_metrics.csv"), rd("rv1_contention.csv"), cfg, out_dir, style),
        make_fa14(rd("rv4_logging.csv"), cfg, out_dir, style),
        make_fa15(rd("rv2_placement.csv"), rd("rv3_budget.csv"), cfg, out_dir, style),
    ]
