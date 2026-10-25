"""Report figures, group 2 (A97): FA6, FA8, FA9, FA10, FA11, FA12.

Same rules as group 1: every drawn number is read from a recorded table, no number is typed here,
style constants are in configs/report_style.json and configs/report_style_g2.json, vector PDF
without creation date, caption file beside each figure, datasets never pooled.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Bbox
import pandas as pd

from src import report_figures as rf

PART_FORECAST = "forecast_ratio"
PART_XA2 = "xa2_ratio"
FIRST, SECOND = "D1", "D2"
PLAIN = FuncFormatter(lambda v, _: format(v, "g"))


def load_style2(path):
    style = rf.load_style(path)
    extra = Path(path).with_name("report_style_g2.json")
    style.update(json.loads(extra.read_text(encoding="utf-8")))
    return style


def numeric_key(value):
    try:
        return (0, float(value), "")
    except (TypeError, ValueError):
        return (1, 0, str(value))


# ---------------------------------------------------------------- data preparation

def heat_grid(frame, row_col, rows, cols):
    """Median R and the number of events with R defined, one cell per (row, column)."""
    grid = np.full((len(rows), len(cols)), np.nan)
    count = np.zeros((len(rows), len(cols)), dtype=int)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            x = frame[(frame[row_col] == r) & (frame["m"] == c)]
            if not x.empty:
                grid[i, j] = x.iloc[0]["median_R"]
                count[i, j] = int(x.iloc[0]["n_events_defined"])
    return grid, count


def ablation_frame(aa, dataset):
    d = aa[aa["dataset"] == dataset].copy()
    ref = d["mean_reference"]
    d["rel"] = d["mean_diff"] / ref
    d["rel_low"] = d["ci_low"] / ref
    d["rel_high"] = d["ci_high"] / ref
    return d.reset_index(drop=True)


def level_order(rows):
    levels = list(dict.fromkeys(rows["level"].astype(str)))
    return sorted(levels, key=numeric_key)


def damage_moves_frame(metrics, style):
    dm = rf.policy_frame(metrics, rf.METRIC_DAMAGE, style)
    mv = rf.policy_frame(metrics, "moves", style)
    cols = ["dataset", "policy", "mean", "ci_low", "ci_high"]
    out = dm[cols].merge(mv[cols], on=["dataset", "policy"], suffixes=("_damage", "_moves"))
    return out


# ---------------------------------------------------------------- drawing helpers

def heat_panel(ax, grid, count, rows, cols, style):
    lo, hi = style["r_limits"]
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=style["cmap"], vmin=lo, vmax=hi, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = grid[i, j]
            if np.isfinite(v):
                colour = "black" if abs(v) < style["text_switch"] else "white"
                ax.text(j, i, "%s\nn=%d" % (format(v, ".2f"), count[i, j]), ha="center", va="center",
                        color=colour, fontsize=rf.matplotlib.rcParams["xtick.labelsize"])
            else:
                ax.text(j, i, "n=%d" % count[i, j], ha="center", va="center", color=style["ref_line_color"],
                        fontsize=rf.matplotlib.rcParams["xtick.labelsize"])
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([str(c) for c in cols])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.grid(False)
    return im


# ---------------------------------------------------------------- label placement (A98)

LABEL_DIRECTIONS = ((1, 1), (1, -1), (-1, 1), (-1, -1), (0, 1), (0, -1), (1, 0), (-1, 0))
LABEL_SCALES = (1, 2, 3)


def boxes_overlap(a, b):
    return a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1


def place_labels(ax, marks, style):
    """Label each (x, y, text) mark. Candidate offsets are tried in a fixed order until the text box meets no
    placed label, no other marker and stays inside the axes; deterministic, so reruns give identical bytes."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    off = style["label_offset"]
    half = renderer.points_to_pixels(off) / 2
    area = ax.get_window_extent(renderer)
    centres = [ax.transData.transform((x, y)) for x, y, _ in marks]
    marker_boxes = [Bbox.from_bounds(cx - half, cy - half, 2 * half, 2 * half) for cx, cy in centres]
    placed, texts = [], []
    for k, (x, y, label) in enumerate(marks):
        keep = None
        for scale in LABEL_SCALES:
            for dx, dy in LABEL_DIRECTIONS:
                t = ax.annotate(label, (x, y), textcoords="offset points", xytext=(dx * scale * off, dy * scale * off),
                                ha="left" if dx > 0 else ("right" if dx < 0 else "center"),
                                va="bottom" if dy > 0 else ("top" if dy < 0 else "center"))
                bb = t.get_window_extent(renderer)
                inside = bb.x0 >= area.x0 and bb.x1 <= area.x1 and bb.y0 >= area.y0 and bb.y1 <= area.y1
                free = not any(boxes_overlap(bb, o) for o in placed) and \
                    not any(boxes_overlap(bb, m) for i, m in enumerate(marker_boxes) if i != k)
                if inside and free:
                    keep = (t, bb)
                    break
                t.remove()
            if keep:
                break
        if keep is None:
            dx, dy = LABEL_DIRECTIONS[0]
            t = ax.annotate(label, (x, y), textcoords="offset points", xytext=(dx * off, dy * off))
            keep = (t, t.get_window_extent(renderer))
        texts.append(keep[0])
        placed.append(keep[1])
    return texts


# ---------------------------------------------------------------- figures

def make_fa6_q2(q2, cfg, out_dir, style):
    dsets = [d for d in ["D1_D3", "D2_D3"] if d in set(q2["dataset"])]
    rows, cols = style["q2_rows"], style["m_levels"]
    fig, axes = plt.subplots(1, len(dsets), figsize=style["size_fa6_q2"], squeeze=False)
    im = None
    for ax, ds in zip(axes[0], dsets):
        d = q2[q2["dataset"] == ds]
        grid, count = heat_grid(d, "onset_class", rows, cols)
        im = heat_panel(ax, grid, count, rows, cols, style)
        ax.set_title("%s (%s)" % (style["dataset_titles"][ds], d["policy"].iloc[0]))
        ax.set_xlabel("migration duration m (steps)")
        ax.set_ylabel("rise-time bucket (rise1 shortest rise)")
    fig.colorbar(im, ax=list(axes[0]), shrink=1, label="median recovery ratio R")
    n = int(q2["n_events"].iloc[0])
    cap = ("FA6a. Measured lag-tax recovery ratio R across migration duration m (x) and flash-crowd rise-time bucket "
           "(y), headroom %s, injected-event data D3 on D1 and on D2 (forecast policy: %s). Cell text is the median R "
           "over events with R defined and the number n of those events (of %s events per bucket); blank cells have "
           "no event with R defined. Colour is clipped to the range %s to %s, the printed value is not clipped. R is "
           "defined only where the lag tax is at least the registered share of the reactive damage. Every cell rests "
           "on few events (flag in tables/regime_map_q2.csv); intervals are in the table. Datasets are not pooled."
           % (cfg["headroom"], "/".join(sorted(set(q2["policy"]))), n, style["r_limits"][0], style["r_limits"][1]))
    return rf.save(fig, out_dir, "fa6_regime_q2", cap, style)


def make_fa6_must(must, cfg, out_dir, style):
    pol = style["regime_policy"]
    d0 = must[(must["policy"] == pol) & (must["headroom"] == cfg["headroom"])]
    dsets = [d for d in [FIRST, SECOND] if d in set(d0["dataset"])]
    rows, cols = ["abrupt", "ramped"], style["m_levels"]
    fig, axes = plt.subplots(1, len(dsets), figsize=style["size_fa6_must"], squeeze=False)
    im = None
    for ax, ds in zip(axes[0], dsets):
        grid, count = heat_grid(d0[d0["dataset"] == ds], "onset_class", rows, cols)
        im = heat_panel(ax, grid, count, rows, cols, style)
        ax.set_title(style["dataset_titles"][ds])
        ax.set_xlabel("migration duration m (steps)")
        ax.set_ylabel("onset class")
    fig.colorbar(im, ax=list(axes[0]), shrink=1, label="median recovery ratio R")
    cap = ("FA6b. Headline regime map on the real traces: median recovery ratio R of %s by migration duration m and "
           "onset class at headroom %s, per dataset. Cell text is the median R over events with R defined and the "
           "number n of those events; blank cells have no event with R defined. Colour clipped to %s to %s. "
           "Intervals and the share of replicates with R defined are in tables/regime_map_must.csv. Datasets are "
           "not pooled." % (pol, cfg["headroom"], style["r_limits"][0], style["r_limits"][1]))
    return rf.save(fig, out_dir, "fa6_regime_must", cap, style)


def make_fa8(c5, cfg, out_dir, style):
    fr = c5[c5["part"] == PART_FORECAST]
    xr = c5[c5["part"] == PART_XA2]
    fig, axes = plt.subplots(2, 2, figsize=style["size_fa8"], squeeze=False)
    ax = axes[0][0]
    for unit in style["forecast_units"]:
        for tgt in [FIRST, SECOND]:
            r = fr[(fr["unit"] == unit) & (fr["target"] == tgt)].sort_values("horizon")
            ax.plot(range(len(r)), r["ratio"], marker=style["dataset_markers"][tgt],
                    linestyle=style["dataset_linestyles"][tgt], color=style["forecaster_colors"][unit],
                    label="%s, target %s" % (unit, tgt))
    hs = sorted(fr["horizon"].dropna().unique())
    ax.set_xticks(range(len(hs)))
    ax.set_xticklabels([str(int(h)) for h in hs])
    ax.set_xlabel(style["horizon_label"])
    ax.set_ylabel("cross over in-dataset MASE")
    ax.set_title("forecast transfer ratio")
    ax.legend()
    pols = style["xa2_policies"]
    ax = axes[0][1]
    for tgt in [FIRST, SECOND]:
        r = xr[xr["target"] == tgt].set_index("unit").reindex(pols)
        ax.plot(range(len(pols)), r["ratio"], linestyle="none", marker=style["dataset_markers"][tgt],
                color=style["target_colors"][tgt], label="target %s" % tgt)
    ax.set_xticks(range(len(pols)))
    ax.set_xticklabels(pols, rotation=style["tick_rotation"])
    ax.set_ylabel("damage ratio, foreign over own parameters")
    ax.set_title("tuning-parameter transfer ratio")
    ax.legend()
    for a in (axes[0][0], axes[0][1]):
        a.axhline(1, color=style["ref_line_color"], linewidth=style["elinewidth"])
        a.axhline(cfg["h5_ratio"], color=style["zero_line_color"], linewidth=style["elinewidth"])
    for k, tgt in enumerate([FIRST, SECOND]):
        ax = axes[1][k]
        r = xr[xr["target"] == tgt].set_index("unit").reindex(pols)
        m = r["regret_mean"]
        ax.errorbar(range(len(pols)), m, yerr=[m - r["regret_ci_low"], r["regret_ci_high"] - m], fmt="o",
                    color=style["palette"][0], capsize=style["capsize"], elinewidth=style["elinewidth"])
        ax.axhline(0, color=style["zero_line_color"], linewidth=style["elinewidth"])
        ax.set_xticks(range(len(pols)))
        ax.set_xticklabels(pols, rotation=style["tick_rotation"])
        ax.set_ylabel("regret (damage units)")
        ax.set_title("target %s: regret" % tgt)
    cap = ("FA8. Cross-dataset transfer. Top left: forecast transfer ratio (fixed model fitted on the source "
           "training prefix, scored on the target, over the same protocol in-dataset), ridge and lightgbm, by "
           "horizon and target. Top right: tuning-parameter transfer ratio of mean per-event damage, foreign over "
           "own parameters, per policy and target. Lines: 1 (grey) and the registered threshold %s (orange). "
           "Bottom: regret (foreign minus own mean damage) with %s, per target on its own damage scale. "
           "Source: tables/c5_transfer_detail.csv; ratios have no interval in the table."
           % (cfg["h5_ratio"], rf.ci_text(cfg)))
    return rf.save(fig, out_dir, "fa8_transfer", cap, style)


def make_fa9(c6, tau, cfg, out_dir, style):
    fig, ax = plt.subplots(figsize=style["size_fa9"])
    for _, r in c6.iterrows():
        p = r["policy"]
        ax.plot([0, 1], [r["rank_D1"], r["rank_D2"]], marker=style["policy_markers"][p],
                color=style["policy_colors"][p])
        ax.annotate(style["policy_labels"][p], (0, r["rank_D1"]), textcoords="offset points",
                    xytext=(-style["label_offset"], 0), ha="right", va="center")
        ax.annotate(style["policy_labels"][p], (1, r["rank_D2"]), textcoords="offset points",
                    xytext=(style["label_offset"], 0), ha="left", va="center")
    ax.invert_yaxis()
    ax.set_xticks([0, 1])
    ax.set_xticklabels([FIRST, SECOND])
    ax.set_yticks(sorted(set(c6["rank_D1"]) | set(c6["rank_D2"])))
    ax.margins(x=style["x_margin"])
    ax.set_ylabel("rank by median per-event damage (1 = lowest)")
    t = tau.iloc[0]
    ax.set_title("Kendall tau %s (%s to %s)" % (format(t["tau"], ".2f"), format(t["ci_low"], ".2f"),
                                                format(t["ci_high"], ".2f")))
    n1, n2 = int(c6["n_events_D1"].iloc[0]), int(c6["n_events_D2"].iloc[0])
    cap = ("FA9. Policy ranking on D1 and D2 by median per-event damage (%s events on D1, %s on D2), rank 1 is the "
           "lowest damage. R0, R1, R2 are P0, P1, P2. Kendall tau between the two rankings, %s and its percentile "
           "bootstrap interval are in the title (%s policies, threshold %s in tables/c6_ranking_tau.csv). "
           "Source: tables/c6_ranking.csv." % (n1, n2, "tau", int(t["n_policies"]), cfg["h6_tau"]))
    return rf.save(fig, out_dir, "fa9_ranking", cap, style)


def make_fa10(aa, dataset, cfg, out_dir, style):
    d = ablation_frame(aa, dataset)
    abl = sorted(d["ablation"].unique(), key=lambda a: int(a[2:]))
    ncol = 4
    nrow = -(-len(abl) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=style["size_fa10"], squeeze=False)
    handles = {}
    for k, a in enumerate(abl):
        ax = axes[k // ncol][k % ncol]
        da = d[d["ablation"] == a]
        levels = level_order(da)
        variants = list(dict.fromkeys(da["variant"]))
        width = style["bar_total"] / len(variants)
        for vi, v in enumerate(variants):
            dv = da[da["variant"] == v]
            xs, ys, lo, hi, solid = [], [], [], [], []
            for _, r in dv.iterrows():
                xs.append(levels.index(str(r["level"])) + (vi - (len(variants) - 1) / 2) * width)
                ys.append(r["rel"])
                lo.append(r["rel"] - r["rel_low"])
                hi.append(r["rel_high"] - r["rel"])
                solid.append(bool(r["holm_below_alpha"]))
            colour = style["variant_colors"][v]
            for x, y, l, h, s in zip(xs, ys, lo, hi, solid):
                ax.errorbar([x], [y], yerr=[[l], [h]], fmt=style["variant_markers"][v], color=colour,
                            markerfacecolor=colour if s else "none", capsize=style["capsize"] / 2,
                            elinewidth=style["elinewidth"])
            handles[v] = plt.Line2D([], [], linestyle="none", marker=style["variant_markers"][v], color=colour)
        ax.axhline(0, color=style["ref_line_color"], linewidth=style["elinewidth"])
        ax.set_yscale("symlog", linthresh=style["linthresh"])
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(levels, rotation=style["tick_rotation"])
        ax.set_title("%s (%s)" % (a, da["param"].iloc[0]))
    for k in range(len(abl), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.supylabel("effect relative to the reference mean damage")
    fig.legend(list(handles.values()), list(handles.keys()), loc="outside lower center", ncol=len(handles))
    n = int(d["n_events"].iloc[0])
    cap = ("FA10 (%s). Ablation effects on mean per-event damage, level minus reference, as a multiple of the "
           "reference mean (symlog axis; 0 means no change, 1 means +100 percent), per ablation AA1 to AA11 and "
           "policy, with %s over %s events scaled by the reference mean. Filled markers: Holm-adjusted p below the "
           "registered alpha within the ablation; open markers: not. One factor at a time, levels not re-tuned; "
           "interactions untested; AA13, XA6 and FA16 were not completed. Source: tables/aa_ablations.csv "
           "(descriptive, no hypothesis label)." % (dataset, rf.ci_text(cfg), n))
    return rf.save(fig, out_dir, "fa10_ablations_" + dataset, cap, style)


def make_fa11(metrics, cfg, out_dir, style):
    d = damage_moves_frame(metrics, style)
    dsets = rf.datasets_in(d, style)
    fig, axes = plt.subplots(1, len(dsets), figsize=style["size_fa11"], squeeze=False)
    for ax, ds in zip(axes[0], dsets):
        for _, r in d[d["dataset"] == ds].iterrows():
            p = str(r["policy"])
            ax.errorbar([r["mean_moves"]], [r["mean_damage"]],
                        xerr=[[r["mean_moves"] - r["ci_low_moves"]], [r["ci_high_moves"] - r["mean_moves"]]],
                        yerr=[[r["mean_damage"] - r["ci_low_damage"]], [r["ci_high_damage"] - r["mean_damage"]]],
                        fmt=style["policy_markers"][p], color=style["policy_colors"][p],
                        capsize=style["capsize"] / 2, elinewidth=style["elinewidth"])
        ax.set_yscale("log")
        ax.set_title(style["dataset_titles"][ds])
        ax.set_xlabel("migrations per window (mean)")
        ax.set_ylabel("normalised damage (log scale)")
    handles = [plt.Line2D([], [], linestyle="none", marker=style["policy_markers"][p], color=style["policy_colors"][p])
               for p in style["policy_order"]]
    fig.legend(handles, [style["policy_labels"][p] for p in style["policy_order"]], loc="outside lower center",
               ncol=4)
    cap = ("FA11. Damage against migration count per policy, per dataset (each panel its own damage scale): mean "
           "per-event value with its %s on both axes. R0, R1, R2 are P0, P1, P2; R0 never migrates. Events per "
           "dataset: %s. Default m and headroom; tables/rt_policy_metrics.csv (descriptive, not pre-registered)."
           % (rf.ci_text(cfg), rf.event_counts(rf.policy_frame(metrics, rf.METRIC_DAMAGE, style), style)))
    return rf.save(fig, out_dir, "fa11_damage_moves", cap, style)


def make_fa12(cells, summary, cfg, out_dir, style):
    width, height = style["size_fa12"]
    fig, ax = plt.subplots(figsize=(width + width / 3, height))
    for _, r in cells.iterrows():
        colour = style["forecaster_colors"][r["forecaster"]]
        ax.plot([r["mase_event_class"]], [r["benefit"]], linestyle="none",
                marker=style["dataset_markers"][r["dataset"]], color=colour,
                markerfacecolor=colour if r["onset_class"] == "abrupt" else "none")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=style["linthresh"], linscale=2)
    ax.xaxis.set_major_formatter(PLAIN)
    ax.xaxis.set_minor_formatter(PLAIN)
    ax.yaxis.set_major_formatter(PLAIN)
    ax.axhline(0, color=style["ref_line_color"], linewidth=style["elinewidth"])
    ax.set_xlabel("MASE inside events of the class (horizon %s)" % int(cells["horizon"].iloc[0]))
    ax.set_ylabel("median recovery ratio R (benefit)")
    grey = style["ref_line_color"]
    handles = [plt.Line2D([], [], linestyle="none", marker="o", color=style["forecaster_colors"][f])
               for f in style["forecasters"]]
    labels = list(style["forecasters"])
    for ds in (FIRST, SECOND):
        handles.append(plt.Line2D([], [], linestyle="none", marker=style["dataset_markers"][ds], color=grey))
        labels.append(ds)
    handles.append(plt.Line2D([], [], linestyle="none", marker="o", color=grey, markerfacecolor=grey))
    labels.append("abrupt (filled)")
    handles.append(plt.Line2D([], [], linestyle="none", marker="o", color=grey, markerfacecolor="none"))
    labels.append("ramped (open)")
    fig.legend(handles, labels, loc="outside right upper")
    s = summary.iloc[0]
    ax.set_title("Spearman rho %s (%s to %s), %s cells" % (format(s["rho"], ".2f"), format(s["ci_low"], ".2f"),
                                                         format(s["ci_high"], ".2f"), int(s["n_cells_used"])))
    cap = ("FA12. Forecast accuracy against control benefit: one point per cell (forecaster, dataset, onset class), "
           "x is the MASE inside events of that class at the horizon nearest m plus d, y is the cell median "
           "recovery ratio R. Circles: D1, squares: D2; filled: abrupt, open: ramped. Spearman rho with its %s "
           "over cells is in the title (exploratory p %s); cell benefits rest on the events with R defined (8 to "
           "19 per cell, tables/c4_accuracy_benefit_cells.csv). Symlog y axis."
           % (rf.ci_text(cfg), format(s["p_spearman_exploratory"], ".3g")))
    return rf.save(fig, out_dir, "fa12_mase_benefit", cap, style)


def make_group2(cfg, tables_dir, out_dir, style_path):
    style = load_style2(style_path)
    t = Path(tables_dir)
    rd = lambda n: pd.read_csv(t / n)  # noqa: E731
    out = [
        make_fa6_q2(rd("regime_map_q2.csv"), cfg, out_dir, style),
        make_fa6_must(rd("regime_map_must.csv"), cfg, out_dir, style),
        make_fa8(rd("c5_transfer_detail.csv"), cfg, out_dir, style),
        make_fa9(rd("c6_ranking.csv"), rd("c6_ranking_tau.csv"), cfg, out_dir, style),
    ]
    aa = rd("aa_ablations.csv")
    out.append(make_fa10(aa, FIRST, cfg, out_dir, style))
    out.append(make_fa10(aa, SECOND, cfg, out_dir, style))
    out.append(make_fa11(rd("rt_policy_metrics.csv"), cfg, out_dir, style))
    out.append(make_fa12(rd("c4_accuracy_benefit_cells.csv"), rd("c4_accuracy_benefit.csv"), cfg, out_dir, style))
    return out
