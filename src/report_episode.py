"""Episode figures FA2 and review Figure 3 (A104).

One replicate per dataset is regenerated with the frozen code and configs. The event is chosen by a rule fixed
before any trajectory was drawn (A104, which replaces the event rule of A88 item 5): among the Must rows of the
default cell (migration_steps_m0, headroom, class abrupt, policy P1) take the sampled events, choose the one with the
median value of the events-file column peak (the lower middle one when the count is even, ties by event id in sorted
order) and use the replicate index it has in those rows. A figure is drawn only if the regenerated damage and the
number of moves of every drawn policy equal the frozen row. No number is typed here: the default cell comes from the
config, sizes and colours from configs/report_style_g4.json.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import Patch

from src.config import step_seconds
from src.metrics import compute_metrics
from src.policies import build_policy
from src.replicates import forecast_window, load_dataset, load_forecast, make_replicate, run_replicate
from src import report_figures as rf
from src.report_figures3 import load_style3, policy_handles
from src.simulator import simulate

DATASETS = ("D1", "D2")
CLASS = "abrupt"
SELECT_VARIANT = "P1"
CHECK_NAME = "fa2_episode_check.csv"
PANELS = ("total load per step", "worst-node wait (steps)", "migrations started (cumulative)")


def load_style4(path):
    style = load_style3(path)
    extra = Path(path).with_name("report_style_g4.json")
    style.update(json.loads(extra.read_text(encoding="utf-8")))
    return style


def horizon_index(cfg):
    """Index of the forecast horizon equal to the default migration duration (ValueError if it is not a horizon)."""
    return list(cfg["horizons"]).index(cfg["migration_steps_m0"])


def forecast_total(window, index):
    """Point forecast of the total load at one horizon: table row i holds the forecast made at local step i."""
    return np.asarray(window[:, index, :], dtype=np.float64).sum(axis=1)


def read_rows(folder, event_id=None):
    """Rows of every grid_*.jsonl directly in folder; with event_id only the lines that mention it."""
    rows = []
    for path in sorted(Path(folder).glob("grid_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            if event_id is not None and ('"%s"' % event_id) not in line:
                continue
            rows.append(json.loads(line))
    return rows


def select_episode(rows, events, cfg, dataset):
    keep = [r for r in rows
            if r["dataset"] == dataset and r["onset_class"] == CLASS and r["variant"] == SELECT_VARIANT
            and r["m"] == cfg["migration_steps_m0"] and r["headroom"] == cfg["headroom"]]
    ids = [r["event_id"] for r in keep]
    if not keep or len(set(ids)) != len(ids):
        raise ValueError("%s: need exactly one %s row per event in the default cell, found %d rows for %d events"
                         % (dataset, SELECT_VARIANT, len(keep), len(set(ids))))
    peak = events.set_index("event_id")["peak"]
    ranked = sorted(keep, key=lambda r: (float(peak[r["event_id"]]), r["event_id"]))
    pick = ranked[(len(ranked) - 1) // 2]
    return {"event_id": pick["event_id"], "replicate": int(pick["replicate"]),
            "peak": float(peak[pick["event_id"]]), "n_events": len(ranked)}


def episode_specs(cfg, tuned, ext, dataset, policies):
    """The drawn policies with the parameters and the frozen-row filter of the grid run that produced them.
    P1, P3 (forecaster of P4) and P5 are Must-grid rows; P4a is not in the Must grid, its default row is the
    Q1 AA6 row at the default margin quantile."""
    t = tuned["tuned"][dataset]
    fc = t["P4"]["forecaster"]
    p4a = dict(ext["tuned_ext"][dataset]["P4a"])
    specs = {
        "P1": {"policy": "P1", "name": "P1", "variant": "P1", "params": dict(t["P1"]), "forecaster": None,
               "where": {}},
        "P3": {"policy": "P3", "name": "P3", "variant": "P3_" + fc, "params": dict(t["P3_" + fc]),
               "forecaster": fc, "where": {}},
        "P4a": {"policy": "P4a", "name": "P4a", "variant": "P4a", "params": p4a, "forecaster": p4a["forecaster"],
                "where": {"factor": "AA6", "level": cfg["margin_quantile"]}},
        "P5": {"policy": "P5", "name": "P5", "variant": "P5", "params": dict(t["P5"]), "forecaster": None,
               "where": {}},
    }
    return [specs[p] for p in policies]


def frozen_row(rows, dataset, event_id, replicate, cfg, spec):
    hit = [r for r in rows
           if r["dataset"] == dataset and r["event_id"] == event_id and r["replicate"] == replicate
           and r["onset_class"] == CLASS and r["m"] == cfg["migration_steps_m0"]
           and r["headroom"] == cfg["headroom"] and r["variant"] == spec["variant"]
           and all(r.get(k) == v for k, v in spec["where"].items())]
    if len(hit) != 1:
        raise ValueError("%s %s replicate %d %s: %d frozen rows, need exactly 1"
                         % (dataset, event_id, replicate, spec["variant"], len(hit)))
    return hit[0]


def compare_row(frozen, metrics):
    return bool(frozen["damage"] == metrics["damage"] and frozen["moves"] == metrics["moves"])


def replay(cfg, rep, spec, window):
    """The path of run_replicate with the defaults of the grid, keeping the trajectory (run_replicate returns totals
    only). The metrics must equal run_replicate's own output, otherwise this replay has drifted from the library."""
    m, d, k = cfg["migration_steps_m0"], cfg["monitor_delay_d"], cfg["decision_interval_k"]
    ctx = {"m": m, "d": d, "horizons": cfg["horizons"], "load": rep.load, "margin_quantile": cfg["margin_quantile"],
           "residual_window": cfg["residual_window"], "gamma": cfg["gamma"], "a_min": cfg["a_min"],
           "a_max": cfg["a_max"]}
    policy = build_policy(spec["name"], spec["params"], rep.mu, ctx)
    res = simulate(rep.load, rep.node0, rep.mu, m=m, kappa=cfg["migration_overhead_kappa"],
                   budget=cfg["migration_budget_B"], d=d, k=k, policy=policy,
                   cooldown=spec["params"].get("cooldown", 0), forecast=window)
    met = compute_metrics(res, rep.load, slo=cfg["slo_threshold"], wasted_window=cfg["wasted_move_window"],
                          wait_quantile=cfg["wait_quantile"])
    ref = run_replicate(cfg, rep, spec["name"], spec["params"], forecast=window)
    if ref != met:
        raise RuntimeError("replay differs from run_replicate for " + spec["policy"])
    return res, met


def build_episode(cfg, dataset, tuned, ext, results, must_rows, policies):
    data = load_dataset(cfg, dataset)
    pick = select_episode(must_rows, data.events, cfg, dataset)
    ev = data.events[data.events["event_id"] == pick["event_id"]].iloc[0]
    rep = make_replicate(cfg, data, ev, pick["replicate"], cfg["headroom"])
    q1_dir = Path(results) / Path(cfg["result_folders"]["q1"]).name
    q1_rows = read_rows(q1_dir, pick["event_id"])
    runs, checks, windows = {}, [], {}
    specs = episode_specs(cfg, tuned, ext, dataset, policies)
    for spec in specs:
        fr = frozen_row(q1_rows if spec["where"] else must_rows, dataset, pick["event_id"], pick["replicate"], cfg,
                        spec)
        window = None
        if spec["forecaster"] is not None:
            window = forecast_window(load_forecast(cfg, dataset, spec["forecaster"]), rep)
        res, met = replay(cfg, rep, spec, window)
        runs[spec["policy"]] = (res, met)
        windows[spec["policy"]] = window
        checks.append({"dataset": dataset, "event_id": pick["event_id"], "replicate": pick["replicate"],
                       "policy": spec["policy"], "frozen_cell": fr["cell"], "frozen_damage": fr["damage"],
                       "regenerated_damage": met["damage"], "frozen_moves": fr["moves"],
                       "regenerated_moves": met["moves"], "equal": compare_row(fr, met)})
    hz = horizon_index(cfg)
    p3 = [s for s in specs if s["policy"] == "P3"][0]
    fc = {"forecaster": p3["forecaster"], "horizon": cfg["horizons"][hz],
          "total": forecast_total(windows["P3"], hz)}
    return pick, rep, runs, checks, fc


def draw_episode(cfg, dataset, pick, rep, runs, checks, fc, style, out_dir):
    policies = style["episode_policies"]
    n_steps = rep.load.shape[0]
    x = np.arange(n_steps) - (rep.onset - rep.start)
    on, off = 0, rep.end - rep.onset
    fig, axes = plt.subplots(len(PANELS), 1, figsize=style["size_fa2"], sharex=True, squeeze=False)
    axes = axes[:, 0]
    axes[0].plot(x, rep.load.sum(axis=1), color=style["episode_load_color"])
    axes[0].plot(x + fc["horizon"], fc["total"], color=style["forecast_color"])
    axes[0].axhline(rep.mu.sum(), color=style["capacity_color"], linestyle=style["capacity_linestyle"])
    for p in policies:
        res = runs[p][0]
        colour = style["policy_colors"][p]
        axes[1].plot(x, res.wait.max(axis=1), color=colour)
        starts = np.array([mv[0] for mv in res.moves], dtype=np.int64)
        axes[2].step(x, np.cumsum(np.bincount(starts, minlength=n_steps)), where="post", color=colour)
    axes[1].set_yscale("symlog", linthresh=cfg["slo_threshold"])
    axes[1].axhline(cfg["slo_threshold"], color=style["zero_line_color"], linestyle=style["slo_linestyle"])
    for ax, label in zip(axes, PANELS):
        ax.axvspan(on, off, color=style["span_color"], alpha=style["span_alpha"], linewidth=0)
        ax.set_ylabel(label)
    axes[len(PANELS) - 1].set_xlabel("step relative to event onset")
    handles, labels = policy_handles(policies, style)
    handles += [plt.Line2D([], [], color=style["episode_load_color"]),
                plt.Line2D([], [], color=style["forecast_color"]),
                plt.Line2D([], [], color=style["capacity_color"], linestyle=style["capacity_linestyle"]),
                Patch(color=style["span_color"], alpha=style["span_alpha"])]
    labels += ["total load", "forecast, %d steps ahead" % fc["horizon"], "total capacity", "event onset to end"]
    fig.legend(handles, labels, loc="outside lower center", ncol=len(policies))
    damages = "; ".join("%s %s" % (style["policy_labels"][c["policy"]], format(c["regenerated_damage"], ".6g"))
                        for c in checks)
    cap = ("FA2 and review Figure 3. Regenerated flash-crowd episode on %s: event %s (peak %s), replicate %d. One "
           "replicate selected by a rule fixed beforehand, not an average, no interval. Rule: among the %d sampled "
           "abrupt evaluation events of the default cell (m %s, headroom %s) the event with the median peak "
           "(lower middle on an even count, ties by event id), with the replicate index it has in the frozen rows. "
           "Top: total offered load per step, total node capacity and the point forecast of the P4 forecaster (%s), summed "
           "over partitions, for the horizon of %d steps (the default migration duration), the forecast made at step i "
           "drawn at its target step i + %d; P3 and P4a act on forecasts of this forecaster and P4a adds an interval "
           "margin that is not drawn; the shaded span runs from event onset to end. "
           "Middle: worst-node wait proxy in steps (symlog axis, SLO threshold %s dotted). Bottom: migrations "
           "started, cumulative. Policies R1 (P1), P3 (forecaster of P4), P4a, P5. The regenerated damage and number "
           "of moves of every drawn policy equal the frozen rows (damage: %s). Window %d steps of %d s, master_seed "
           "%d. Sources: results_frozen Must rows (P1, P3, P5) and Q1 AA6 rows (P4a), regenerated with the frozen "
           "code and configs; table of the comparison: figures/%s."
           % (dataset, pick["event_id"], format(pick["peak"], ".6g"), pick["replicate"], pick["n_events"],
              cfg["migration_steps_m0"], cfg["headroom"], fc["forecaster"], fc["horizon"], fc["horizon"],
              cfg["slo_threshold"], damages, n_steps,
              step_seconds(cfg, dataset), cfg["master_seed"], CHECK_NAME))
    return rf.save(fig, out_dir, "fa2_episode_" + dataset, cap, style)


def make_episodes(cfg, results, tuned_path, ext_path, out_dir, style_path, datasets=DATASETS):
    style = load_style4(style_path)
    plt.rcParams.update(style["rcparams"])
    tuned = yaml.safe_load(Path(tuned_path).read_text(encoding="utf-8"))
    ext = yaml.safe_load(Path(ext_path).read_text(encoding="utf-8"))
    must = read_rows(results)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths, all_checks, bad = [], [], []
    for ds in datasets:
        pick, rep, runs, checks, fc = build_episode(cfg, ds, tuned, ext, results, must, style["episode_policies"])
        all_checks.extend(checks)
        print("%s event %s replicate %d peak %s (%d sampled events)"
              % (ds, pick["event_id"], pick["replicate"], pick["peak"], pick["n_events"]))
        for c in checks:
            print("  %-4s frozen %s regenerated %s moves %d/%d equal %s"
                  % (c["policy"], c["frozen_damage"], c["regenerated_damage"], c["frozen_moves"],
                     c["regenerated_moves"], c["equal"]))
        if all(c["equal"] for c in checks):
            paths.append(draw_episode(cfg, ds, pick, rep, runs, checks, fc, style, out_dir))
        else:
            bad.append(ds)
            print("  MISMATCH: %s not drawn" % ds)
    pd.DataFrame(all_checks).to_csv(out / CHECK_NAME, index=False, lineterminator="\n")
    return paths, bad
