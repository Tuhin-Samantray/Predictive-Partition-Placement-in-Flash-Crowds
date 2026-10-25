"""Cross-layer metrics CL1 to CL6 and CL9 (Phase 12, A74). Pure functions; every parameter is an argument.

Rows are the Q3 rows (P1, P3, P5 and P6 under the cross-layer model) at the default m and headroom. A setting is
(dataset, variant, mode, bandwidth tier h_net, phi); constant-m rows carry no phi. The unit is the event: a per-event
value is the mean over the rows of that event inside the setting, and the bootstrap resamples events (BCa). The
tables are descriptive: no hypothesis label and no Holm family (the guide gives none).
CL1 per-layer utilization, CL2 per-layer share of violations, CL6 migration traffic share: mean per setting.
CL3 attribution: ratio of sums of the attributed violating node-steps (A57 item 5). CL4: ratio of the sums of the
coupled wait and the replay wait (migration traffic off), above 1 = migration traffic adds wait. CL5: Spearman
correlation across events of the CPU and the network overload node-steps. CL9: joint cost over a weight simplex
on terms scaled by their mean over all coupled rows of the dataset, then the Pareto front over the policies.
"""
import numpy as np
import pandas as pd

from src.seeds import derive_seed
from src.stats_tools import bca_interval, bca_paired, mean_stat, spearman

NAN = float("nan")
COUPLED = "coupled"
UNIT = ["dataset", "event_id"]
SETTING = ["dataset", "variant", "mode", "h_net", "phi"]
LAYER_METRICS = (("CL1", ("cl_util_cpu", "cl_util_net")),
                 ("CL2", ("cl_viol_cpu", "cl_viol_net", "cl_viol_both")),
                 ("CL6", ("cl_mig_share", "cl_bytes_sent")))
TERMS = ("cl_bytes_sent", "cl_wait_net_mean", "cl_wait_cpu_mean")
WEIGHTS = ("weight_bytes", "weight_nic_wait", "weight_cpu_wait")


def default_rows(q3, m, headroom):
    return q3[(q3["m"] == m) & (q3["headroom"] == headroom)]


def coupled_rows(q3, m, headroom):
    rows = default_rows(q3, m, headroom)
    return rows[rows["mode"] == COUPLED]


def settings(rows, columns=SETTING):
    """(key, sub-frame) per setting in sorted order; a missing phi (constant-m) is kept as its own group."""
    for key, sub in rows.groupby(list(columns), sort=True, dropna=False):
        yield key, sub


def label(key):
    return [str(k) for k in key]


def per_event(sub, columns):
    return sub.groupby(UNIT, sort=True)[list(columns)].mean()


def ratio_of_sums(x, y):
    """Sum of x over the sum of y; nan when the denominator sum is not positive."""
    denominator = float(np.sum(y))
    return float(np.sum(x) / denominator) if denominator > 0 else NAN


def key_fields(key, columns=SETTING):
    return dict(zip(columns, key))


def layers_table(q3, m, headroom, n_boot, level, master_seed):
    """CL1, CL2 and CL6: mean per event with a BCa interval, per setting and metric (long format)."""
    rows = []
    for key, sub in settings(default_rows(q3, m, headroom)):
        for cl, metrics in LAYER_METRICS:
            events = per_event(sub, metrics)
            for metric in metrics:
                values = events[metric].to_numpy()
                lo, hi = bca_interval(values, mean_stat, n_boot, level,
                                      derive_seed("cl1236", master_seed, metric, *label(key)))
                rows.append(dict(key_fields(key), cl=cl, metric=metric, n_events=int(values.size),
                                 mean=float(values.mean()), ci_low=lo, ci_high=hi))
    return pd.DataFrame(rows)


def attribution_table(q3, m, headroom, n_boot, level, master_seed):
    """CL3: share of the violating node-steps whose binding layer is the network (and the CPU), ratio of sums over
    events with a BCa interval (the two sums are resampled together); nan without violating steps."""
    rows = []
    for key, sub in settings(default_rows(q3, m, headroom)):
        work = sub.assign(net_steps=sub["attr_net_share"] * sub["attr_violating_steps"],
                          cpu_steps=sub["attr_cpu_share"] * sub["attr_violating_steps"])
        events = work.groupby(UNIT, sort=True)[["net_steps", "cpu_steps", "attr_violating_steps"]].sum()
        total = events["attr_violating_steps"].to_numpy()
        row = dict(key_fields(key), n_events=int(len(events)), n_violating_steps=int(total.sum()))
        for name in ("net", "cpu"):
            part = events[name + "_steps"].to_numpy()
            if total.sum() > 0:
                lo, hi = bca_paired(part, total, ratio_of_sums, n_boot, level,
                                    derive_seed("cl3", master_seed, name, *label(key)))
            else:
                lo, hi = NAN, NAN
            row.update({name + "_share": ratio_of_sums(part, total), name + "_ci_low": lo, name + "_ci_high": hi})
        rows.append(row)
    return pd.DataFrame(rows)


def interference_table(q3, m, headroom, n_boot, level, master_seed):
    """CL4: sum of the coupled mean wait over the sum of the replay mean wait (same moves, same start and finish
    steps, migration bytes off), per coupled setting; replay_ok_all is the logged consistency flag."""
    rows = []
    for key, sub in settings(coupled_rows(q3, m, headroom)):
        events = per_event(sub, ("cl_wait_mean", "cl_wait_mean_replay"))
        ok = bool(sub["cl_replay_ok"].astype(bool).all())
        wait = events["cl_wait_mean"].to_numpy()
        replay = events["cl_wait_mean_replay"].to_numpy()
        lo, hi = bca_paired(wait, replay, ratio_of_sums, n_boot, level, derive_seed("cl4", master_seed, *label(key)))
        rows.append(dict(key_fields(key), n_events=int(wait.size), mean_wait=float(wait.mean()),
                         mean_wait_replay=float(replay.mean()), ratio=ratio_of_sums(wait, replay), ci_low=lo,
                         ci_high=hi, replay_ok_all=ok))
    return pd.DataFrame(rows)


def coupling_table(q3, m, headroom, phi, n_boot, level, master_seed):
    """CL5: per (dataset, variant, tier) in coupled mode at phi, the Spearman correlation across events of the
    per-event CPU overload node-steps and network overload node-steps with a BCa interval over events; nan when
    either variable is constant over the events (stated, not imputed)."""
    sel = coupled_rows(q3, m, headroom)
    sel = sel[np.isclose(sel["phi"].astype(float), phi)]
    rows = []
    for key, sub in settings(sel, ("dataset", "variant", "h_net")):
        events = per_event(sub, ("cl_over_cpu", "cl_over_net"))
        cpu = events["cl_over_cpu"].to_numpy()
        net = events["cl_over_net"].to_numpy()
        constant = bool(np.ptp(cpu) == 0 or np.ptp(net) == 0)
        if constant or cpu.size < 2:
            rho, lo, hi = NAN, NAN, NAN
        else:
            rho = spearman(cpu, net)
            lo, hi = bca_paired(cpu, net, spearman, n_boot, level,
                                derive_seed("cl5", master_seed, *label(key)))
        rows.append(dict(key_fields(key, ("dataset", "variant", "h_net")), phi=phi, n_events=int(cpu.size),
                         rho=rho, ci_low=lo, ci_high=hi, constant=constant,
                         mean_over_cpu=float(cpu.mean()), mean_over_net=float(net.mean())))
    return pd.DataFrame(rows)


def weight_grid(step):
    """All weight triples (bytes, NIC wait, CPU wait) in multiples of step that sum to 1 with every weight at
    least step."""
    k = int(round(1 / step))
    return [(i / k, j / k, (k - i - j) / k) for i in range(1, k) for j in range(1, k - i)]


def dominated(points):
    """Flag of the rows (policies) that another row beats or ties on every term and beats on at least one."""
    p = np.asarray(points, dtype=np.float64)
    flags = []
    for i in range(len(p)):
        flags.append(bool(any(np.all(p[j] <= p[i]) and np.any(p[j] < p[i]) for j in range(len(p)) if j != i)))
    return flags


def joint_tables(q3, m, headroom, phi, step):
    """CL9: terms scaled by their mean over all coupled rows of the dataset (zero-mean terms stay 0); per
    (dataset, tier) the mean scaled terms per policy at phi, the Pareto flag, and J over the weight simplex.
    Returns (terms table, weights table in long format). Ties at the minimum credit every tied policy."""
    allrows = coupled_rows(q3, m, headroom)
    scale = allrows.groupby("dataset", sort=True)[list(TERMS)].mean()
    sel = allrows[np.isclose(allrows["phi"].astype(float), phi)]
    grid = weight_grid(step)
    terms, weights = [], []
    for (dataset, tier), data in sel.groupby(["dataset", "h_net"], sort=True):
        means = {}
        for variant, sub in data.groupby("variant", sort=True):
            events = per_event(sub, TERMS)
            means[variant] = [float(events[t].mean() / scale.loc[dataset, t]) if scale.loc[dataset, t] > 0 else 0.0
                              for t in TERMS]
        names = sorted(means)
        flags = dominated([means[v] for v in names])
        wins = dict.fromkeys(names, 0)
        for a, b, c in grid:
            cost = {v: a * means[v][0] + b * means[v][1] + c * means[v][2] for v in names}
            best = min(cost.values())
            tied = [v for v in names if np.isclose(cost[v], best)]
            for v in tied:
                wins[v] += 1
            for v in names:
                weights.append(dict(dataset=dataset, h_net=tier, phi=phi, weight_bytes=a, weight_nic_wait=b,
                                    weight_cpu_wait=c, variant=v, J=cost[v], best=v in tied))
        for v, flag in zip(names, flags):
            terms.append(dict(dataset=dataset, h_net=tier, phi=phi, variant=v, bytes_scaled=means[v][0],
                              nic_wait_scaled=means[v][1], cpu_wait_scaled=means[v][2], pareto=not flag,
                              weight_points=len(grid), weight_wins=wins[v],
                              scale_bytes=float(scale.loc[dataset, TERMS[0]]),
                              scale_nic_wait=float(scale.loc[dataset, TERMS[1]]),
                              scale_cpu_wait=float(scale.loc[dataset, TERMS[2]])))
    return pd.DataFrame(terms), pd.DataFrame(weights)
