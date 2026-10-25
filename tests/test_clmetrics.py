import numpy as np
import pandas as pd
import pytest

from src.clmetrics import (COUPLED, attribution_table, coupling_table, dominated, interference_table, joint_tables,
                           layers_table, weight_grid)

TIERS = [1.3, 2.0, 4.0]
N_BOOT = 199
LEVEL = 0.95
SEED = 7
M = 5
HEADROOM = 1.6
LOW_PHI = 0.01
CONST_M = "const_m"
DATASETS = ("D1", "D2")
VARIANTS = ("P1", "P3", "P5", "P6")


def q3_frame(n, spec):
    """spec(dataset, variant, tier, mode, phi, j) -> dict of field values (missing fields default to 0)."""
    rows = []
    for dataset in DATASETS:
        for j in range(n):
            for variant in VARIANTS:
                for tier in TIERS:
                    for mode, phi in ((COUPLED, 1.0), (COUPLED, LOW_PHI), (CONST_M, None)):
                        row = dict(dataset=dataset, onset_class="abrupt", m=M, headroom=HEADROOM, replicate=j,
                                   event_id="%s-%d" % (dataset, j), variant=variant, mode=mode, h_net=tier, phi=phi,
                                   damage=1.0, cl_util_cpu=0.5, cl_util_net=0.2, cl_viol_cpu=0.1, cl_viol_net=0.05,
                                   cl_viol_both=0.01, cl_mig_share=0.02, cl_bytes_sent=10.0, cl_over_cpu=3.0,
                                   cl_over_net=2.0, cl_wait_mean=2.0, cl_wait_mean_replay=1.0, cl_wait_cpu_mean=1.0,
                                   cl_wait_net_mean=1.0, cl_replay_ok=True, attr_net_share=0.25, attr_cpu_share=0.75,
                                   attr_violating_steps=10)
                        row.update(spec(dataset, variant, tier, mode, phi, j))
                        rows.append(row)
    return pd.DataFrame(rows)


def plain(*_args):
    return {}


def test_layers_table_recovers_planted_means_and_keeps_const_m_rows():
    q3 = q3_frame(12, lambda d, v, t, mode, phi, j: {"cl_util_cpu": 0.4 + 0.01 * j} if v == "P6" else {})
    table = layers_table(q3, M, HEADROOM, N_BOOT, LEVEL, SEED)
    assert len(table) == 2 * 4 * 3 * 3 * 7
    row = table[(table["dataset"] == "D1") & (table["variant"] == "P6") & (table["h_net"] == 1.3)
                & (table["mode"] == COUPLED) & (table["phi"] == 1.0) & (table["metric"] == "cl_util_cpu")].iloc[0]
    assert row["cl"] == "CL1" and row["n_events"] == 12
    assert row["mean"] == pytest.approx(0.4 + 0.01 * 5.5)
    assert row["ci_low"] < row["mean"] < row["ci_high"]
    const = table[table["mode"] == CONST_M]
    assert len(const) == 2 * 4 * 3 * 7 and const["phi"].isna().all()
    flat = table[(table["variant"] == "P1") & (table["metric"] == "cl_util_net")]
    assert np.allclose(flat["mean"], 0.2) and (flat["ci_low"] == flat["ci_high"]).all()
    other = layers_table(q3.assign(m=10), M, HEADROOM, N_BOOT, LEVEL, SEED)
    assert other.empty


def test_attribution_is_a_ratio_of_sums_not_a_mean_of_shares():
    def spec(d, v, t, mode, phi, j):
        return {"attr_net_share": 0.5 if j % 2 == 0 else 0.1, "attr_violating_steps": 10 if j % 2 == 0 else 30,
                "attr_cpu_share": 0.5 if j % 2 == 0 else 0.9}
    table = attribution_table(q3_frame(10, spec), M, HEADROOM, N_BOOT, LEVEL, SEED)
    row = table[(table["dataset"] == "D1") & (table["variant"] == "P3")].iloc[0]
    assert row["n_violating_steps"] == 200 and row["net_share"] == pytest.approx((5 * 5 + 5 * 3) / 200)
    assert row["cpu_share"] == pytest.approx((5 * 5 + 5 * 27) / 200)
    assert row["net_ci_low"] < row["net_share"] < row["net_ci_high"]
    none = attribution_table(q3_frame(10, lambda *a: {"attr_violating_steps": 0}), M, HEADROOM, N_BOOT, LEVEL, SEED)
    assert none["net_share"].isna().all() and (none["n_violating_steps"] == 0).all()


def test_interference_ratio_uses_coupled_rows_only_and_the_replay_flag():
    def spec(d, v, t, mode, phi, j):
        return {"cl_wait_mean": 3.0 + 0.1 * j, "cl_wait_mean_replay": 1.5 + 0.05 * j} if mode == COUPLED else {
            "cl_wait_mean": 9.0, "cl_wait_mean_replay": float("nan")}
    q3 = q3_frame(10, spec)
    table = interference_table(q3, M, HEADROOM, N_BOOT, LEVEL, SEED)
    assert len(table) == 2 * 4 * 3 * 2 and set(table["mode"]) == {COUPLED}
    assert np.allclose(table["ratio"], 2.0) and table["replay_ok_all"].all()
    broken = q3.copy()
    broken.loc[(broken["dataset"] == "D1") & (broken["variant"] == "P1") & (broken["replicate"] == 0), "cl_replay_ok"] = False
    flags = interference_table(broken, M, HEADROOM, N_BOOT, LEVEL, SEED)
    bad = flags[(flags["dataset"] == "D1") & (flags["variant"] == "P1")]
    assert not bad["replay_ok_all"].any() and flags["replay_ok_all"].sum() == len(flags) - len(bad)


def test_coupling_spearman_is_one_for_monotone_overloads_and_nan_for_constants():
    def spec(d, v, t, mode, phi, j):
        if v == "P1":
            return {"cl_over_cpu": float(j), "cl_over_net": 2.0 * j + 1}
        if v == "P3":
            return {"cl_over_cpu": float(j), "cl_over_net": float(-j)}
        return {}
    table = coupling_table(q3_frame(12, spec), M, HEADROOM, 1.0, N_BOOT, LEVEL, SEED)
    assert len(table) == 2 * 4 * 3 and (table["phi"] == 1.0).all()
    p1 = table[(table["dataset"] == "D1") & (table["variant"] == "P1")]
    p3 = table[(table["dataset"] == "D1") & (table["variant"] == "P3")]
    p5 = table[(table["dataset"] == "D1") & (table["variant"] == "P5")]
    assert np.allclose(p1["rho"], 1.0) and np.allclose(p3["rho"], -1.0)
    assert p5["rho"].isna().all() and p5["constant"].all() and not p1["constant"].any()


def test_weight_grid_and_dominance():
    grid = weight_grid(0.1)
    assert len(grid) == 36
    assert all(abs(sum(w) - 1) < 1e-9 and min(w) >= 0.1 - 1e-9 for w in grid)
    assert len(set(grid)) == 36
    assert dominated([[1, 1, 1], [2, 2, 2], [1, 3, 0]]) == [False, True, False]
    assert dominated([[1, 1, 1], [1, 1, 1]]) == [False, False]


def test_joint_cost_pareto_flags_wins_and_scales():
    def spec(d, v, t, mode, phi, j):
        bytes_sent, net, cpu = {"P1": (10.0, 5.0, 5.0), "P3": (30.0, 5.0, 2.0), "P5": (40.0, 9.0, 9.0),
                                "P6": (10.0, 1.0, 4.0)}[v]
        return {"cl_bytes_sent": bytes_sent, "cl_wait_net_mean": net, "cl_wait_cpu_mean": cpu}
    q3 = q3_frame(8, spec)
    terms, weights = joint_tables(q3, M, HEADROOM, 1.0, 0.1)
    assert len(terms) == 2 * 3 * 4 and len(weights) == 2 * 3 * 4 * 36
    top = terms[(terms["dataset"] == "D1") & (terms["h_net"] == 1.3)].set_index("variant")
    assert top["scale_bytes"].iloc[0] == pytest.approx(22.5) and top["scale_nic_wait"].iloc[0] == pytest.approx(5.0)
    assert top.loc["P6", "bytes_scaled"] == pytest.approx(10 / 22.5)
    assert list(top["pareto"]) == [False, True, False, True] or dict(top["pareto"]) == {"P1": False, "P3": True,
                                                                                         "P5": False, "P6": True}
    assert top.loc["P5", "weight_wins"] == 0 and top.loc["P1", "weight_wins"] == 0
    assert top["weight_wins"].sum() == 36
    assert top.loc["P6", "weight_wins"] + top.loc["P3", "weight_wins"] == 36
    best = weights[(weights["dataset"] == "D1") & (weights["h_net"] == 1.3)]
    assert best.groupby(["weight_bytes", "weight_nic_wait", "weight_cpu_wait"])["best"].sum().eq(1).all()
    assert (best.groupby("variant")["best"].sum() == top["weight_wins"]).all()
    const_only = q3[q3["mode"] == CONST_M]
    assert joint_tables(const_only, M, HEADROOM, 1.0, 0.1)[0].empty


def test_joint_cost_zero_terms_stay_zero_and_ties_credit_every_policy():
    q3 = q3_frame(6, lambda d, v, t, mode, phi, j: {"cl_wait_net_mean": 0.0})
    terms, _weights = joint_tables(q3, M, HEADROOM, 1.0, 0.1)
    assert (terms["nic_wait_scaled"] == 0).all() and (terms["scale_nic_wait"] == 0).all()
    assert terms["pareto"].all() and (terms["weight_wins"] == 36).all()
