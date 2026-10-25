"""Analysis stages (Phase 12). Usage:
  python scripts/05_analyze.py --config configs/default.yaml --results results --out tables --stage interaction_trigger
Every table is regenerated from the result files, nothing is hand-edited. A stage is a function of (cfg, folders, out);
--results re-roots the folders of config result_folders (results, a frozen copy or a clean-room copy). --stage all runs
every stage in the order of STAGES.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis_data import load_queues, resolve_folders  # noqa: E402
from src.config import load_config  # noqa: E402
from src.interaction import interaction_effects, interaction_trigger  # noqa: E402
from src.lagtax import c1_table, load_best_reactive  # noqa: E402
from src.recovery import c2_table, c3_table, load_headline, recovery_frame  # noqa: E402
from src.regime import regime_must, regime_q2  # noqa: E402
from src.maincomp import c11_table, c13_table  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from src.benefit import POOLED, benefit_cells, c4_table, class_event_mase  # noqa: E402
from src.ranking import POLICIES, c6_tables  # noqa: E402
from src.replicates import load_dataset  # noqa: E402
from src.transfer import FIXED, POOLED as ALL_GROUP, c5_tables  # noqa: E402
from src.crosslayer import c7_table, c8_table, c9_table  # noqa: E402
from src.p4a import c10_table, rerun_consistency  # noqa: E402
from src.clmetrics import (attribution_table, coupling_table, interference_table, joint_tables,  # noqa: E402
                           layers_table)
from src.rvtables import rv0_table, rv1_table, rv23_tables, rv4_table  # noqa: E402
from src.reviewtables import policy_metrics_table, rv1_metrics_table  # noqa: E402
from src.ablations import ablation_table  # noqa: E402
from src.xatables import baseline_overlap_mismatch, xa1_table, xa5_table, xa7_mase_table, xa7_warning_table  # noqa: E402
from src.ledger import NEEDED as LEDGER_TABLES, ledger_table  # noqa: E402
from src.tatables import ta3, ta4, ta5, ta6, ta7, ta8, ta9  # noqa: E402


def stage_interaction_trigger(cfg, folders, out):
    """A38: evaluated on the Q1 rows before any Q1 effect is interpreted; writes the effects and the trigger table."""
    queues = load_queues({name: folders[name] for name in ("must", "q1")})
    effects = interaction_effects(queues["q1"], queues["must"], cfg["migration_steps_m0"], cfg["headroom"],
                                  cfg["margin_quantile"], cfg["n_boot"], cfg["ci_level"],
                                  cfg["interaction_effect_frac"], cfg["master_seed"])
    trigger = interaction_trigger(effects, cfg["interaction_min_active"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    effects.to_csv(out / "interaction_effects.csv", index=False)
    trigger.to_csv(out / "interaction_trigger.csv", index=False)
    print(trigger.to_string(index=False))
    print("effect rows %d; trigger fires for %d of %d (dataset, policy) pairs" % (
        len(effects), int(trigger["fires"].sum()), len(trigger)))
    return trigger


def stage_c1_lag_tax(cfg, folders, out):
    """C1 (H1, A62): lag tax per dataset, one-sided Wilcoxon at the default m and headroom and Spearman of tax against m."""
    must = load_queues({"must": folders["must"]})["must"]
    best = load_best_reactive(Path(__file__).resolve().parents[1] / "configs" / "tuned.yaml")
    table = c1_table(must, best, cfg["migration_steps_m0"], cfg["headroom"], cfg["m_levels"], cfg["n_boot"],
                     cfg["ci_level"], cfg["alpha_adj"], cfg["min_events"], cfg["master_seed"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "c1_lag_tax.csv", index=False)
    print(table.to_string(index=False))
    return table


def recovery_rows(cfg, folders):
    """Per-replicate reactive, headline and oracle damage with the lag tax and R (A62, A64)."""
    must = load_queues({"must": folders["must"]})["must"]
    tuned = Path(__file__).resolve().parents[1] / "configs" / "tuned.yaml"
    return recovery_frame(must, load_best_reactive(tuned), load_headline(tuned), cfg["min_lag_tax_share"])


def stage_c2_recovery_class(cfg, folders, out):
    """C2 (H2, A62, A64): median R by onset class at the default m and headroom, Mann-Whitney U and Cliff's delta."""
    table = c2_table(recovery_rows(cfg, folders), cfg["migration_steps_m0"], cfg["headroom"], cfg["h2_split"],
                     cfg["n_boot"], cfg["ci_level"], cfg["alpha_adj"], cfg["min_events"], cfg["master_seed"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "c2_recovery_class.csv", index=False)
    print(table.to_string(index=False))
    return table


def stage_c3_cheap_migration(cfg, folders, out):
    """C3 (H3, A62, A64): median R at the smallest m level against m at least the default m."""
    table = c3_table(recovery_rows(cfg, folders), cfg["m_levels"], cfg["migration_steps_m0"], cfg["headroom"],
                     cfg["h3_ub"], cfg["n_boot"], cfg["ci_level"], cfg["alpha_adj"], cfg["min_events"],
                     cfg["master_seed"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "c3_cheap_migration.csv", index=False)
    print(table.to_string(index=False))
    return table


def stage_regime_map_q2(cfg, folders, out):
    """Regime map of the D3 events (A66): median R by rise bucket and m at the default headroom, Q2 forecast variant (A36)."""
    q2 = load_queues({"q2": folders["q2"]})["q2"]
    best = load_best_reactive(Path(__file__).resolve().parents[1] / "configs" / "tuned.yaml")
    table = regime_q2(q2, best, cfg["q2_forecast_variants"], cfg["min_lag_tax_share"], cfg["n_boot"],
                      cfg["ci_level"], cfg["min_events"], cfg["master_seed"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "regime_map_q2.csv", index=False)
    print(table.to_string(index=False))
    return table


def stage_regime_map_must(cfg, folders, out):
    """Regime map of the Must grid (A66): median R by onset class, m and headroom for the headline P3 and for P4."""
    must = load_queues({"must": folders["must"]})["must"]
    tuned = Path(__file__).resolve().parents[1] / "configs" / "tuned.yaml"
    best = load_best_reactive(tuned)
    policies = [load_headline(tuned), {dataset: "P4" for dataset in best}]
    table = regime_must(must, best, policies, cfg["min_lag_tax_share"], cfg["n_boot"], cfg["ci_level"],
                        cfg["min_events"], cfg["master_seed"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "regime_map_must.csv", index=False)
    print(table.to_string(index=False))
    return table


def stage_c11_main_comparison(cfg, folders, out):
    """C11 (A68, family F9): each policy against the best tuned reactive policy at the default m and headroom."""
    queues = load_queues({name: folders[name] for name in ("must", "q1", "q3")})
    tuned = Path(__file__).resolve().parents[1] / "configs" / "tuned.yaml"
    table = c11_table(queues["must"], queues["q1"], queues["q3"], load_best_reactive(tuned), load_headline(tuned),
                      cfg["migration_steps_m0"], cfg["headroom"], cfg["margin_quantile"], cfg["h_net"], cfg["phi"],
                      cfg["alpha_adj"], cfg["n_boot"], cfg["ci_level"], cfg["master_seed"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "c11_main_comparison.csv", index=False)
    print(table.to_string(index=False))
    return table


def stage_c13_p4a_pairs(cfg, folders, out):
    """C13 (A68, family F10): P4a against each of P1, P0, P3, P4 and P5 at the default m and headroom."""
    queues = load_queues({name: folders[name] for name in ("must", "q1")})
    tuned = Path(__file__).resolve().parents[1] / "configs" / "tuned.yaml"
    table = c13_table(queues["must"], queues["q1"], load_headline(tuned), cfg["migration_steps_m0"], cfg["headroom"],
                      cfg["margin_quantile"], cfg["alpha_adj"], cfg["n_boot"], cfg["ci_level"], cfg["master_seed"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "c13_p4a_pairs.csv", index=False)
    print(table.to_string(index=False))
    return table


def stage_c4_accuracy_benefit(cfg, folders, out):
    """C4 (A70, H4): Spearman of the per-class event MASE at the stored horizon nearest m + d against the cell median R of P3."""
    must = load_queues({"must": folders["must"]})["must"]
    tuned = Path(__file__).resolve().parents[1] / "configs" / "tuned.yaml"
    names = cfg["forecaster_names"]
    cells = benefit_cells(must, load_best_reactive(tuned), names, cfg["migration_steps_m0"], cfg["headroom"],
                          cfg["min_lag_tax_share"])
    horizon = min(cfg["horizons"], key=lambda h: abs(h - cfg["migration_steps_m0"] - cfg["monitor_delay_d"]))
    base = Path(cfg["data_root"]) / cfg["processed_dirname"] / cfg["forecast_dirname"]
    rows = []
    for dataset in sorted(must["dataset"].unique()):
        data = load_dataset(cfg, dataset)
        for name in names:
            table = np.load(base / ("%s_%s.npy" % (dataset, name)), mmap_mode="r")
            got = class_event_mase(table, data.load, data.events, data.train_end, data.tune_end, cfg["horizons"],
                                   horizon, cfg["smape_eps"])
            for cls, (value, n_cells) in got.items():
                if cls != POOLED:
                    rows.append({"forecaster": name, "dataset": dataset, "onset_class": cls, "horizon": horizon,
                                 "mase_event_class": value, "n_mase_cells": n_cells})
    detail, summary = c4_table(cells, pd.DataFrame(rows), cfg["h4_rho"], cfg["h4_ub"], cfg["n_boot"], cfg["ci_level"],
                               cfg["master_seed"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out / "c4_accuracy_benefit_cells.csv", index=False)
    summary.to_csv(out / "c4_accuracy_benefit.csv", index=False)
    print(detail.to_string(index=False))
    print(summary.to_string(index=False))
    return summary


def stage_c5_transfer(cfg, folders, out):
    """C5 (A70, H5): forecaster transfer ratios (tables/forecast_cross.csv, fixed protocol) and XA2 tuning-transfer ratios (Q8)."""
    cross = pd.read_csv(Path(__file__).resolve().parents[1] / "tables" / "forecast_cross.csv")
    q8 = load_queues({"q8": folders["q8"]})["q8"]
    forecasters = sorted(cross.loc[cross["protocol"] == FIXED, "forecaster"].unique())
    detail, summary = c5_tables(cross, q8, forecasters, cfg["horizons"], ALL_GROUP, cfg["h5_ratio"], cfg["n_boot"],
                                cfg["ci_level"], cfg["master_seed"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out / "c5_transfer_detail.csv", index=False)
    summary.to_csv(out / "c5_transfer_summary.csv", index=False)
    print(detail.to_string(index=False))
    print(summary.to_string(index=False))
    return summary


def stage_c6_ranking(cfg, folders, out):
    """C6 (A70, H6): Kendall tau of the D1 and D2 policy rankings by median per-event damage at the default m and headroom."""
    must = load_queues({"must": folders["must"]})["must"]
    tuned = Path(__file__).resolve().parents[1] / "configs" / "tuned.yaml"
    detail, summary = c6_tables(must, load_headline(tuned), list(POLICIES), cfg["migration_steps_m0"], cfg["headroom"],
                                cfg["h6_tau"], cfg["n_boot"], cfg["ci_level"], cfg["master_seed"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out / "c6_ranking.csv", index=False)
    summary.to_csv(out / "c6_ranking_tau.csv", index=False)
    print(detail.to_string(index=False))
    print(summary.to_string(index=False))
    return summary


def write_table(out, name, table):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / name, index=False)
    print(table.to_string(index=False))


def stage_c7_h9_share(cfg, folders, out):
    """C7 (A72, H9, family F5): share of violating node-steps attributed to the network layer under P3 and the damage difference P3 minus P6, per dataset and bandwidth tier."""
    q3 = load_queues({"q3": folders["q3"]})["q3"]
    table = c7_table(q3, cfg["migration_steps_m0"], cfg["headroom"], cfg["h_net_levels"], cfg["phi"], cfg["h9_share"],
                     cfg["alpha_adj"], cfg["min_events"], cfg["n_boot"], cfg["ci_level"], cfg["master_seed"])
    write_table(out, "c7_h9.csv", table)
    return table


def stage_c8_h10_coupling(cfg, folders, out):
    """C8 (A72, H10, family F6): per-event R of P3 in coupled mode minus R in constant-m mode, per dataset and tier."""
    q3 = load_queues({"q3": folders["q3"]})["q3"]
    table = c8_table(q3, cfg["migration_steps_m0"], cfg["headroom"], cfg["h_net_levels"], cfg["phi"],
                     cfg["min_lag_tax_share"], cfg["alpha_adj"], cfg["min_events"], cfg["n_boot"], cfg["ci_level"],
                     cfg["master_seed"])
    write_table(out, "c8_h10.csv", table)
    return table


def stage_c9_h11_disaggregated(cfg, folders, out):
    """C9 (A72, H11, family F7): lag tax and R at the smallest phi against phi 1.0 at the moderate tier, coupled mode."""
    q3 = load_queues({"q3": folders["q3"]})["q3"]
    table = c9_table(q3, cfg["migration_steps_m0"], cfg["headroom"], cfg["h_net"], cfg["phi"], min(cfg["phi_levels"]),
                     cfg["min_lag_tax_share"], cfg["alpha_adj"], cfg["min_events"], cfg["n_boot"], cfg["ci_level"],
                     cfg["master_seed"])
    write_table(out, "c9_h11.csv", table)
    return table


def stage_c10_h12_p4a(cfg, folders, out):
    """C10 (A72, H12, family F8): P4a against P4 on wasted moves and damage non-inferiority, plus the rerun consistency table."""
    queues = load_queues({name: folders[name] for name in ("must", "q1", "q11_rv0")})
    table = c10_table(queues["must"], queues["q1"], cfg["migration_steps_m0"], cfg["headroom"], cfg["margin_quantile"],
                      cfg["h12_noninf"], cfg["alpha_adj"], cfg["min_events"], cfg["n_boot"], cfg["ci_level"],
                      cfg["master_seed"])
    write_table(out, "c10_h12.csv", table)
    check = rerun_consistency(queues["must"], queues["q1"], queues["q11_rv0"], cfg["migration_steps_m0"],
                              cfg["headroom"], cfg["margin_quantile"])
    write_table(out, "c10_consistency.csv", check)
    return table


def stage_cl1236_layers(cfg, folders, out):
    """CL1, CL2, CL6 (mean per setting) and CL3 (attribution) from the Q3 rows (A74)."""
    q3 = load_queues({"q3": folders["q3"]})["q3"]
    layers = layers_table(q3, cfg["migration_steps_m0"], cfg["headroom"], cfg["n_boot"], cfg["ci_level"],
                          cfg["master_seed"])
    Path(out).mkdir(parents=True, exist_ok=True)
    layers.to_csv(Path(out) / "cl_layers.csv", index=False)
    show = layers[(layers["mode"] == "coupled") & (layers["phi"] == cfg["phi"])]
    print(show.pivot_table(index=["dataset", "variant", "h_net"], columns="metric", values="mean").to_string())
    print("cl_layers.csv rows %d" % len(layers))
    table = attribution_table(q3, cfg["migration_steps_m0"], cfg["headroom"], cfg["n_boot"], cfg["ci_level"],
                              cfg["master_seed"])
    write_table(out, "cl3_attribution.csv", table)
    return table


def stage_cl4_interference(cfg, folders, out):
    """CL4 (A74): coupled wait over the replay wait with migration traffic off, per coupled setting."""
    q3 = load_queues({"q3": folders["q3"]})["q3"]
    table = interference_table(q3, cfg["migration_steps_m0"], cfg["headroom"], cfg["n_boot"], cfg["ci_level"],
                               cfg["master_seed"])
    write_table(out, "cl4_interference.csv", table)
    return table


def stage_cl5_coupling(cfg, folders, out):
    """CL5 (A74): Spearman correlation of CPU and network overload node-steps across events, coupled mode at the default phi."""
    q3 = load_queues({"q3": folders["q3"]})["q3"]
    table = coupling_table(q3, cfg["migration_steps_m0"], cfg["headroom"], cfg["phi"], cfg["n_boot"], cfg["ci_level"],
                           cfg["master_seed"])
    write_table(out, "cl5_coupling.csv", table)
    return table


def stage_cl9_joint(cfg, folders, out):
    """CL9 (A74): joint cost over the weight simplex and the Pareto front, coupled mode at the default phi."""
    q3 = load_queues({"q3": folders["q3"]})["q3"]
    terms, weights = joint_tables(q3, cfg["migration_steps_m0"], cfg["headroom"], cfg["phi"], cfg["simplex_step"])
    Path(out).mkdir(parents=True, exist_ok=True)
    weights.to_csv(Path(out) / "cl9_weights.csv", index=False)
    print("cl9_weights.csv rows %d" % len(weights))
    write_table(out, "cl9_terms.csv", terms)
    return terms


def stage_rv0_policies(cfg, folders, out):
    """RV0 (A76): mean per-event damage of the seven policies per dataset (D1 and D2 Must and Q1 rows, D3 from q11_rv0)."""
    queues = load_queues({name: folders[name] for name in ("must", "q1", "q11_rv0")})
    tuned = Path(__file__).resolve().parents[1] / "configs" / "tuned.yaml"
    table = rv0_table(queues["must"], queues["q1"], queues["q11_rv0"], load_headline(tuned), cfg["migration_steps_m0"],
                      cfg["headroom"], cfg["margin_quantile"], cfg["n_boot"], cfg["ci_level"], cfg["master_seed"])
    write_table(out, "rv0_policies.csv", table)
    return table


def stage_rv1_contention(cfg, folders, out):
    """RV1 (A76): contention ablation, mean damage, paired difference to the full row and recovery fraction."""
    rv123 = load_queues({"q11_rv123": folders["q11_rv123"]})["q11_rv123"]
    table = rv1_table(rv123, cfg["migration_steps_m0"], cfg["headroom"], cfg["n_boot"], cfg["ci_level"],
                      cfg["master_seed"])
    write_table(out, "rv1_contention.csv", table)
    return table


def stage_rv23_placement_budget(cfg, folders, out):
    """RV2 and RV3 (A76): skewed against balanced placement and throttle budget 1 against the default."""
    queues = load_queues({name: folders[name] for name in ("must", "q1", "q11_rv123")})
    rv2, rv3 = rv23_tables(queues["must"], queues["q1"], queues["q11_rv123"], cfg["migration_steps_m0"],
                           cfg["headroom"], cfg["margin_quantile"], cfg["n_boot"], cfg["ci_level"], cfg["master_seed"])
    write_table(out, "rv2_placement.csv", rv2)
    write_table(out, "rv3_budget.csv", rv3)
    return rv2


def stage_rv4_logging(cfg, folders, out):
    """RV4 (A76): logged point error, interval coverage and width, adaptive alpha and trigger decisions of P4 and P4a."""
    rv0 = load_queues({"q11_rv0": folders["q11_rv0"]})["q11_rv0"]
    table = rv4_table(rv0, cfg["migration_steps_m0"], cfg["headroom"], cfg["n_boot"], cfg["ci_level"],
                      cfg["master_seed"])
    write_table(out, "rv4_logging.csv", table)
    return table


def stage_aa_ablations(cfg, folders, out):
    """AA1 to AA11 (A78, family C12 per ablation): each ablation level against the default level, paired by event."""
    queues = load_queues({name: folders[name] for name in ("must", "q1", "q3", "q4", "q6")})
    tuned = Path(__file__).resolve().parents[1] / "configs" / "tuned.yaml"
    table = ablation_table(queues["must"], queues["q1"], queues["q3"], queues["q4"], queues["q6"], load_headline(tuned),
                           cfg["migration_steps_m0"], cfg["headroom"], cfg["margin_quantile"], cfg["h_net"], cfg["phi"],
                           cfg["alpha_adj"], cfg["n_boot"], cfg["ci_level"], cfg["master_seed"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "aa_ablations.csv", index=False)
    return table


def read_recorded(name, out):
    """A recorded table from the output folder, else from the repository tables folder (inputs and scratch runs)."""
    root = Path(__file__).resolve().parents[1]
    for base in (Path(out), root / "tables"):
        if (base / name).exists():
            return pd.read_csv(base / name)
    raise FileNotFoundError(name)


def stage_xa1_transfer_matrix(cfg, folders, out):
    """XA1 (A80): fixed-protocol transfer matrix against the in-dataset cell, from tables/forecast_cross.csv."""
    cross = read_recorded("forecast_cross.csv", Path(__file__).resolve().parents[1] / "tables")
    table = xa1_table(cross)
    write_table(out, "xa1_transfer_matrix.csv", table)
    return table


def stage_xa5_rolling_origin(cfg, folders, out):
    """XA5 (A80): accuracy by evaluation day from the day groups of tables/forecast_cross.csv."""
    cross = read_recorded("forecast_cross.csv", Path(__file__).resolve().parents[1] / "tables")
    table = xa5_table(cross)
    write_table(out, "xa5_rolling_origin.csv", table)
    return table


def stage_xa7_event_accuracy(cfg, folders, out):
    """XA7 (A80): MASE overall, inside and outside event windows, and early warning per forecaster."""
    root = Path(__file__).resolve().parents[1] / "tables"
    cross = read_recorded("forecast_cross.csv", root)
    chronos = read_recorded("forecast_accuracy_chronos.csv", root)
    main = read_recorded("forecast_accuracy_early_warning.csv", root)
    base = read_recorded("forecast_accuracy_baselines_early_warning.csv", root)
    warn_chronos = read_recorded("forecast_accuracy_chronos_early_warning.csv", root)
    write_table(out, "xa7_event_accuracy.csv", xa7_mase_table(cross, chronos))
    print("early-warning rows present in both main and baseline tables but different: %d" % baseline_overlap_mismatch(main, base))
    table = xa7_warning_table(main, base, warn_chronos)
    write_table(out, "xa7_early_warning.csv", table)
    return table


def stage_ledger(cfg, folders, out):
    """Hypothesis ledger (A80): one row per pre-registered comparison, labels and counts read from the recorded tables."""
    table = ledger_table(cfg, {name: read_recorded(name + ".csv", out) for name in LEDGER_TABLES})
    write_table(out, "ledger.csv", table)
    return table


def stage_ta_tables(cfg, folders, out):
    """TA3 to TA9 (A80), assembled from recorded tables; no new statistic."""
    root = Path(__file__).resolve().parents[1] / "tables"
    got = lambda name: read_recorded(name + ".csv", out)
    write_table(out, "ta3_forecast_accuracy.csv", ta3(read_recorded("forecast_accuracy.csv", root),
                                                      read_recorded("forecast_accuracy_baselines.csv", root),
                                                      read_recorded("forecast_accuracy_chronos.csv", root)))
    write_table(out, "ta4_main_results.csv", ta4(got("c11_main_comparison"), got("c13_p4a_pairs")))
    write_table(out, "ta5_regime_map.csv", ta5(got("regime_map_q2"), got("regime_map_must")))
    write_table(out, "ta6_ablations.csv", ta6(got("aa_ablations")))
    write_table(out, "ta7_cross_data.csv", ta7(got("c5_transfer_summary"), got("c6_ranking_tau"), got("xa1_transfer_matrix")))
    parts = [("C7", got("c7_h9")), ("C8", got("c8_h10")), ("C9", got("c9_h11")), ("CL3", got("cl3_attribution")),
             ("CL4", got("cl4_interference")), ("RV1", got("rv1_contention"))]
    write_table(out, "ta8_cross_layer.csv", ta8(parts, got("cl3_attribution"), "coupled", cfg["phi"]))
    table = ta9(got("ledger"))
    write_table(out, "ta9_statistics.csv", table)
    return table


def stage_rt_review_tables(cfg, folders, out):
    """RT (A89, descriptive, post-freeze): damage, slo_violation_frac, peak_backlog_steps and moves per dataset and policy with the ratio to P1, and the same metrics of the RV1 rows."""
    queues = load_queues({name: folders[name] for name in ("must", "q1", "q11_rv0", "q11_rv123")})
    tuned = Path(__file__).resolve().parents[1] / "configs" / "tuned.yaml"
    policy = policy_metrics_table(queues["must"], queues["q1"], queues["q11_rv0"], load_headline(tuned),
                                  cfg["migration_steps_m0"], cfg["headroom"], cfg["margin_quantile"], cfg["n_boot"],
                                  cfg["ci_level"], cfg["master_seed"])
    contention = rv1_metrics_table(queues["q11_rv123"], cfg["migration_steps_m0"], cfg["headroom"], cfg["n_boot"],
                                   cfg["ci_level"], cfg["master_seed"])
    write_table(out, "rt_policy_metrics.csv", policy)
    write_table(out, "rt_rv1_metrics.csv", contention)
    return policy


STAGES = {"interaction_trigger": stage_interaction_trigger, "c1_lag_tax": stage_c1_lag_tax,
          "c2_recovery_class": stage_c2_recovery_class, "c3_cheap_migration": stage_c3_cheap_migration,
          "regime_map_q2": stage_regime_map_q2, "regime_map_must": stage_regime_map_must,
          "c11_main_comparison": stage_c11_main_comparison, "c13_p4a_pairs": stage_c13_p4a_pairs,
          "c4_accuracy_benefit": stage_c4_accuracy_benefit, "c5_transfer": stage_c5_transfer,
          "c6_ranking": stage_c6_ranking,
          "c7_h9_share": stage_c7_h9_share, "c8_h10_coupling": stage_c8_h10_coupling,
          "c9_h11_disaggregated": stage_c9_h11_disaggregated, "c10_h12_p4a": stage_c10_h12_p4a,
          "cl1236_layers": stage_cl1236_layers, "cl4_interference": stage_cl4_interference,
          "cl5_coupling": stage_cl5_coupling, "cl9_joint": stage_cl9_joint,
          "rv0_policies": stage_rv0_policies, "rv1_contention": stage_rv1_contention,
          "rv23_placement_budget": stage_rv23_placement_budget, "rv4_logging": stage_rv4_logging,
          "aa_ablations": stage_aa_ablations,
          "xa1_transfer_matrix": stage_xa1_transfer_matrix,
          "xa5_rolling_origin": stage_xa5_rolling_origin,
          "xa7_event_accuracy": stage_xa7_event_accuracy,
          "ledger": stage_ledger,
          "ta_tables": stage_ta_tables,
          "rt_review_tables": stage_rt_review_tables}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stage", required=True, choices=list(STAGES) + ["all"])
    args = parser.parse_args()
    cfg = load_config(args.config)
    folders = resolve_folders(cfg["result_folders"], args.results, "must")
    for name in (list(STAGES) if args.stage == "all" else [args.stage]):
        STAGES[name](cfg, folders, args.out)


if __name__ == "__main__":
    main()
