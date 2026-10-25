"""Q11 part 2 (A45): skewed placement, kappa keyword, RV1 to RV3 plan and rows."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from src.config import load_config
from src.layer_runs import CrossData, run_replicate_layers
from src.layers import link_capacity, partition_sizes
from src.replicates import (Replicate, load_dataset, make_replicate, make_replicate_skewed, select_events,
                            skewed_placement)
from src.simulator import node_capacity

ROOT = Path(__file__).parent.parent
SPEC = importlib.util.spec_from_file_location("run_grid_q11b", ROOT / "scripts" / "04_run_grid.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def configs():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    tuned = yaml.safe_load((ROOT / "configs" / "tuned.yaml").read_text(encoding="utf-8"))
    ext = yaml.safe_load((ROOT / "configs" / "tuned_ext.yaml").read_text(encoding="utf-8"))
    layers = yaml.safe_load((ROOT / "configs" / "tuned_layers.yaml").read_text(encoding="utf-8"))
    return cfg, tuned, ext, layers


def test_skewed_placement_definition():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    n_part, n_nodes = cfg["n_partitions"], cfg["n_nodes"]
    hot_nodes = cfg["skew_hot_nodes"]
    n_hot = int(cfg["skew_hot_share"] * n_part)
    rng = np.random.default_rng(3)
    data = SimpleNamespace(load=rng.gamma(2.0, 1.0, (40, n_part)) * rng.uniform(0.1, 5.0, n_part), train_end=30,
                           name="S")
    label = (cfg["master_seed"], "S", "e1", 0)
    node0 = skewed_placement(cfg, data, label)
    order = np.argsort(-data.load[:30].mean(axis=0), kind="stable")
    hot = order[:n_hot]
    assert (node0[hot] < hot_nodes).all()
    assert (np.bincount(node0[hot], minlength=n_nodes)[:hot_nodes] == n_hot // hot_nodes).all()
    assert node0[hot[0]] == 0 and node0[hot[1]] == 1
    assert (node0[order[n_hot:]] >= hot_nodes).all() and node0.max() < n_nodes
    assert (skewed_placement(cfg, data, label) == node0).all()
    other = skewed_placement(cfg, data, (cfg["master_seed"], "S", "e1", 1))
    assert (other[hot] == node0[hot]).all() and not (other == node0).all()


def test_make_replicate_skewed_changes_only_placement():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    data = load_dataset(cfg, "D1")
    ev = select_events(cfg, data, "tune", 1)[0]
    plain = make_replicate(cfg, data, ev, 0, cfg["headroom"])
    skew = make_replicate_skewed(cfg, data, ev, 0, cfg["headroom"])
    assert skew.start == plain.start and (skew.load == plain.load).all() and (skew.mu == plain.mu).all()
    assert not (skew.node0 == plain.node0).all()
    assert (np.bincount(skew.node0, minlength=cfg["n_nodes"]) == cfg["n_partitions"] // cfg["n_nodes"]).all()


def test_layers_kappa_keyword():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    steps = 120
    n_nodes = cfg["n_nodes"]
    n_part = 4 * n_nodes
    rng = np.random.default_rng(7)
    node0 = np.arange(n_part) % n_nodes
    load = rng.gamma(2.0, 1.0, (steps, n_part)) * (1 + 3 * (node0 == 0))
    mu = node_capacity(load, cfg["headroom"], n_nodes)
    rep = Replicate("ev", "D1", "tune", "x", 0, 0, steps - 1, 0, load, node0, mu)
    nbytes = np.random.default_rng(11).gamma(2.0, 1.0, (steps, n_part))
    link = link_capacity(nbytes, cfg["h_net"], n_nodes)
    sizes = partition_sizes(n_part, cfg["migration_steps_m0"], cfg["rho"], link, cfg["sigma_size"], 1)
    cross = CrossData(nbytes, np.ones(n_part), link, sizes, cfg["h_net"])
    params = {"ewma_alpha": 0.3, "theta_hi": 0.8, "cooldown": 1}
    default, res_d = run_replicate_layers(cfg, rep, cross, "P1", params, return_result=True)
    same, _ = run_replicate_layers(cfg, rep, cross, "P1", params, kappa=cfg["migration_overhead_kappa"],
                                   return_result=True)
    assert same == default
    zero, res_z = run_replicate_layers(cfg, rep, cross, "P1", params, kappa=0, return_result=True)
    assert default["moves"] > 0
    assert abs(res_z.arrivals.sum() - load.sum()) < 1e-6 * load.sum()
    assert res_d.arrivals.sum() > res_z.arrivals.sum()


def test_plan_q11b():
    cfg, tuned, ext, layers = configs()
    specs = MOD.q11b_specs(cfg)
    factors = [s["factor"] for s in specs.values()]
    assert len(specs) == 23 and [factors.count(f) for f in ("RV1", "RV2", "RV3")] == [16, 3, 4]
    assert cfg["throttle_budget_B"] < cfg["migration_budget_B"]
    assert {s["base"] for s in specs.values() if s["factor"] == "RV1"} == {"P1", "P3", "P4a", "P5"}
    assert {s["base"] for s in specs.values() if s["factor"] == "RV2"} == {"P0", "P1", "P4a"}
    assert all(s["budget"] == cfg["throttle_budget_B"] for s in specs.values() if s["factor"] == "RV3")
    tasks = MOD.plan_q11b(cfg, tuned, False)
    assert {t["dataset"] for t in tasks} == set(MOD.DATASETS)
    assert all(t["queue"] == "q11b" and t["split"] == "eval" and t["variants"] == list(specs) for t in tasks)
    cells = {}
    for t in tasks:
        cells.setdefault((t["dataset"], t["cls"]), []).append(t["j"])
    assert all(sorted(js) == list(range(cfg["replicates_per_cell"])) for js in cells.values())
    assert len(tasks) * len(specs) == len(cells) * cfg["replicates_per_cell"] * len(specs)
    pilot = MOD.plan_q11b(cfg, tuned, True)
    assert len(pilot) == len(MOD.DATASETS) * cfg["pilot_replicates"]
    assert all(t["split"] == "tune" and t["cls"] == "all" and t["variants"] == list(specs) for t in pilot)


def test_task_rows(tmp_path):
    cfg, tuned, ext, layers = configs()
    pick = ["P4a:q11b-rv1+nic_off", "P1:q11b-rv1+both_off", "P0:q11b-rv2+skewed", "P1:q11b-rv3+B1-skewed",
            "P4a:q11b-rv3+B1-balanced"]
    task = dict(MOD.plan_q11b(cfg, tuned, False)[0])
    assert all(v in task["variants"] for v in pick)
    task["variants"] = pick
    MOD.init(cfg, tuned, str(tmp_path), ext, layers)
    MOD.run_task(task)
    rows = []
    for path in sorted(Path(tmp_path).glob("grid_*.jsonl")):
        rows += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    by = {v: next(r for r in rows if r["cell"].endswith("-" + v)) for v in pick}
    assert len(rows) == len(pick) == len({r["cell"] for r in rows}) and set(by) == set(pick)
    nic_off, both_off = by[pick[0]], by[pick[1]]
    assert nic_off["mode"] == "const_m" and nic_off["cl_bytes_sent"] == 0 and nic_off["cl_mig_share"] == 0
    assert nic_off["kappa_applied"] == cfg["migration_overhead_kappa"] and both_off["kappa_applied"] == 0
    assert "mig_duration_mean" in both_off and "mig_censored" in both_off
    assert any(k.startswith("rv4_") for k in nic_off) and not any(k.startswith("rv4_") for k in both_off)
    skew0 = by[pick[2]]
    assert skew0["moves"] == 0 and skew0["placement"] == "skewed" and skew0["factor"] == "RV2"
    p1_b1 = by[pick[3]]
    assert p1_b1["budget"] == cfg["throttle_budget_B"] and p1_b1["level"] == cfg["throttle_budget_B"]
    p4a_b1 = by[pick[4]]
    assert p4a_b1["placement"] == "balanced" and "rv4_ad_cov" in p4a_b1 and "trig_decisions" in p4a_b1
    assert all(r["queue"] == "q11b" and r["conservation_ok"] for r in rows)
