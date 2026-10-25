import importlib.util
import json
import os
from pathlib import Path

import yaml

from src.config import load_config

spec = importlib.util.spec_from_file_location("run_grid", Path(__file__).resolve().parents[1] / "scripts" / "04_run_grid.py")
rg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rg)


def files():
    cfg = load_config("configs/default.yaml")
    tuned = yaml.safe_load(Path("configs/tuned.yaml").read_text(encoding="utf-8"))
    layers = yaml.safe_load(Path("configs/tuned_layers.yaml").read_text(encoding="utf-8"))
    return cfg, tuned, layers


def test_q3_specs_count_and_labels():
    cfg, _, _ = files()
    specs = rg.q3_specs(cfg)
    labels = [rg.q3_label(s["base"], s["h_net"], s["mode"], s["phi"]) for s in specs]
    assert len(specs) == len(rg.Q3_POLICIES) * len(cfg["h_net_levels"]) * (len(cfg["phi_levels"]) + 1)
    assert len(set(labels)) == len(specs)
    assert all(s["phi"] is None for s in specs if s["mode"] == rg.CONST_M)
    assert all(s["phi"] in cfg["phi_levels"] for s in specs if s["mode"] == rg.COUPLED)


def test_q3_events_equal_must_events_and_runs_total():
    cfg, _, _ = files()
    q3 = rg.plan_q3(cfg, False)
    must = rg.plan(cfg, False)
    default = {(t["dataset"], t["cls"], t["j"]): t["event_id"] for t in must
               if t["m"] == cfg["migration_steps_m0"] and t["headroom"] == cfg["headroom"]}
    got = {(t["dataset"], t["cls"], t["j"]): t["event_id"] for t in q3}
    assert got == default
    assert sum(len(t["variants"]) for t in q3) == len(rg.q3_specs(cfg)) * len(q3)


def test_q3_pilot_plan():
    cfg, _, _ = files()
    tasks = rg.plan_q3(cfg, True)
    assert len(tasks) == len(rg.DATASETS) * cfg["pilot_replicates"]
    assert all(t["variants"] == [b + ":q3" for b in rg.Q3_POLICIES] for t in tasks)
    assert all(s["mode"] == rg.COUPLED and s["h_net"] == cfg["h_net"] for t in tasks for s in t["specs"].values())


def test_q3_run_task_rows(tmp_path):
    cfg, tuned, layers = files()
    task = rg.plan_q3(cfg, False)[0]
    wanted = {("P6", rg.COUPLED, cfg["phi_levels"][-1]), ("P1", rg.CONST_M, None), ("P3", rg.COUPLED, cfg["phi"])}
    pick = [lab for lab, s in task["specs"].items()
            if (s["base"], s["mode"], s["phi"]) in wanted and s["h_net"] == cfg["h_net_levels"][0]]
    assert len(pick) == 3
    rg.init(cfg, tuned, str(tmp_path), None, layers)
    timing = rg.run_task(dict(task, variants=pick))
    rows = [json.loads(line) for line in (tmp_path / ("grid_%d.jsonl" % os.getpid())).read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3 and len(timing) == 3 and len({r["cell"] for r in rows}) == 3
    assert all(r["conservation_ok"] and r["queue"] == "q3" and "cl_wait_mean" in r for r in rows)
    coupled = [r for r in rows if r["mode"] == rg.COUPLED]
    assert len(coupled) == 2 and all(r["cl_replay_ok"] and "cl_wait_mean_replay" in r for r in coupled)
    const = [r for r in rows if r["mode"] == rg.CONST_M]
    assert len(const) == 1 and const[0]["phi"] is None and "cl_replay_ok" not in const[0]
