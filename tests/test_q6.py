import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import yaml

import src.replicates as rp
from src.config import load_config
from src.forecasters import naive_table
from src.replicates import Replicate, run_replicate

spec = importlib.util.spec_from_file_location("run_grid", Path(__file__).resolve().parents[1] / "scripts" / "04_run_grid.py")
rg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rg)


def files():
    cfg = load_config("configs/default.yaml")
    tuned = yaml.safe_load(Path("configs/tuned.yaml").read_text(encoding="utf-8"))
    return cfg, tuned


def world(cfg):
    rng = np.random.default_rng(1)
    n_part, n_nodes, steps = cfg["n_partitions"], cfg["n_nodes"], 60
    load = rng.gamma(2.0, 1.0, size=(steps, n_part))
    node0 = rng.permutation(np.arange(n_part) % n_nodes)
    mu = np.full(n_nodes, load.sum(axis=1).mean() / n_nodes * cfg["headroom"])
    return Replicate("e", "DX", "tune", "c", 0, 10, 20, 0, load, node0, mu)


def test_q6_specs_count_and_defaults_skipped():
    cfg, tuned = files()
    for ds in ("D1", "D2"):
        specs = rg.q6_specs(cfg, tuned, ds)
        labels = [rg.q1_label(s["base"], s["factor"], s["param"], s["level"]) for s in specs]
        assert len(specs) == 10 and len(set(labels)) == 10
        assert all(s["level"] != (cfg["monitor_delay_d"] if s["factor"] == "AA7" else cfg["decision_interval_k"]) for s in specs)
        assert not any(s["base"] == "P5" and s["factor"] == "AA7" for s in specs)
        assert {s["base"] for s in specs} == {"P1", "P3_" + tuned["tuned"][ds]["P4"]["forecaster"], "P5"}


def test_q6_events_equal_must_events_and_run_total():
    cfg, tuned = files()
    q6 = rg.plan_q6(cfg, tuned)
    must = rg.plan(cfg, False)
    default = {(t["dataset"], t["cls"], t["j"]): t["event_id"] for t in must
               if t["m"] == cfg["migration_steps_m0"] and t["headroom"] == cfg["headroom"]}
    got = {(t["dataset"], t["cls"], t["j"]): t["event_id"] for t in q6}
    assert got == default
    assert len(q6) == 120 and sum(len(t["variants"]) for t in q6) == 1200


def test_d_and_k_overrides_default_identical_and_reach_simulator_and_ctx(monkeypatch):
    cfg, _ = files()
    rep = world(cfg)
    params = {"ewma_alpha": 0.3, "theta_hi": 0.9, "cooldown": 0}
    base = run_replicate(cfg, rep, "P1", params)
    assert base == run_replicate(cfg, rep, "P1", params, d=cfg["monitor_delay_d"], k=cfg["decision_interval_k"])
    seen = {}
    real_sim, real_build = rp.simulate, rp.build_policy

    def spy_sim(*args, **kwargs):
        seen["d_sim"], seen["k_sim"] = kwargs["d"], kwargs["k"]
        return real_sim(*args, **kwargs)

    def spy_build(name, params, mu, ctx=None):
        seen["ctx_d"] = ctx["d"]
        return real_build(name, params, mu, ctx)

    monkeypatch.setattr(rp, "simulate", spy_sim)
    monkeypatch.setattr(rp, "build_policy", spy_build)
    run_replicate(cfg, rep, "P1", params, d=0, k=15)
    assert (seen["d_sim"], seen["k_sim"], seen["ctx_d"]) == (0, 15, 0)
    run_replicate(cfg, rep, "P1", params)
    assert (seen["d_sim"], seen["k_sim"], seen["ctx_d"]) == (cfg["monitor_delay_d"], cfg["decision_interval_k"], cfg["monitor_delay_d"])


def test_extreme_delay_and_interval_run_clean():
    cfg, _ = files()
    rep = world(cfg)
    table = naive_table(rep.load, cfg["horizons"])
    p1 = {"ewma_alpha": 0.3, "theta_hi": 0.9, "cooldown": 0}
    p3 = {"hysteresis": 0.0, "cooldown": 0}
    p5 = {"hysteresis": 0.0, "cooldown": 0, "oracle_window": 5}
    for d in (0, 3):
        assert run_replicate(cfg, rep, "P1", p1, d=d)["conservation_ok"]
        assert run_replicate(cfg, rep, "P3", p3, forecast=table, d=d)["conservation_ok"]
    for k in (5, 15):
        for name, params, fc in (("P1", p1, None), ("P3", p3, table), ("P5", p5, None)):
            assert run_replicate(cfg, rep, name, params, forecast=fc, k=k)["conservation_ok"]


def test_oracle_is_insensitive_to_d():
    cfg, _ = files()
    rep = world(cfg)
    p5 = {"hysteresis": 0.0, "cooldown": 0, "oracle_window": 5}
    assert run_replicate(cfg, rep, "P5", p5, d=0) == run_replicate(cfg, rep, "P5", p5, d=3)


def test_q6_run_task_rows(tmp_path):
    cfg, tuned = files()
    task = rg.plan_q6(cfg, tuned)[0]
    ds = task["dataset"]
    p3 = "P3_" + tuned["tuned"][ds]["P4"]["forecaster"]
    wanted = {("AA7", p3, 0), ("AA8", "P5", 15), ("AA7", "P1", 3)}
    pick = [lab for lab, s in task["specs"].items() if (s["factor"], s["base"], s["level"]) in wanted]
    assert len(pick) == 3
    rg.init(cfg, tuned, str(tmp_path), None, None)
    rg.run_task(dict(task, variants=pick))
    rows = [json.loads(line) for line in (tmp_path / ("grid_%d.jsonl" % os.getpid())).read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3 and len({r["cell"] for r in rows}) == 3
    assert all(r["conservation_ok"] and r["queue"] == "q6" for r in rows)
    assert {(r["factor"], r["param"]) for r in rows} == {("AA7", "d"), ("AA8", "k")}
    assert {r["variant"] for r in rows} == {p3, "P5", "P1"}
