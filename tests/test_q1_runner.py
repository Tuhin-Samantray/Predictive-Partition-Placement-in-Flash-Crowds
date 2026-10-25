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
    ext = yaml.safe_load(Path("configs/tuned_ext.yaml").read_text(encoding="utf-8"))
    return cfg, tuned, ext


def test_q1_specs_count_and_defaults_skipped():
    cfg, tuned, ext = files()
    for ds in ("D1", "D2"):
        specs = rg.q1_specs(cfg, tuned, ext, ds)
        labels = [rg.q1_label(s["base"], s["factor"], s["param"], s["level"]) for s in specs]
        assert len(specs) == 66 and len(set(labels)) == 66
        assert all(s["level"] != 0.0 for s in specs if s["factor"] == "AA3")
        assert all(s["applied"] == cfg["n_partitions"] for s in specs if s["level"] == rg.UNLIMITED)
        assert not any(s["base"] == "P4" and s["factor"] == "AA6" and s["level"] == cfg["margin_quantile"] for s in specs)
        assert any(s["base"] == "P4a" and s["factor"] == "AA6" and s["level"] == cfg["margin_quantile"] for s in specs)


def test_q1_events_equal_must_events_and_runs_total():
    cfg, tuned, ext = files()
    q1 = rg.plan_q1(cfg, tuned, ext, False)
    must = rg.plan(cfg, False)
    default = {(t["dataset"], t["cls"], t["j"]): t["event_id"] for t in must
               if t["m"] == cfg["migration_steps_m0"] and t["headroom"] == cfg["headroom"]}
    got = {(t["dataset"], t["cls"], t["j"]): t["event_id"] for t in q1}
    assert got == default
    assert sum(len(t["variants"]) for t in q1) == 66 * len(q1)


def test_q1_run_task_rows(tmp_path):
    cfg, tuned, ext = files()
    task = rg.plan_q1(cfg, tuned, ext, False)[0]
    p3 = "P3_" + tuned["tuned"][task["dataset"]]["P4"]["forecaster"]
    wanted = {("AA2", p3, 0), ("AA5", "P1", rg.UNLIMITED), ("AA6", "P4a", 0.95)}
    pick = [lab for lab, s in task["specs"].items() if (s["factor"], s["base"], s["level"]) in wanted]
    assert len(pick) == 3
    rg.init(cfg, tuned, str(tmp_path), ext)
    rg.run_task(dict(task, variants=pick))
    rows = [json.loads(line) for line in (tmp_path / ("grid_%d.jsonl" % os.getpid())).read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3 and len({r["cell"] for r in rows}) == 3
    assert all(r["conservation_ok"] and r["queue"] == "q1" for r in rows)
    assert {r["factor"] for r in rows} == {"AA2", "AA5", "AA6"}
    unl = [r for r in rows if r["factor"] == "AA5"][0]
    assert unl["level"] == rg.UNLIMITED and unl["applied"] == cfg["n_partitions"] and unl["variant"] == "P1"
