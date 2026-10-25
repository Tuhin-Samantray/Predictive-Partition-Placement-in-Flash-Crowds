import importlib.util
import json
import os
from pathlib import Path

import yaml

from src.config import load_config
from src.layer_runs import run_replicate_layers
from src.replicates import forecast_window, load_dataset, make_replicate, run_replicate, select_events

spec = importlib.util.spec_from_file_location("run_grid", Path(__file__).resolve().parents[1] / "scripts" / "04_run_grid.py")
rg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rg)


def files():
    cfg = load_config("configs/default.yaml")
    tuned = yaml.safe_load(Path("configs/tuned.yaml").read_text(encoding="utf-8"))
    ext = yaml.safe_load(Path("configs/tuned_ext.yaml").read_text(encoding="utf-8"))
    layers = yaml.safe_load(Path("configs/tuned_layers.yaml").read_text(encoding="utf-8"))
    return cfg, tuned, ext, layers


def test_q4_specs_count_and_labels():
    cfg, tuned, _, _ = files()
    for ds in ("D1", "D2"):
        specs = rg.q4_specs(cfg, tuned, ds)
        labels = [rg.q4_label(s) for s in specs]
        assert len(specs) == 6 * (len(cfg["kappa_levels"]) - 1) + 2 and len(set(labels)) == len(specs)
        assert all(s["level"] != cfg["migration_overhead_kappa"] for s in specs if s["factor"] == "AA10")
        assert {s["base"] for s in specs if s["factor"] == "AA11"} == {"P3", "P6"}


def test_q4_events_equal_must_events_and_runs_total():
    cfg, tuned, _, _ = files()
    q4 = rg.plan_q4(cfg, tuned, False)
    must = rg.plan(cfg, False)
    default = {(t["dataset"], t["cls"], t["j"]): t["event_id"] for t in must
               if t["m"] == cfg["migration_steps_m0"] and t["headroom"] == cfg["headroom"]}
    assert {(t["dataset"], t["cls"], t["j"]): t["event_id"] for t in q4} == default
    assert sum(len(t["variants"]) for t in q4) == len(rg.q4_specs(cfg, tuned, "D1")) * len(q4)


def test_q4_pilot_plan():
    cfg, tuned, _, _ = files()
    tasks = rg.plan_q4(cfg, tuned, True)
    assert len(tasks) == len(rg.DATASETS) * cfg["pilot_replicates"]
    assert all(t["variants"] == [b + ":q4+AA11:layers=off" for b in rg.Q4_FORECAST] for t in tasks)


def test_q4_run_task_rows(tmp_path):
    cfg, tuned, ext, layers = files()
    task = rg.plan_q4(cfg, tuned, False)[0]
    pick = [lab for lab, s in task["specs"].items()
            if (s["factor"], s["base"], s["level"]) in {("AA10", "P1", 0.5), ("AA11", "P3", "off"), ("AA11", "P6", "off")}]
    assert len(pick) == 3
    rg.init(cfg, tuned, str(tmp_path), ext, layers)
    timing = rg.run_task(dict(task, variants=pick))
    rows = [json.loads(line) for line in (tmp_path / ("grid_%d.jsonl" % os.getpid())).read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3 and len(timing) == 3 and len({r["cell"] for r in rows}) == 3
    assert all(r["conservation_ok"] and r["queue"] == "q4" for r in rows)
    assert {r["factor"] for r in rows} == {"AA10", "AA11"} and {r["variant"] for r in rows} == {"P1", "P3", "P6"}


def test_q4_aa11_off_on_d2_equals_the_layers_off_model(tmp_path):
    cfg, tuned, ext, layers = files()
    task = [t for t in rg.plan_q4(cfg, tuned, False) if t["dataset"] == "D2"][0]
    lab = [k for k, s in task["specs"].items() if s["factor"] == "AA11" and s["base"] == "P3"][0]
    rg.init(cfg, tuned, str(tmp_path), ext, layers)
    rg.run_task(dict(task, variants=[lab]))
    row = json.loads((tmp_path / ("grid_%d.jsonl" % os.getpid())).read_text(encoding="utf-8").splitlines()[0])
    data, by_id = rg.dataset_of("D2")
    rep = make_replicate(cfg, data, by_id[task["event_id"]], task["j"], task["headroom"])
    params = dict(layers["tuned_layers"]["D2"]["P3"])
    fc = params.pop("forecaster")
    ref = run_replicate(cfg, rep, "P3", params, m=task["m"], forecast=forecast_window(rg.table_of("D2", fc), rep))
    assert row["damage"] == ref["damage"] and row["moves"] == ref["moves"]


def test_kappa_override_default_is_identical():
    cfg, _, _, _ = files()
    data = load_dataset(cfg, "D2")
    rep = make_replicate(cfg, data, select_events(cfg, data, "tune", 1)[0], 0, cfg["headroom"])
    p1 = {"ewma_alpha": 0.3, "theta_hi": 0.9, "cooldown": 5}
    assert run_replicate(cfg, rep, "P1", p1) == run_replicate(cfg, rep, "P1", p1, kappa=cfg["migration_overhead_kappa"])
