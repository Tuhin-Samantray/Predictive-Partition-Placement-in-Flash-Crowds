"""Q11 runner part 1 (A44): RV0 plan, RV4 logging rows and agreement with the Must grid."""
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from src.config import load_config

ROOT = Path(__file__).parent.parent
SPEC = importlib.util.spec_from_file_location("run_grid_q11", ROOT / "scripts" / "04_run_grid.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)
TRIG = {"trig_decisions", "trig_proposed", "trig_proposed_steps", "trig_no_slot_steps"}
POLICIES = {"P0", "P1", "P2", "P3", "P4", "P4a", "P5"}


def configs():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    tuned = yaml.safe_load((ROOT / "configs" / "tuned.yaml").read_text(encoding="utf-8"))
    ext = yaml.safe_load((ROOT / "configs" / "tuned_ext.yaml").read_text(encoding="utf-8"))
    return cfg, tuned, ext


def run_one(task, out):
    cfg, tuned, ext = configs()
    MOD.init(cfg, tuned, str(out), ext, None)
    MOD.run_task(task)
    rows = []
    for path in sorted(Path(out).glob("grid_*.jsonl")):
        rows += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return rows


def find_must_row(cell, replicate):
    needle = '"cell": "%s"' % cell
    for path in sorted((ROOT / "results").glob("grid_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if needle in line:
                row = json.loads(line)
                if row["cell"] == cell and row["replicate"] == replicate:
                    return row
    return None


def test_plan_structure():
    cfg, tuned, ext = configs()
    tasks = MOD.plan_q11(cfg, tuned, False)
    d3 = [t for t in tasks if "base" in t]
    plain = [t for t in tasks if "base" not in t]
    assert {t["dataset"] for t in plain} == set(MOD.DATASETS)
    assert {t["dataset"] for t in d3} == {b + "_" + cfg["inject_label"] for b in MOD.DATASETS}
    assert all(t["queue"] == "q11" and t["split"] == "eval" for t in tasks)
    assert all(t["m"] == cfg["migration_steps_m0"] and t["headroom"] == cfg["headroom"] for t in tasks)
    for t in plain:
        assert t["variants"] == [MOD.q11_label("P4"), MOD.q11_label("P4a")]
    for t in d3:
        resolved = [MOD.resolve_spec(v, t["specs"].get(v), tuned, ext, t["base"]) for v in t["variants"]]
        assert len(resolved) == len(POLICIES) and {r[0] for r in resolved} == POLICIES
        forecasters = {r[0]: r[2] for r in resolved}
        assert forecasters["P3"] == tuned["tuned"][t["base"]]["P4"]["forecaster"]
        assert forecasters["P4a"] == ext["tuned_ext"][t["base"]]["P4a"]["forecaster"]
        assert {r[0] for r in resolved if r[3].get("log")} == {"P4", "P4a"}
    cells = {}
    for t in tasks:
        cells.setdefault((t["dataset"], t["cls"]), []).append(t["j"])
    assert all(sorted(js) == list(range(cfg["replicates_per_cell"])) for js in cells.values())


def test_pilot_plan():
    cfg, tuned, ext = configs()
    tasks = MOD.plan_q11(cfg, tuned, True)
    assert all("base" not in t and t["split"] == "tune" and t["cls"] == "all" for t in tasks)
    assert len(tasks) == len(MOD.DATASETS) * cfg["pilot_replicates"]
    assert all(t["variants"] == [MOD.q11_label("P4"), MOD.q11_label("P4a")] for t in tasks)


def test_resolve_spec_accepts_log():
    cfg, tuned, ext = configs()
    label = MOD.q11_label("P4")
    name, params, forecaster, kwargs = MOD.resolve_spec(label, MOD.q11_specs()[label], tuned, ext, "D1")
    assert name == "P4" and kwargs == {"log": True} and "log" not in params
    assert forecaster == tuned["tuned"]["D1"]["P4"]["forecaster"]


def test_d3_task_rows(tmp_path):
    cfg, tuned, ext = configs()
    task = next(t for t in MOD.plan_q11(cfg, tuned, False) if "base" in t)
    rows = run_one(task, tmp_path)
    expected = {"P0", "P1", "P2", "P3_" + tuned["tuned"][task["base"]]["P4"]["forecaster"], "P4", "P4a", "P5"}
    assert len(rows) == len(POLICIES) and {r["variant"] for r in rows} == expected
    for r in rows:
        assert r["queue"] == "q11" and r["dataset"].endswith("_" + cfg["inject_label"])
        logged = r["variant"] in ("P4", "P4a")
        assert any(k.startswith("rv4_") for k in r) == logged
        assert (TRIG <= set(r)) == logged
        if not logged:
            assert not any(k.startswith("trig_") for k in r)
    assert "rv4_ad_cov" in next(r for r in rows if r["variant"] == "P4a")


def test_p4_rows_match_must_grid(tmp_path):
    cfg, tuned, ext = configs()
    task = next(t for t in MOD.plan_q11(cfg, tuned, False) if "base" not in t)
    rows = run_one(task, tmp_path)
    row = next(r for r in rows if r["variant"] == "P4")
    cell = MOD.cell_id(task["dataset"], task["cls"], task["m"], task["headroom"], "P4")
    ref = find_must_row(cell, task["j"])
    if ref is None:
        pytest.skip("Must grid row not available")
    for key in ("damage", "moves", "wasted_moves", "dropped_moves", "migrated_load", "slo_violation_frac"):
        assert row[key] == ref[key]
