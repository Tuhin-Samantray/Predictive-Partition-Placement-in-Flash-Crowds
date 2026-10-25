import importlib.util
import json
import os
from pathlib import Path

import yaml

from src.config import load_config
from src.replicates import make_replicate, run_replicate

spec = importlib.util.spec_from_file_location("run_grid", Path(__file__).resolve().parents[1] / "scripts" / "04_run_grid.py")
rg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rg)


def files():
    cfg = load_config("configs/default.yaml")
    tuned = yaml.safe_load(Path("configs/tuned.yaml").read_text(encoding="utf-8"))
    ext = yaml.safe_load(Path("configs/tuned_ext.yaml").read_text(encoding="utf-8"))
    return cfg, tuned, ext


def test_q8_specs_cover_every_policy_and_both_sources():
    cfg, tuned, ext = files()
    for ds in rg.DATASETS:
        specs = rg.q8_specs(tuned, ds)
        labels = [rg.q1_label(s["base"], s["factor"], s["param"], s["level"]) for s in specs]
        assert len(specs) == 12 and len(set(labels)) == 12
        assert {s["base"] for s in specs} == set(rg.q1_base_variants(tuned, ds))
        assert {s["level"] for s in specs} == set(rg.DATASETS)
        assert all(s["applied"] == s["level"] and s["factor"] == "XA2" for s in specs)


def test_own_source_keeps_the_tuned_parameters_and_foreign_source_takes_the_donor_values():
    cfg, tuned, ext = files()
    differs = 0
    for ds in rg.DATASETS:
        other = [d for d in rg.DATASETS if d != ds][0]
        for base in rg.q1_base_variants(tuned, ds):
            own = rg.resolve_spec(base, None, tuned, ext, ds)
            for source in (ds, other):
                s = {"queue": "q8", "base": base, "factor": "XA2", "param": "source", "level": source, "applied": source}
                name, params, forecaster, kwargs = rg.resolve_spec(base, s, tuned, ext, ds)
                assert (name, forecaster, kwargs) == (own[0], own[2], {})
                donor = rg.resolve_spec(base, None, tuned, ext, source)[1]
                for key, value in donor.items():
                    if key != "forecaster":
                        assert params[key] == value
                if source == ds:
                    assert params == own[1]
                elif params != own[1]:
                    differs += 1
    assert differs > 0


def test_q8_events_equal_must_events_and_run_total():
    cfg, tuned, ext = files()
    q8 = rg.plan_q8(cfg, tuned)
    must = rg.plan(cfg, False)
    default = {(t["dataset"], t["cls"], t["j"]): t["event_id"] for t in must
               if t["m"] == cfg["migration_steps_m0"] and t["headroom"] == cfg["headroom"]}
    got = {(t["dataset"], t["cls"], t["j"]): t["event_id"] for t in q8}
    assert got == default
    assert len(q8) == 120 and sum(len(t["variants"]) for t in q8) == 1440


def test_q8_run_task_rows_and_control_equals_a_direct_run(tmp_path):
    cfg, tuned, ext = files()
    task = rg.plan_q8(cfg, tuned)[0]
    ds = task["dataset"]
    other = [d for d in rg.DATASETS if d != ds][0]
    pick = ["P1+XA2:source=%s" % ds, "P1+XA2:source=%s" % other, "P4a+XA2:source=%s" % other,
            "P5+XA2:source=%s" % other]
    assert all(v in task["variants"] for v in pick)
    rg.init(cfg, tuned, str(tmp_path), ext, None)
    rg.run_task(dict(task, variants=pick))
    lines = (tmp_path / ("grid_%d.jsonl" % os.getpid())).read_text(encoding="utf-8").splitlines()
    rows = {json.loads(line)["cell"]: json.loads(line) for line in lines}
    assert len(rows) == 4
    assert all(r["conservation_ok"] and r["queue"] == "q8" and r["factor"] == "XA2" and r["param"] == "source"
               for r in rows.values())
    data, by_id = rg.dataset_of(ds)
    rep = make_replicate(cfg, data, by_id[task["event_id"]], task["j"], task["headroom"])
    direct = run_replicate(cfg, rep, "P1", dict(tuned["tuned"][ds]["P1"]), m=task["m"])
    own = [r for r in rows.values() if r["variant"] == "P1" and r["level"] == ds][0]
    assert own["damage"] == direct["damage"]
