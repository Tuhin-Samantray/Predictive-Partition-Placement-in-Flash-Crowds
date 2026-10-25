import importlib.util
import json
import os
from pathlib import Path

import yaml

from src.config import load_config

spec = importlib.util.spec_from_file_location("run_grid", Path(__file__).resolve().parents[1] / "scripts" / "04_run_grid.py")
rg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rg)


def test_plan_q2_counts_and_events():
    cfg = load_config("configs/default.yaml")
    tasks = rg.plan_q2(cfg)
    n_cls = cfg["rise_buckets"]
    assert len(tasks) == 2 * n_cls * cfg["replicates_per_cell"] * len(cfg["m_levels"])
    assert sum(len(t["variants"]) for t in tasks) == 3 * len(tasks)
    for base in ("D1", "D2"):
        sub = [t for t in tasks if t["base"] == base]
        assert {t["variants"][1] for t in sub} == {cfg["q2_forecast_variants"][base]}
        for cls in {t["cls"] for t in sub}:
            ids = {t["event_id"] for t in sub if t["cls"] == cls}
            assert len(ids) == cfg["injection_events"] // n_cls
    assert all(t["dataset"].endswith("_" + cfg["inject_label"]) and t["queue"] == "q2" for t in tasks)


def test_q2_run_task_rows(tmp_path):
    cfg = load_config("configs/default.yaml")
    tuned = yaml.safe_load(Path("configs/tuned.yaml").read_text(encoding="utf-8"))
    for base in ("D1", "D2"):
        task = [t for t in rg.plan_q2(cfg) if t["base"] == base][0]
        rg.init(cfg, tuned, str(tmp_path / base), None)
        rg.run_task(task)
        rows = [json.loads(line) for line in (tmp_path / base / ("grid_%d.jsonl" % os.getpid())).read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 3 and all(r["conservation_ok"] and r["queue"] == "q2" for r in rows)
        assert {r["dataset"] for r in rows} == {base + "_" + cfg["inject_label"]}
        assert all(r["event_class"].startswith("rise") for r in rows)
        assert {r["variant"] for r in rows} == {"P1", cfg["q2_forecast_variants"][base], "P5"}
