"""Planted tests of the episode figures (A104): the event rule, the frozen-row lookup, the comparison and the drawing."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.replicates import Replicate
from src import report_episode as ep

ROOT = Path(__file__).resolve().parents[1]
CFG = {"migration_steps_m0": 5, "headroom": 1.6, "margin_quantile": 0.9}


def row(ds, eid, j, variant="P1", cls="abrupt", m=5, h=1.6, **extra):
    base = {"dataset": ds, "event_id": eid, "replicate": j, "variant": variant, "onset_class": cls, "m": m,
            "headroom": h, "cell": "%s-%s-m%d-h%s-%s" % (ds, cls, m, h, variant), "damage": 1.5, "moves": 3}
    base.update(extra)
    return base


def events(ids, peaks):
    return pd.DataFrame({"event_id": ids, "peak": peaks})


def test_select_median_of_odd_count_ignores_other_rows():
    ev = events(list("abcde"), [5, 1, 3, 2, 4])
    rows = [row("D1", e, 10 + i) for i, e in enumerate("abcde")]
    rows += [row("D1", "a", 0, variant="P5"), row("D1", "b", 0, m=2), row("D1", "b", 0, cls="ramped"),
             row("D2", "b", 0), row("D1", "b", 0, h=2.0)]
    pick = ep.select_episode(rows, ev, CFG, "D1")
    assert pick == {"event_id": "c", "replicate": 12, "peak": 3.0, "n_events": 5}


def test_select_lower_middle_on_even_count():
    ev = events(list("abcd"), [4, 3, 2, 1])
    rows = [row("D1", e, i) for i, e in enumerate("abcd")]
    assert ep.select_episode(rows, ev, CFG, "D1")["event_id"] == "c"


def test_select_ties_by_event_id_sorted():
    ev = events(["z", "a", "m"], [2, 2, 2])
    rows = [row("D1", "z", 0), row("D1", "a", 1), row("D1", "m", 2)]
    pick = ep.select_episode(rows, ev, CFG, "D1")
    assert pick["event_id"] == "m" and pick["replicate"] == 2


def test_select_refuses_duplicates_and_empty():
    ev = events(["a"], [1])
    with pytest.raises(ValueError):
        ep.select_episode([row("D1", "a", 0), row("D1", "a", 1)], ev, CFG, "D1")
    with pytest.raises(ValueError):
        ep.select_episode([], ev, CFG, "D1")


def test_frozen_row_needs_exactly_one_and_honours_where():
    spec = {"variant": "P4a", "where": {"factor": "AA6", "level": 0.9}}
    rows = [row("D2", "e", 3, variant="P4a", factor="AA6", level=0.9),
            row("D2", "e", 3, variant="P4a", factor="AA6", level=0.8),
            row("D2", "e", 3, variant="P4a", factor="AA3", level=0.9)]
    assert ep.frozen_row(rows, "D2", "e", 3, CFG, spec)["level"] == 0.9
    with pytest.raises(ValueError):
        ep.frozen_row(rows, "D2", "e", 4, CFG, spec)
    with pytest.raises(ValueError):
        ep.frozen_row(rows + [row("D2", "e", 3, variant="P4a", factor="AA6", level=0.9)], "D2", "e", 3, CFG, spec)


def test_compare_row_exact():
    assert ep.compare_row({"damage": 0.1, "moves": 2}, {"damage": 0.1, "moves": 2})
    assert not ep.compare_row({"damage": 0.1, "moves": 2}, {"damage": 0.1 + 2 ** -55, "moves": 2})
    assert not ep.compare_row({"damage": 0.1, "moves": 2}, {"damage": 0.1, "moves": 3})


def test_read_rows_filters_by_event(tmp_path):
    lines = [json.dumps(row("D1", "D1-7", 0)), json.dumps(row("D1", "D1-70", 1)), ""]
    (tmp_path / "grid_1.jsonl").write_text("\n".join(lines), encoding="utf-8")
    assert len(ep.read_rows(tmp_path)) == 2
    only = ep.read_rows(tmp_path, "D1-7")
    assert [r["event_id"] for r in only] == ["D1-7"]


def test_episode_specs_order_and_p4a_filter():
    tuned = {"tuned": {"D1": {"P1": {"cooldown": 1}, "P3_x": {"cooldown": 2}, "P4": {"forecaster": "x"},
                              "P5": {"cooldown": 3}}}}
    ext = {"tuned_ext": {"D1": {"P4a": {"forecaster": "x", "cooldown": 4}}}}
    specs = ep.episode_specs(CFG, tuned, ext, "D1", ["P1", "P3", "P4a", "P5"])
    assert [s["policy"] for s in specs] == ["P1", "P3", "P4a", "P5"]
    assert specs[1]["variant"] == "P3_x" and specs[1]["forecaster"] == "x"
    assert specs[2]["where"] == {"factor": "AA6", "level": 0.9} and specs[2]["params"]["forecaster"] == "x"
    assert specs[0]["forecaster"] is None and specs[3]["where"] == {}


def test_draw_episode_writes_pdf_and_caption(tmp_path):
    cfg = load_config(str(ROOT / "configs" / "default.yaml"))
    style = ep.load_style4(str(ROOT / "configs" / "report_style.json"))
    n_steps, n_part, n_nodes = 40, 6, 3
    rng = np.random.default_rng(0)
    rep = Replicate("D1-9", "D1", "eval", "abrupt", 4, 20, 30, 0, rng.random((n_steps, n_part)),
                    np.arange(n_part) % n_nodes, np.full(n_nodes, 2.0))
    runs, checks = {}, []
    for k, p in enumerate(style["episode_policies"]):
        res = SimpleNamespace(wait=rng.random((n_steps, n_nodes)) * 5, moves=[(2 + k, 0, 0, 1, 7 + k)])
        runs[p] = (res, {"damage": 0.5 + k})
        checks.append({"policy": p, "regenerated_damage": 0.5 + k})
    pick = {"event_id": "D1-9", "replicate": 4, "peak": 12.0, "n_events": 30}
    fc = {"forecaster": "x", "horizon": 5, "total": rng.random(n_steps)}
    path = ep.draw_episode(cfg, "D1", pick, rep, runs, checks, fc, style, str(tmp_path))
    assert Path(path).exists() and Path(path).stat().st_size > 0
    caps = list(tmp_path.glob("fa2_episode_D1*.txt"))
    assert caps and "D1-9" in caps[0].read_text(encoding="utf-8") and "not an average" in caps[0].read_text(encoding="utf-8")
    assert "5 steps" in caps[0].read_text(encoding="utf-8")


def test_forecast_total_sums_partitions_at_one_horizon():
    window = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    assert np.array_equal(ep.forecast_total(window, 1), window[:, 1, :].sum(axis=1))
    assert ep.forecast_total(window, 1).shape == (2,)


def test_horizon_index_is_the_migration_duration_and_refuses_otherwise():
    assert ep.horizon_index({"horizons": [1, 2, 5, 10], "migration_steps_m0": 5}) == 2
    with pytest.raises(ValueError):
        ep.horizon_index({"horizons": [1, 2, 10], "migration_steps_m0": 5})
