import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.accuracy import accuracy_rows
from src.benefit import POOLED, benefit_cells, c4_table, class_event_mase
from src.ranking import c6_tables
from src.stats_tools import INCONCLUSIVE, NOT_SUPPORTED, SUPPORTED
from src.transfer import c5_tables

LEVEL = 0.95
BOOT = 999
SEED = 42
FORECASTERS = ["ridge", "lightgbm"]
HORIZONS = [1, 2, 5, 10]
TRAIN_END, TUNE_END, N_STEPS, N_PART = 20, 35, 80, 7


def _world():
    rng = np.random.default_rng(7)
    load = rng.gamma(2.0, 2.0, size=(N_STEPS, N_PART))
    boost = rng.choice([1.0, 1.0, 1.0, 9.0], size=(N_STEPS, 3, N_PART))
    table = (load[:, None, :] * boost).astype(np.float32)
    cfg = {"horizons": [1, 2, 3], "smape_eps": 0.001, "warn_ratio": 3.0, "warn_baseline_window": 5}
    events = pd.DataFrame({
        "partition": [3, 5, 1, 4, 2, 6], "onset": [22, 26, 45, 55, 62, 70], "end": [24, 27, 47, 58, 63, 72],
        "split": ["tune", "tune", "eval", "eval", "eval", "eval"],
        "onset_class": ["abrupt", "ramped", "abrupt", "ramped", "abrupt", "ramped"]})
    return cfg, load, table, events


def test_class_event_mase_pooled_equals_phase6_and_classes_partition_it():
    cfg, load, table, events = _world()
    for h in cfg["horizons"]:
        base = [r for r in accuracy_rows(cfg, "X", "k", table, load, events, TRAIN_END, TUNE_END)
                if r["split"] == "eval" and r["horizon"] == h][0]
        got = class_event_mase(table, load, events, TRAIN_END, TUNE_END, cfg["horizons"], h, cfg["smape_eps"])
        assert got[POOLED][0] == pytest.approx(base["mase_event"], abs=1e-12)
        assert got[POOLED][1] == base["n_cells_event"]
        assert got["abrupt"][1] + got["ramped"][1] == got[POOLED][1]
        assert got["abrupt"][0] != got["ramped"][0]


def make_must(forecasters, tax_ok=True):
    rows = []
    for ds in ("D1", "D2"):
        for cls in ("abrupt", "ramped"):
            for i in range(6):
                event = "%s-%s-%d" % (ds, cls, i)
                base = dict(dataset=ds, onset_class=cls, m=5, headroom=1.6, replicate=i, event_id=event)
                oracle = 4.0 if (tax_ok or ds != "D2") else 10.0
                rows.append(dict(base, variant="P1", damage=10.0))
                rows.append(dict(base, variant="P5", damage=oracle))
                for f, value in zip(forecasters, (7.0, 10.0, 4.0)):
                    rows.append(dict(base, variant="P3_" + f, damage=value))
    return pd.DataFrame(rows)


def test_benefit_cells_planted_and_undefined_cells():
    names = ["ridge", "lightgbm", "naive"]
    must = make_must(names)
    cells = benefit_cells(must, {"D1": "P1", "D2": "P1"}, names, 5, 1.6, 0.05)
    assert len(cells) == 12 and (cells["n_events_r"] == 6).all()
    got = cells.set_index(["forecaster", "dataset", "onset_class"])["benefit"]
    assert got[("ridge", "D1", "abrupt")] == pytest.approx(0.5)
    assert got[("lightgbm", "D2", "ramped")] == pytest.approx(0.0)
    assert got[("naive", "D1", "ramped")] == pytest.approx(1.0)
    none = benefit_cells(make_must(names, tax_ok=False), {"D1": "P1", "D2": "P1"}, names, 5, 1.6, 0.05)
    d2 = none[none["dataset"] == "D2"]
    assert d2["benefit"].isna().all() and (d2["n_events_r"] == 0).all() and len(d2) == 6


def _c4_inputs(mase, benefit):
    n = len(mase)
    cells = pd.DataFrame({"forecaster": ["f%03d" % i for i in range(n)], "dataset": "D1", "onset_class": "abrupt",
                          "n_events_r": 5, "benefit": benefit})
    return cells, pd.DataFrame({"forecaster": cells["forecaster"], "dataset": "D1", "onset_class": "abrupt",
                                "mase_event_class": mase})


def test_c4_strong_negative_relation_is_not_supported():
    rng = np.random.default_rng(3)
    mase = np.arange(1.0, 21.0)
    cells, mase_frame = _c4_inputs(mase, -mase + rng.normal(0, 2.0, 20))
    detail, summary = c4_table(cells, mase_frame, 0.5, 0.7, BOOT, LEVEL, SEED)
    row = summary.iloc[0]
    assert row["rho"] < -0.5 and row["ci_high"] < 0 and row["h4_label"] == NOT_SUPPORTED
    assert row["n_cells_used"] == 20 and len(detail) == 20


def test_c4_no_relation_with_many_cells_is_supported_and_few_cells_inconclusive():
    rng = np.random.default_rng(5)
    cells, mase_frame = _c4_inputs(rng.random(200), rng.random(200))
    _detail, summary = c4_table(cells, mase_frame, 0.5, 0.7, BOOT, LEVEL, SEED)
    row = summary.iloc[0]
    assert abs(row["rho"]) < 0.2 and row["abs_ci_bound"] < 0.7 and row["h4_label"] == SUPPORTED
    cells, mase_frame = _c4_inputs(rng.random(6), rng.random(6))
    _detail, small = c4_table(cells, mase_frame, 0.5, 0.7, BOOT, LEVEL, SEED)
    assert small.iloc[0]["h4_label"] == INCONCLUSIVE


def test_c4_drops_cells_without_benefit_and_counts_them():
    rng = np.random.default_rng(9)
    benefit = rng.random(30)
    benefit[:4] = np.nan
    cells, mase_frame = _c4_inputs(rng.random(30), benefit)
    detail, summary = c4_table(cells, mase_frame, 0.5, 0.7, BOOT, LEVEL, SEED)
    assert summary.iloc[0]["n_cells_without_benefit"] == 4 and summary.iloc[0]["n_cells_used"] == 26 and len(detail) == 30


def make_cross(cross_by_target):
    rows = []
    for f in FORECASTERS:
        for target in ("D1", "D2"):
            for source in ("D1", "D2"):
                for h in HORIZONS:
                    mase = 1.0 if source == target else cross_by_target[f][target]
                    rows.append(dict(protocol="fixed", source=source, target=target, forecaster=f, horizon=h,
                                     group="all", mase=mase))
                    rows.append(dict(protocol="fixed", source=source, target=target, forecaster=f, horizon=h,
                                     group="event", mase=99.0))
    return pd.DataFrame(rows)


def make_q8(foreign_factor):
    rows = []
    for ds, other in (("D1", "D2"), ("D2", "D1")):
        for variant in ("P1", "P2", "P3_lightgbm", "P4", "P4a", "P5"):
            for cls in ("abrupt", "ramped"):
                for i in range(5):
                    for level in (ds, other):
                        rows.append(dict(dataset=ds, onset_class=cls, m=5, headroom=1.6, replicate=i, variant=variant,
                                         event_id="%s-%s-%d" % (ds, cls, i), level=level,
                                         damage=10.0 + i if level == ds else (10.0 + i) * foreign_factor))
    return pd.DataFrame(rows)


def run5(cross, q8):
    return c5_tables(cross, q8, FORECASTERS, HORIZONS, "all", 1.25, BOOT, LEVEL, SEED)


def test_c5_poor_transfer_everywhere_is_supported():
    both = {"ridge": {"D1": 1.5, "D2": 1.5}, "lightgbm": {"D1": 1.5, "D2": 1.5}}
    detail, summary = run5(make_cross(both), make_q8(1.5))
    assert len(detail) == 16 + 12
    by = summary.set_index("unit")
    assert by.loc["ridge", "median_ratio"] == pytest.approx(1.5) and by.loc["all policies", "median_ratio"] == pytest.approx(1.5)
    assert summary.iloc[-1]["h5_label"] == SUPPORTED and bool(summary["met"].all())
    xa2 = detail[detail["part"] == "xa2_ratio"]
    assert (xa2["n_events"] == 10).all() and (xa2["regret_mean"] > 0).all()


def test_c5_good_transfer_is_not_supported_and_direction_medians_are_reported():
    good = {"ridge": {"D1": 1.0, "D2": 1.0}, "lightgbm": {"D1": 1.0, "D2": 1.0}}
    _detail, summary = run5(make_cross(good), make_q8(1.5))
    assert summary.iloc[-1]["h5_label"] == NOT_SUPPORTED
    asym = {"ridge": {"D1": 3.0, "D2": 1.0}, "lightgbm": {"D1": 3.0, "D2": 1.0}}
    _detail, summary = run5(make_cross(asym), make_q8(1.5))
    row = summary[summary["unit"] == "ridge"].iloc[0]
    assert row["median_ratio"] == pytest.approx(2.0) and row["median_target_D1"] == pytest.approx(3.0)
    assert row["median_target_D2"] == pytest.approx(1.0)


def test_c5_pairing_failures_raise():
    q8 = make_q8(1.5)
    drop = q8[(q8["level"] != q8["dataset"])].index[0]
    with pytest.raises(ValueError, match="no partner"):
        run5(make_cross({"ridge": {"D1": 1.5, "D2": 1.5}, "lightgbm": {"D1": 1.5, "D2": 1.5}}), q8.drop(index=drop))
    cross = make_cross({"ridge": {"D1": 1.5, "D2": 1.5}, "lightgbm": {"D1": 1.5, "D2": 1.5}})
    with pytest.raises(ValueError, match="need one"):
        run5(cross[~((cross["forecaster"] == "ridge") & (cross["target"] == "D1") & (cross["source"] == "D2")
                      & (cross["horizon"] == 5) & (cross["group"] == "all"))], make_q8(1.5))


POLICIES = ["P0", "P1", "P2", "P3", "P4", "P5"]
LEVELS = {"P5": 4.0, "P3_lightgbm": 6.0, "P4": 7.0, "P1": 9.0, "P2": 11.0, "P0": 30.0}


def make_ranking_must(order_d2):
    rows = []
    rng = np.random.default_rng(1)
    for ds in ("D1", "D2"):
        for i in range(20):
            noise = rng.normal(0, 0.1)
            for variant, level in LEVELS.items():
                value = level if ds == "D1" else LEVELS[order_d2[variant]]
                for rep in range(2):
                    rows.append(dict(dataset=ds, onset_class="abrupt" if i % 2 else "ramped", m=5, headroom=1.6,
                                     replicate=rep, event_id="%s-%d" % (ds, i), variant=variant, damage=value + noise))
    return pd.DataFrame(rows)


def run6(must):
    return c6_tables(must, {"D1": "P3_lightgbm", "D2": "P3_lightgbm"}, POLICIES, 5, 1.6, 0.6, BOOT, LEVEL, SEED)


def test_c6_same_ranking_is_supported_and_reversed_is_not():
    same = {name: name for name in LEVELS}
    detail, summary = run6(make_ranking_must(same))
    assert summary.iloc[0]["tau"] == pytest.approx(1.0) and summary.iloc[0]["h6_label"] == SUPPORTED
    row = detail.set_index("policy")
    assert row.loc["P5", "rank_D1"] == 1 and row.loc["P0", "rank_D1"] == 6 and row.loc["P3", "median_damage_D1"] == pytest.approx(6.0, abs=0.5)
    order = sorted(LEVELS, key=LEVELS.get)
    reverse = {a: b for a, b in zip(order, reversed(order))}
    _detail, summary = run6(make_ranking_must(reverse))
    assert summary.iloc[0]["tau"] == pytest.approx(-1.0) and summary.iloc[0]["h6_label"] == NOT_SUPPORTED


def test_c6_missing_policy_raises():
    must = make_ranking_must({name: name for name in LEVELS})
    with pytest.raises(ValueError, match="every event needs every policy"):
        run6(must[~((must["variant"] == "P4") & (must["event_id"] == "D1-3"))])


def test_tables_reproduce_exactly():
    names = ["ridge", "lightgbm", "naive"]
    must = make_must(names)
    pd.testing.assert_frame_equal(benefit_cells(must, {"D1": "P1", "D2": "P1"}, names, 5, 1.6, 0.05),
                                  benefit_cells(must, {"D1": "P1", "D2": "P1"}, names, 5, 1.6, 0.05))
    a = run6(make_ranking_must({name: name for name in LEVELS}))
    b = run6(make_ranking_must({name: name for name in LEVELS}))
    pd.testing.assert_frame_equal(a[1], b[1])


def test_stages_are_registered():
    spec = importlib.util.spec_from_file_location("analyze", Path(__file__).resolve().parents[1] / "scripts" / "05_analyze.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert {"c4_accuracy_benefit", "c5_transfer", "c6_ranking"} <= set(mod.STAGES)
