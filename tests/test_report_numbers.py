import json

import numpy as np
import pandas as pd
import pytest

from src import report_numbers as rn


def test_macro_name_and_format():
    assert rn.macro_name("cfg_n_boot") == "paCfgNBoot"
    assert rn.macro_name("cfg_d1_train_days") == "paCfgDOneTrainDays"
    assert rn.macro_name("rt_policy_D1_D3_P4a_damage_mean") == "paRtPolicyDOneDThreePFouraDamageMean"
    assert rn.format_value(168) == "168" and rn.format_value(np.int64(7)) == "7"
    assert rn.format_value(19.08123) == "19.08" and rn.format_value(1234.5678) == "1235"
    assert rn.format_value(0.000391234, "ledger_C1_p_holm_min") == "0.000391"
    assert rn.format_value(0.2765939, "ledger_C2_p_holm_min") == "0.277"
    assert rn.format_value(1.5e-5) == r"\ensuremath{1.5\times 10^{-5}}"
    assert rn.format_value(float("nan")) == "n/a" and rn.format_value(None) == "n/a"
    assert rn.latex_escape("a_b & 5% {x}") == r"a\_b \& 5\% \{x\}"
    with pytest.raises(TypeError):
        rn.format_value(True)


def test_macros_skip_and_clash():
    e = {"cfg_a": {"value": 3, "source": "s"}, "cfg_flag": {"value": False, "source": "s"},
         "cfg_list": {"value": [1, 2], "source": "s"}, "cfg_text": {"value": "x", "source": "s"},
         "ledger_C1_label": {"value": "inconclusive", "source": "s"}, "rt_x": {"value": None, "source": "s"}}
    text = rn.macros_text(e)
    assert r"\newcommand{\paCfgA}{3}" in text and r"\paLedgerCOneLabel}{inconclusive}" in text
    assert r"\paRtX}{n/a}" in text and "Flag" not in text and "List" not in text and "CfgText" not in text
    with pytest.raises(ValueError):
        rn.macros_text({"a_b": {"value": 1, "source": "s"}, "a__b": {"value": 2, "source": "s"}})


def test_table_entries_keys_and_uniqueness():
    df = pd.DataFrame({"dataset": ["D1", "D1"], "policy": ["P1", "P3"], "mean": [1.5, np.nan], "n": [60, 60]})
    e = {}
    rn.table_entries(e, "rt_policy", df, "src")
    assert e["rt_policy_D1_P1_mean"]["value"] == 1.5 and e["rt_policy_D1_P3_mean"]["value"] is None
    assert e["rt_policy_D1_P3_n"]["value"] == 60
    dup = pd.DataFrame({"dataset": ["D1", "D1"], "m": [1, 5], "mean": [1.0, 2.0]})
    e2 = {}
    rn.table_entries(e2, "t", dup, "src")
    assert "t_D1_5_mean" in e2
    with pytest.raises(ValueError):
        rn.table_entries({}, "t", pd.DataFrame({"dataset": ["D1", "D1"], "mean": [1.0, 2.0]}), "src")


LEDGER = pd.DataFrame({
    "id": ["C1", "C2", "A1"], "hypothesis": ["H1", "H2", "H7"], "quantity": ["q", "q", "q"], "threshold": ["t", "t", "t"],
    "result": ["a | b", "r", "r"], "label": ["inconclusive", "supported", "inconclusive"], "n_tests": [4, 2, 3],
    "holm_family": ["F1", "F2", "F9"], "p_holm_min": [0.0004, 0.2765939942410531, np.nan],
    "status": ["confirmatory", "confirmatory", "ablation"], "exploratory_columns": [np.nan, np.nan, "c"], "note": [np.nan] * 3})
QUEUE = pd.DataFrame({
    "queue": ["must", "q1", "q5", "q9"], "planned_cells": [480.0, 264.0, np.nan, np.nan], "completed_cells": [480.0, 264.0, np.nan, np.nan],
    "completed_rows": [14400.0, 7920.0, 468.0, np.nan], "status": ["complete", "complete", "complete", "not completed"],
    "note": [np.nan, np.nan, np.nan, "FA16 not made"]})


def test_ledger_and_queue_entries():
    e = {}
    rn.ledger_entries(e, LEDGER, "l")
    v = lambda k: e[k]["value"]
    assert v("ledger_n_hypotheses") == 3 and v("ledger_n_tests_total") == 9
    assert v("ledger_n_status_confirmatory") == 2 and v("ledger_n_tests_status_confirmatory") == 6
    assert v("ledger_n_label_inconclusive") == 2 and v("ledger_C2_label") == "supported"
    assert v("ledger_C1_n_tests") == 4 and v("ledger_C2_p_holm_min") == 0.2765939942410531
    assert "ledger_A1_p_holm_min" not in e
    q = {}
    rn.queue_entries(q, QUEUE, "q")
    assert q["queue_n_grids"]["value"] == 2 and q["queue_total_completed_rows"]["value"] == 22320
    assert q["queue_must_planned_cells"]["value"] == 480 and isinstance(q["queue_must_planned_cells"]["value"], int)
    assert q["queue_n_not_completed"]["value"] == 1 and "queue_q5_planned_cells" not in q


def test_render_ledger_md():
    md = rn.render_ledger_md(LEDGER, QUEUE)
    assert "- confirmatory: 2 hypotheses, 6 tests" in md and "- ablation: 1 hypotheses, 3 tests" in md
    assert "- q9: FA16 not made" in md and "a \\| b" in md and "| C2 |" in md and "0.277" in md
    assert md.count("\n| ") == 5 and "\r" not in md


def test_numbers_json_is_valid_and_ordered():
    e = {}
    rn._add(e, "b", np.float64(1.0 / 3), "s")
    rn._add(e, "a", float("nan"), "s")
    data = json.loads(rn.numbers_json_text(e))
    assert list(data) == ["b", "a"] and data["b"]["value"] == 1.0 / 3 and data["a"]["value"] is None
    with pytest.raises(ValueError):
        rn._add(e, "a", 1, "s")


def test_build_index_status(tmp_path):
    out, tab = tmp_path / "f", tmp_path / "t"
    out.mkdir(), tab.mkdir()
    for n in ("ta3_forecast_accuracy.csv", "c1_lag_tax.csv", "rt_policy_metrics.csv"):
        (tab / n).write_text("a\n1\n")
    (out / "ta1_datasets.csv").write_text("a\n")
    (out / "fa5_lag_tax.pdf").write_text("x")
    idx = rn.build_index(out, tab).set_index("guide_id")
    assert idx.loc["TA1", "status"] == "made" and idx.loc["TA2", "status"] == "to make"
    assert idx.loc["TA3", "status"] == "made" and idx.loc["TA4", "status"] == "to make"
    assert idx.loc["FA5", "status"] == "made" and idx.loc["FA3", "status"] == "to make"
    assert idx.loc["FA4", "status"] == "source missing" and "forecast_accuracy_early_warning.csv" in idx.loc["FA4", "note"]
    assert idx.loc["FA16", "status"] == "not completed"
    assert list(rn.build_index(out, tab).columns) == rn.INDEX_COLUMNS
