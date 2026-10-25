"""C1 lag tax (A62): planted values are recovered, the cell filters work, the Spearman test is clustered and
pairing failures are loud. No real result row is read."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src import lagtax as lt

spec = importlib.util.spec_from_file_location("analyze", Path(__file__).resolve().parents[1] / "scripts" / "05_analyze.py")
an = importlib.util.module_from_spec(spec)
spec.loader.exec_module(an)

N = 40
M_LEVELS = [1, 2, 5, 10]
CFG = {"migration_steps_m0": 5, "headroom": 1.6, "m_levels": M_LEVELS, "n_boot": 999, "ci_level": 0.95,
       "alpha_adj": 0.05, "min_events": 24, "master_seed": 42}
BEST = {"D1": "P1", "D2": "P1"}


def growing(j, k, level):
    return level + 0.1 * k + 0.05 * ((j % 5) - 2)


def falling(j, k, level):
    return -(level + 0.1 * k) + 0.05 * ((j % 5) - 2)


def null_tax():
    noise = np.random.default_rng(7).normal(0, 0.02, (N, len(M_LEVELS)))

    def tax(j, k, level):
        return 0.5 * (-1) ** j * (1 + j / N) + noise[j, k]
    return tax


def make_must(tax_d1, tax_d2, drop=None):
    out = []
    for dataset, tax in (("D1", tax_d1), ("D2", tax_d2)):
        for j in range(N):
            for headroom in (1.6, 1.3):
                for k, m in enumerate(M_LEVELS):
                    base = 1 + 0.1 * j
                    value = tax(j, k, 0.5) if headroom == 1.6 else 99.0
                    for variant, damage in (("P1", base + value), ("P5", base), ("P2", 1000.0)):
                        out.append({"cell": "%s-c%d-m%d-h%s-%s" % (dataset, j % 2, m, headroom, variant),
                                    "replicate": j // 2, "dataset": dataset, "onset_class": "c%d" % (j % 2),
                                    "m": m, "headroom": headroom, "variant": variant, "event_id": "e%d" % j,
                                    "damage": damage})
    frame = pd.DataFrame(out)
    if drop is not None:
        frame = frame[~drop(frame)]
    return frame


def table_of(must):
    return lt.c1_table(must, BEST, CFG["migration_steps_m0"], CFG["headroom"], M_LEVELS, CFG["n_boot"],
                       CFG["ci_level"], CFG["alpha_adj"], CFG["min_events"], CFG["master_seed"])


def test_lag_tax_is_the_paired_difference_at_the_default_cell():
    must = make_must(growing, growing)
    paired = lt.lag_tax_frame(must, BEST)
    assert len(paired) == 2 * N * 2 * len(M_LEVELS)
    events = lt.default_events(paired, "D1", 5, 1.6)
    assert len(events) == N
    assert abs(events["tax"].mean() - 0.7) < 1e-9
    assert abs(paired.loc[(paired["m"] == 5) & (paired["headroom"] == 1.3), "tax"].mean() - 99.0) < 1e-9


def test_c1_is_supported_when_tax_is_positive_and_grows_with_m():
    table = table_of(make_must(growing, growing))
    assert len(table) == 4 and set(table["test"]) == {lt.WILCOXON, lt.SPEARMAN}
    w = table[(table["dataset"] == "D1") & (table["test"] == lt.WILCOXON)].iloc[0]
    s = table[(table["dataset"] == "D1") & (table["test"] == lt.SPEARMAN)].iloc[0]
    assert abs(w["estimate"] - 0.7) < 1e-9 and w["ci_low"] > 0 and w["n_events"] == N
    assert s["estimate"] > 0 and s["ci_low"] > 0
    assert (table["p_holm"] < CFG["alpha_adj"]).all() and (table["p_holm"] >= table["p_raw"]).all()
    assert (table["h1_label"] == "supported").all()


def test_c1_is_inconclusive_when_one_dataset_has_no_tax_and_not_supported_when_it_reverses():
    table = table_of(make_must(growing, null_tax()))
    d2 = table[table["dataset"] == "D2"]
    assert (d2["p_raw"] > CFG["alpha_adj"]).all()
    assert (table["h1_label"] == "inconclusive").all()
    reversed_table = table_of(make_must(growing, falling))
    assert (reversed_table["h1_label"] == "not supported").all()
    assert (reversed_table.loc[reversed_table["dataset"] == "D2", "ci_high"] < 0).all()


def test_spearman_matches_scipy_and_the_permutation_test_is_clustered():
    rng = np.random.default_rng(3)
    levels = np.asarray(M_LEVELS, dtype=np.float64)
    tax = rng.normal(0, 1, (30, 4))
    assert abs(lt.pooled_rho(tax, levels) - stats.spearmanr(tax.ravel(), np.tile(levels, 30)).statistic) < 1e-12
    p1 = lt.permutation_p(tax, levels, 999, 5)
    assert p1 == lt.permutation_p(tax, levels, 999, 5) and 0 < p1 <= 1
    offsets = np.where(np.arange(30) % 2 == 0, 100.0, -100.0)[:, None]
    heterogeneous = offsets + rng.normal(0, 0.1, (30, 4))
    assert lt.permutation_p(heterogeneous, levels, 999, 5) > CFG["alpha_adj"]
    sloped = offsets + 0.5 * np.arange(4)[None, :] + rng.normal(0, 0.1, (30, 4))
    assert lt.permutation_p(sloped, levels, 999, 5) < 0.01
    constant = np.ones((30, 4))
    assert lt.permutation_p(constant, levels, 99, 5) == 1.0
    assert all(np.isnan(v) for v in lt.rho_interval(constant, levels, 99, 0.95, 5))


def test_label_rules():
    def table(estimate, ci_high, p_holm, n):
        return pd.DataFrame({"estimate": estimate, "ci_high": ci_high, "p_holm": p_holm, "n_events": n})
    ok = table([1.0, 0.5], [2.0, 1.0], [0.01, 0.02], [40, 40])
    assert lt.c1_label(ok, 0.05, 24) == "supported"
    assert lt.c1_label(table([1.0, 0.5], [2.0, 1.0], [0.01, 0.2], [40, 40]), 0.05, 24) == "inconclusive"
    assert lt.c1_label(table([1.0, -0.5], [2.0, -0.1], [0.01, 0.9], [40, 40]), 0.05, 24) == "not supported"
    assert lt.c1_label(table([1.0, 0.5], [2.0, 1.0], [0.01, 0.02], [40, 10]), 0.05, 24) == "inconclusive"


def test_pairing_and_missing_levels_are_loud():
    broken = make_must(growing, growing, drop=lambda f: (f["variant"] == "P5") & (f["event_id"] == "e3") & (f["m"] == 2))
    with pytest.raises(ValueError):
        lt.lag_tax_frame(broken, BEST)
    paired = lt.lag_tax_frame(make_must(growing, growing), BEST)
    short = paired[~((paired["event_id"] == "e3") & (paired["m"] == 10))]
    with pytest.raises(ValueError):
        lt.tax_matrix(short, "D1", 1.6, M_LEVELS)


def test_stage_writes_the_table(tmp_path):
    folder = tmp_path / "must"
    folder.mkdir()
    frame = make_must(growing, growing)
    (folder / "grid_1.jsonl").write_text("\n".join(json.dumps(r) for r in frame.to_dict("records")) + "\n", encoding="utf-8")
    out = tmp_path / "tables"
    table = an.stage_c1_lag_tax(CFG, {"must": str(folder)}, out)
    saved = pd.read_csv(out / "c1_lag_tax.csv")
    assert len(saved) == 4 and (saved["h1_label"] == "supported").all() and table["h1_label"].iloc[0] == "supported"
