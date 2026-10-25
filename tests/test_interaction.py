import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from src import interaction as ix
from src.analysis_data import resolve_folders

spec = importlib.util.spec_from_file_location("analyze", Path(__file__).resolve().parents[1] / "scripts" / "05_analyze.py")
an = importlib.util.module_from_spec(spec)
spec.loader.exec_module(an)

N = 40
CFG = {"migration_steps_m0": 5, "headroom": 1.6, "margin_quantile": 0.9, "n_boot": 999, "ci_level": 0.95,
       "interaction_effect_frac": 0.05, "interaction_min_active": 2, "master_seed": 42}


def base(j):
    return 10 + 0.2 * j


def rows(variant, factor, param, level, damage_of, m=5, headroom=1.6):
    out = []
    for j in range(N):
        out.append({"cell": "D1-abrupt-m%d-h%s-%s-%s-%s" % (m, headroom, variant, param, level), "replicate": j,
                    "dataset": "D1", "onset_class": "abrupt", "m": m, "headroom": headroom, "variant": variant,
                    "event_id": "e%d" % j, "damage": damage_of(j), "factor": factor, "param": param, "level": level})
    return out


def scaled(eff):
    return lambda j: base(j) * (1 + eff) + 0.05 * ((j % 5) - 2)


def alternating(j):
    return base(j) + 5 * (-1) ** j


def planted():
    must = (rows("P1", None, None, None, base) + rows("P2", None, None, None, base)
            + rows("P1", None, None, None, lambda j: 99.0, m=10) + rows("P2", None, None, None, lambda j: 99.0, headroom=2.0))
    q1 = (rows("P1", "AA4", "cooldown", 5, scaled(0.30)) + rows("P1", "AA5", "budget", 1, scaled(-0.25))
          + rows("P1", "AA5", "budget", "unlimited", scaled(0.30)) + rows("P1", "AA9", "ewma_alpha", 0.1, scaled(0.02))
          + rows("P2", "AA3", "hysteresis", 0.1, scaled(0.30)) + rows("P2", "AA4", "cooldown", 5, alternating)
          + rows("P4a", "AA6", "margin_quantile", 0.9, base) + rows("P4a", "AA6", "margin_quantile", 0.8, scaled(0.40))
          + rows("P4a", "AA4", "cooldown", 5, scaled(0.40)))
    return pd.DataFrame(q1), pd.DataFrame(must)


def effects_of(q1, must):
    return ix.interaction_effects(q1, must, CFG["migration_steps_m0"], CFG["headroom"], CFG["margin_quantile"],
                                  CFG["n_boot"], CFG["ci_level"], CFG["interaction_effect_frac"], CFG["master_seed"])


def test_planted_interaction_fires_only_where_two_factors_are_active():
    q1, must = planted()
    effects = effects_of(q1, must)
    act = effects.set_index(["variant", "factor", "level"])["active"]
    assert act[("P1", "AA4", "5")] and act[("P1", "AA5", "1")] and act[("P1", "AA5", "unlimited")]
    assert not act[("P1", "AA9", "0.1")]
    row = effects.set_index(["variant", "factor", "level"]).loc[("P1", "AA9", "0.1")]
    assert row["excludes_zero"] and not row["large"]
    assert act[("P2", "AA3", "0.1")] and not act[("P2", "AA4", "5")]
    assert act[("P4a", "AA6", "0.8")] and act[("P4a", "AA4", "5")]
    assert ("P4a", "AA6", "0.9") not in act.index
    trig = ix.interaction_trigger(effects, CFG["interaction_min_active"]).set_index("variant")
    assert bool(trig.loc["P1", "fires"]) and trig.loc["P1", "active_factors"] == "AA4,AA5"
    assert not bool(trig.loc["P2", "fires"]) and trig.loc["P2", "n_active_factors"] == 1
    assert bool(trig.loc["P4a", "fires"]) and trig.loc["P4a", "active_factors"] == "AA4,AA6"


def test_effect_sizes_are_the_planted_ones_and_the_reference_ignores_other_m_and_headroom():
    q1, must = planted()
    effects = effects_of(q1, must).set_index(["variant", "factor", "level"])
    assert abs(effects.loc[("P1", "AA4", "5"), "rel_effect"] - 0.30) < 0.01
    assert abs(effects.loc[("P1", "AA5", "1"), "rel_effect"] + 0.25) < 0.01
    assert effects.loc[("P1", "AA4", "5"), "n_events"] == N


def test_effects_are_deterministic_and_pairing_failures_are_loud():
    q1, must = planted()
    assert effects_of(q1, must).equals(effects_of(q1, must))
    broken = q1[~((q1["variant"] == "P1") & (q1["factor"] == "AA4") & (q1["replicate"] == 3))]
    with pytest.raises(ValueError):
        effects_of(broken, must)


def test_resolve_folders_reroots_the_mapping():
    out = resolve_folders({"must": "results", "q1": "results/q1", "q8": "results/q8"}, "frozen", "must")
    assert Path(out["must"]) == Path("frozen") and Path(out["q1"]) == Path("frozen/q1") and Path(out["q8"]) == Path("frozen/q8")


def test_stage_writes_both_tables_and_returns_the_trigger(tmp_path):
    q1, must = planted()
    for name, frame in (("must", must), ("q1", q1)):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "grid_1.jsonl").write_text("\n".join(json.dumps(r) for r in frame.to_dict("records")) + "\n", encoding="utf-8")
    out = tmp_path / "tables"
    trigger = an.stage_interaction_trigger(CFG, {"must": str(tmp_path / "must"), "q1": str(tmp_path / "q1")}, out)
    assert (out / "interaction_effects.csv").exists() and (out / "interaction_trigger.csv").exists()
    assert trigger["fires"].sum() == 2
    assert pd.read_csv(out / "interaction_trigger.csv")["fires"].sum() == 2
