import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from src.config import load_config
from src.seeds import derive_seed

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "oracle_check"
ORACLE = "P5"
IN_SAMPLE = {"D1": "held-out tune events", "D2": "IN SAMPLE (all tune events were used in tuning)"}


def _damage():
    if not OUT.exists():
        pytest.fail("results/oracle_check missing: run 04_run_grid.py --oracle-check (A20, A21) first")
    table = {}
    for path in sorted(OUT.glob("grid_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            r = json.loads(line)
            table.setdefault(r["dataset"], {}).setdefault(r["variant"], {})[(r["event_id"], r["replicate"])] = r["damage"]
    return table


def test_oracle_dominance():
    cfg = load_config(str(ROOT / "configs" / "default.yaml"))
    table = _damage()
    failures = []
    for ds in sorted(table):
        by_variant = table[ds]
        assert ORACLE in by_variant, ds + " has no oracle rows"
        keys = sorted(by_variant[ORACLE])
        if ds == "D1":
            assert len(keys) == cfg["oracle_check_events"], "D1 must have oracle_check_events held-out events"
        print("\n%s (%s), %d events, mean damage per variant:" % (ds, IN_SAMPLE[ds], len(keys)))
        for variant in sorted(by_variant):
            assert sorted(by_variant[variant]) == keys, variant + " rows differ from the oracle rows"
            print("  %-18s %.6g" % (variant, np.mean([by_variant[variant][k] for k in keys])))
        oracle = np.array([by_variant[ORACLE][k] for k in keys])
        for variant in sorted(by_variant):
            if variant == ORACLE:
                continue
            other = np.array([by_variant[variant][k] for k in keys])
            diff = oracle - other
            rng = np.random.default_rng(derive_seed("oracle_check", cfg["master_seed"], ds, variant))
            ci = stats.bootstrap((diff,), np.mean, n_resamples=cfg["n_boot"], confidence_level=cfg["ci_level"],
                                 random_state=rng).confidence_interval
            verdict = "WORSE" if ci.low > 0 else "ok"
            print("  P5 - %-14s mean %.4g CI [%.4g, %.4g] %s" % (variant, diff.mean(), ci.low, ci.high, verdict))
            if ci.low > 0:
                failures.append((ds, variant, float(diff.mean()), float(ci.low), float(ci.high)))
    assert not failures, "oracle significantly worse than: %s" % failures
