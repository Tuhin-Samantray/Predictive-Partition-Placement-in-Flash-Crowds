import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.config import load_config
from src.layer_runs import ReplayPolicy, load_cross, run_replicate_layers
from src.layers import simulate_layers
from src.replicates import load_dataset, make_replicate, select_events

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("layers_world", ROOT / "tests" / "layers_world.py")
lw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lw)
P1 = {"ewma_alpha": 0.3, "theta_hi": 0.9, "cooldown": 5}


def replay_of(w, res, **kw):
    args = dict(lw.DEFAULTS)
    args.update(kw)
    finish = {(t0, p): tf for t0, p, _s, _d, tf in res.moves}
    return simulate_layers(w["load"], w["nbytes"], w["rel"], w["node0"], w["mu"], w["link"], w["sizes"],
                           policy=ReplayPolicy(res.moves), replay_finish=finish, mode="const_m", **args)


@pytest.mark.parametrize("phi", [1.0, 60.0])
def test_replay_keeps_the_placements_and_removes_only_the_traffic(phi):
    w = lw.world()
    orig = lw.run(w, lw.PLANS, mode="coupled", phi=phi)
    rep = replay_of(w, orig, phi=phi)
    assert len(orig.moves) > 0
    assert rep.moves == orig.moves
    assert np.array_equal(rep.node_hist, orig.node_hist)
    assert np.array_equal(rep.arrivals, orig.arrivals)
    assert np.array_equal(rep.wait_cpu, orig.wait_cpu)
    assert not rep.mig_charge.any() and rep.sent.sum() == 0
    fgb = np.array([np.bincount(h, weights=w["nbytes"][t], minlength=lw.N_NODES) for t, h in enumerate(rep.node_hist)])
    assert np.allclose(rep.net_in, fgb)
    assert (rep.wait_net <= orig.wait_net + 1e-12).all()
    if phi > 1:
        assert any(tf == w["load"].shape[0] for _t, _p, _s, _d, tf in orig.moves)


def test_replay_finish_needs_const_m():
    w = lw.world()
    with pytest.raises(ValueError):
        lw.run(w, None, mode="coupled", replay_finish={})


def test_run_replicate_layers_cl_fields_and_replay():
    cfg = load_config(str(ROOT / "configs" / "default.yaml"))
    data = load_dataset(cfg, "D2")
    rep = make_replicate(cfg, data, select_events(cfg, data, "tune", 1)[0], 0, cfg["headroom"])
    cross = load_cross(cfg, data)
    a = run_replicate_layers(cfg, rep, cross, "P1", P1, cl=True, replay=True)
    b = run_replicate_layers(cfg, rep, cross, "P1", P1, cl=True, replay=True)
    assert a == b and a["cl_replay_ok"]
    assert a["cl_damage_replay"] <= a["damage"] + 1e-12
    assert a["cl_wait_mean_replay"] <= a["cl_wait_mean"] + 1e-12
    assert 0 <= a["cl_mig_share"] <= 1
    assert a["cl_viol_both"] <= min(a["cl_viol_cpu"], a["cl_viol_net"])
    base = run_replicate_layers(cfg, rep, cross, "P1", P1)
    assert all(a[k] == v for k, v in base.items())
    with pytest.raises(ValueError):
        run_replicate_layers(cfg, rep, cross, "P1", P1, mode="const_m", replay=True)
