from pathlib import Path

import numpy as np
import pytest

from src.config import load_config
from src.policies import P4Hedged, P4aAdaptive, build_policy
from src.replicates import forecast_window, load_dataset, load_forecast, make_replicate, run_replicate, select_events
from src.simulator import simulate
from src.tuning import ALL_TUNABLE, tuning_grid

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = [1, 2, 5, 10]


def make(gamma, m=3, d=1, window=8, a_min=0.01, a_max=0.2, q=0.9, hyst=0.0, horizons=HORIZONS):
    return P4aAdaptive(hyst, m, d, horizons, q, window, gamma, a_min, a_max)


def arrays(seed=0, n=60, parts=6):
    rng = np.random.default_rng(seed)
    obs = rng.uniform(1, 10, (n, parts))
    fc = obs[:, None, :] + rng.normal(0, 2, (n, len(HORIZONS), parts))
    return obs, fc


def test_gamma_zero_reproduces_the_p4_prediction_at_every_step():
    obs, fc = arrays()
    p4 = P4Hedged(0.0, 3, 1, HORIZONS, 0.9, 8)
    p4a = make(0.0)
    for n in range(1, obs.shape[0] + 1):
        assert np.array_equal(p4a.advance(obs[:n], fc[:n]), p4.pred_load(obs[:n], fc[:n])), n


def test_catching_up_in_one_call_equals_step_by_step():
    obs, fc = arrays(1)
    a, b = make(0.05), make(0.05)
    for n in range(1, obs.shape[0] + 1):
        last = a.advance(obs[:n], fc[:n])
    one = b.advance(obs, fc)
    assert np.array_equal(last, one) and np.array_equal(a.alpha, b.alpha)
    assert a.alpha_trace == b.alpha_trace and (a.hits, a.trials) == (b.hits, b.trials)


def test_alpha_follows_the_hand_computed_trajectory_and_clamp():
    obs = np.array([[5.0], [5.0], [5.0], [9.0], [5.0], [5.0]])
    fc = np.full((6, 1, 1), 5.0)
    pol = P4aAdaptive(0.0, 1, 0, [1], 0.9, 100, 0.2, 0.01, 0.2)
    want = [0.1, 0.12, 0.14, 0.01, 0.03, 0.05]
    for n, w in enumerate(want, start=1):
        pol.advance(obs[:n], fc[:n])
        assert pol.alpha.item() == pytest.approx(w, abs=1e-12), n
        if n == 4:
            assert pol.pending[3].item() == pytest.approx(9.0)
    assert (pol.hits, pol.trials) == (4, 5)
    assert pol.coverage() == pytest.approx(0.8)


def test_needs_a_positive_horizon_and_a_fresh_instance():
    with pytest.raises(ValueError):
        make(0.0, m=0, d=0)
    obs, fc = arrays(2)
    pol = make(0.0)
    pol.advance(obs[:10], fc[:10])
    with pytest.raises(ValueError):
        pol.advance(obs[:5], fc[:5])
    with pytest.raises(ValueError):
        make(0.0).advance(obs[:5], fc[:6])


def test_p4a_is_tuned_like_p4_and_built_by_name():
    cfg = load_config(str(ROOT / "configs" / "default.yaml"))
    assert "P4a" in ALL_TUNABLE and tuning_grid(cfg, "P4a") == tuning_grid(cfg, "P4")
    ctx = {"m": 3, "d": 1, "horizons": HORIZONS, "margin_quantile": 0.9, "residual_window": 8,
           "gamma": 0.005, "a_min": 0.01, "a_max": 0.2}
    assert isinstance(build_policy("P4a", {"hysteresis": 0.1}, np.ones(4), ctx), P4aAdaptive)


def test_decisions_do_not_depend_on_data_after_the_visible_prefix():
    rng = np.random.default_rng(3)
    n_steps, n_part, n_nodes, cut, d = 60, 8, 4, 30, 1
    load = rng.uniform(0.5, 1.5, (n_steps, n_part))
    load[:, 0] = 50.0
    fc = np.repeat(load[:, None, :], len(HORIZONS), axis=1) + rng.normal(0, 0.1, (n_steps, len(HORIZONS), n_part))
    load2, fc2 = load.copy(), fc.copy()
    load2[cut + 1:] = rng.uniform(0, 100, load2[cut + 1:].shape)
    fc2[cut + 1:] = rng.uniform(0, 100, fc2[cut + 1:].shape)
    mu = np.full(n_nodes, 30.0)
    node0 = np.arange(n_part) % n_nodes
    moves = []
    for lo, fo in ((load, fc), (load2, fc2)):
        pol = make(0.05, m=2, d=d)
        res = simulate(lo, node0, mu, m=2, kappa=0.3, budget=4, d=d, k=1, policy=pol, cooldown=0, forecast=fo)
        moves.append([mv for mv in res.moves if mv[0] <= cut + d])
    assert len(moves[0]) > 0 and moves[0] == moves[1]


def test_gamma_zero_run_matches_p4_on_a_real_replicate():
    cfg = load_config(str(ROOT / "configs" / "default.yaml"))
    data = load_dataset(cfg, "D1")
    ev = select_events(cfg, data, "tune", 1)[0]
    rep = make_replicate(cfg, data, ev, 0, cfg["headroom"])
    fw = forecast_window(load_forecast(cfg, "D1", "lightgbm"), rep)
    params = {"hysteresis": 0.1, "cooldown": 5}
    ref = run_replicate(cfg, rep, "P4", params, forecast=fw)
    zero = run_replicate({**cfg, "gamma": 0.0}, rep, "P4a", params, forecast=fw)
    assert ref == zero
    assert run_replicate(cfg, rep, "P4a", params, forecast=fw)["conservation_ok"]
