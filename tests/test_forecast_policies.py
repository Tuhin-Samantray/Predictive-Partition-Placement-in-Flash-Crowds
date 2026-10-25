import numpy as np
import pytest

from small_world import real_cfg, small
from src.forecast_select import tune_mase
from src.forecasters import naive_table
from src.policies import (FORECAST_TUNABLE, positive_quantile, P3Forecast, P4Hedged, P5Oracle, blend, build_policy,
                          horizon_weights)
from src.replicates import forecast_window, tune_replicates
from src.simulator import simulate
from src.tuning import tune_policy
from stub_policy import make_case

HORIZONS = [1, 2, 5, 10]


def _state(node_of, forecast=None, free=4, n_nodes=2):
    node_of = np.array(node_of)
    return dict(node_of=node_of, moving=np.zeros(node_of.size, bool),
                movable=np.ones(node_of.size, bool), free_slots=free, n_nodes=n_nodes,
                forecast=forecast)


def test_horizon_weights():
    assert horizon_weights(HORIZONS, 1) == (0, 0, 0)
    assert horizon_weights(HORIZONS, 2) == (0, 1, 1.0)
    i0, i1, w = horizon_weights(HORIZONS, 6)
    assert (i0, i1) == (2, 3) and w == pytest.approx(0.2)
    assert horizon_weights(HORIZONS, 11) == (3, 3, 0)
    assert horizon_weights(HORIZONS, 0) == (0, 0, 0)


def test_blend_is_linear_in_horizon():
    f = np.tile(np.array(HORIZONS, dtype=float)[:, None], (1, 3))      # F[h] = h
    for target, expect in [(1, 1.0), (2, 2.0), (3, 3.0), (6, 6.0), (10, 10.0), (11, 10.0)]:
        np.testing.assert_allclose(blend(f, *horizon_weights(HORIZONS, target)), expect)
    hist = np.stack([f, 2 * f])
    np.testing.assert_allclose(blend(hist, *horizon_weights(HORIZONS, 6)), [[6.0] * 3, [12.0] * 3])


def test_p3_reads_completion_horizon_m_plus_d():
    pol = P3Forecast(hysteresis=0.0, m=5, d=1, horizons=HORIZONS)       # target horizon 6
    fc = np.arange(4 * 3, dtype=float).reshape(1, 4, 3)
    expect = 0.8 * fc[0, 2] + 0.2 * fc[0, 3]
    np.testing.assert_allclose(pol.pred_load(None, fc), expect)


def test_p3_requires_forecast_and_waits_for_first_origin():
    pol = P3Forecast(0.0, 2, 1, HORIZONS)
    with pytest.raises(ValueError):
        pol.decide(0, np.zeros((0, 4)), _state([0, 0, 1, 1], forecast=None))
    assert pol.decide(0, np.zeros((0, 4)), _state([0, 0, 1, 1], forecast=np.zeros((0, 4, 4)))) == []


def test_p4_margin_is_residual_quantile_and_zero_without_history():
    d, m = 1, 1                                                          # target horizon 2
    pol = P4Hedged(0.0, m, d, HORIZONS, margin_quantile=0.9, residual_window=5)
    load = np.stack([np.arange(1, 21, dtype=float), np.full(20, 3.0)], axis=1)
    fc = naive_table(load, HORIZONS)
    np.testing.assert_allclose(pol.margin(load, fc), [2.0, 0.0])        # growth 2 per two steps
    np.testing.assert_allclose(pol.margin(load[:2], fc[:2]), 0)         # no observed target yet
    base = P3Forecast(0.0, m, d, HORIZONS).pred_load(load, fc)
    np.testing.assert_allclose(pol.pred_load(load, fc), base + np.array([2.0, 0.0]))


def test_p5_uses_true_load_at_completion_step():
    load = np.zeros((6, 4))
    load[2] = [9.0, 1.0, 1.0, 1.0]
    plan = P5Oracle(0.0, 2, load, 1).decide(0, np.zeros((0, 4)), _state([0, 0, 1, 1]))
    assert plan == [(1, 0, 1)]


def test_simulator_hands_policy_only_forecast_rows_up_to_t_minus_d():
    load, node0, mu = make_case(T=40)
    table = naive_table(load, HORIZONS)
    d = 3
    seen = []

    class Probe:
        def decide(self, t, observed, state):
            fc = state["forecast"]
            seen.append((t, fc.shape[0], observed.shape[0], fc.flags.writeable))
            return []

    simulate(load, node0, mu, m=2, kappa=0.3, budget=4, d=d, k=1, policy=Probe(), forecast=table)
    assert len(seen) == 40
    for t, n_fc, n_obs, writeable in seen:
        assert n_fc == n_obs == max(t - d + 1, 0)
        assert not writeable
    with pytest.raises(ValueError):
        simulate(load, node0, mu, m=2, kappa=0.3, budget=4, d=d, k=1, forecast=table[:-1])


@pytest.mark.parametrize("name", ["P3", "P4"])
def test_forecast_policy_no_lookahead(name):
    load, node0, mu = make_case(T=160, headroom=1.6)
    d, m, t_cut = 3, 3, 80
    corrupt = load.copy()
    corrupt[t_cut:] = np.random.default_rng(99).gamma(2.0, 30.0, size=corrupt[t_cut:].shape)
    params = {"hysteresis": 0.0, "cooldown": 0}

    def run(truth):
        ctx = dict(m=m, d=d, horizons=HORIZONS, load=truth, margin_quantile=0.9, residual_window=20)
        pol = build_policy(name, params, mu, ctx)
        return simulate(truth, node0, mu, m=m, kappa=0.3, budget=4, d=d, k=1, policy=pol,
                        forecast=naive_table(truth, HORIZONS))

    early = lambda r: [mv for mv in r.moves if mv[0] <= t_cut + d - 1]
    r1, r2 = run(load), run(corrupt)
    assert len(early(r1)) > 0
    assert early(r1) == early(r2)


def test_oracle_does_read_the_future():
    load, node0, mu = make_case(T=160, headroom=1.6)
    d, m, t_cut = 3, 3, 80
    corrupt = load.copy()
    corrupt[t_cut:] = np.random.default_rng(99).gamma(2.0, 30.0, size=corrupt[t_cut:].shape)

    def run(truth):
        pol = build_policy("P5", {"hysteresis": 0.0, "cooldown": 0, "oracle_window": 1}, mu, dict(m=m, load=truth))
        return simulate(truth, node0, mu, m=m, kappa=0.3, budget=4, d=d, k=1, policy=pol)

    early = lambda r: [mv for mv in r.moves if mv[0] <= t_cut + d - 1]
    assert early(run(load)) != early(run(corrupt))


def test_forecast_grids_exist_and_are_below_tuning_budget():
    cfg = real_cfg()
    assert set(FORECAST_TUNABLE) == {"P3", "P4", "P4a", "P5"}
    for grid in FORECAST_TUNABLE.values():
        size = int(np.prod([len(cfg[k]) for k in grid.values()]))
        assert size <= cfg["tuning_budget"]


def test_exhaustive_tuning_covers_the_whole_grid():
    cfg, data = small()
    reps = tune_replicates(cfg, data)
    table = naive_table(data.load, cfg["horizons"])
    rows, summary, timing = tune_policy(cfg, "P3", "S", reps, table=table, exhaustive=True,
                                        forecaster="naive")
    size = len(cfg["hysteresis_grid"]) * len(cfg["cooldown_grid"])
    assert len(rows) == len(timing) == summary["n_configs"] == size
    assert len({tuple(sorted(r["params"].items())) for r in rows}) == size
    assert summary["exhaustive"] and summary["converged"] and summary["forecaster"] == "naive"
    fw = forecast_window(table, reps[0])
    assert fw.shape[0] == reps[0].load.shape[0]
    cfg["tuning_budget"] = size + 1
    with pytest.raises(ValueError):
        tune_policy(cfg, "P3", "S", reps, table=table, exhaustive=False)


def test_tune_mase_is_zero_for_a_perfect_forecast_and_ignores_constant_partitions():
    cfg = real_cfg()
    rng = np.random.default_rng(3)
    load = rng.gamma(2.0, 3.0, size=(200, 5))
    load[:, 4] = 7.0                                                    # zero scale, excluded
    perfect = np.stack([np.vstack([load[h:], np.zeros((h, 5))]) for h in cfg["horizons"]], axis=1)
    assert tune_mase(cfg, perfect, load, 80, 160)[0] == pytest.approx(0, abs=1e-12)
    naive = tune_mase(cfg, naive_table(load, cfg["horizons"]), load, 80, 160)[0]
    assert naive > 0


@pytest.mark.parametrize("q", [0.8, 0.9, 0.95, 0.5, 0.0, 1.0])
def test_positive_quantile_matches_nanquantile_reference(q):
    rng = np.random.default_rng(11)
    resid = rng.normal(0.0, 3.0, size=(37, 9))
    resid[:, 2] = -1.0                      # no positive entries
    resid[:5, 3] = np.abs(resid[:5, 3]) + 0.1
    resid[5:, 3] = -1.0                     # a single-sample window of positives
    resid[:1, 4] = 2.0
    resid[1:, 4] = -2.0                     # exactly one positive entry
    ref = np.zeros(9)
    for j in range(9):
        col = resid[:, j][resid[:, j] > 0]
        ref[j] = np.quantile(col, q) if col.size else 0.0
    np.testing.assert_allclose(positive_quantile(resid, q), ref, rtol=1e-12, atol=1e-12)


def test_p5_oracle_window_averages_true_load_from_completion_step():
    load = np.arange(12, dtype=np.float64).reshape(6, 2)
    one = P5Oracle(0.0, 2, load, 1)
    wide = P5Oracle(0.0, 2, load, 3)
    assert np.array_equal(one.target(1), load[3])
    assert np.array_equal(wide.target(1), load[3:6].mean(axis=0))
    assert np.array_equal(wide.target(5), load[5])
    assert np.array_equal(wide.target(0), load[2:5].mean(axis=0))
