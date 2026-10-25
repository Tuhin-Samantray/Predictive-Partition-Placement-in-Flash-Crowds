import numpy as np
import pytest

from small_world import real_cfg
from src.forecasters import BASELINES, LEARNED, build_table, day_steps
from src.policies import Ewma

KINDS = BASELINES + LEARNED
HP = {"ridge": 1.0, "lightgbm": 15}


def world():
    cfg = real_cfg()
    cfg.update(lookback_L=12, roll_window=5, refit_interval=40, origin_stride=2,
               lgbm_trees=5, feature_chunk_origins=16)
    load = np.random.default_rng(3).gamma(2.0, 3.0, size=(300, 6))
    return cfg, load, 150, 60


def table(cfg, kind, load, train_end, day):
    return build_table(cfg, "S", kind, load, train_end, day, hp=HP.get(kind), n_jobs=1)


def test_every_configured_forecaster_is_built():
    cfg, load, train_end, day = world()
    assert set(cfg["forecaster_names"]) == set(KINDS)
    for kind in KINDS:
        f = table(cfg, kind, load, train_end, day)
        assert f.shape == (300, len(cfg["horizons"]), 6) and f.dtype == np.float32
        assert np.isfinite(f).all() and (f >= 0).all()


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("cut", [160, 200, 233])
def test_forecast_no_leak(kind, cut):
    cfg, load, train_end, day = world()
    other = load.copy()
    other[cut + 1:] = np.random.default_rng(99).gamma(2.0, 30.0, size=other[cut + 1:].shape)
    a = table(cfg, kind, load, train_end, day)
    b = table(cfg, kind, other, train_end, day)
    np.testing.assert_array_equal(a[:cut + 1], b[:cut + 1])
    assert not np.array_equal(a[cut + 1:], b[cut + 1:])


def test_baseline_values():
    cfg, load, train_end, day = world()
    hs = cfg["horizons"]
    f = table(cfg, "naive", load, train_end, day)
    np.testing.assert_allclose(f[:, 0, :], load, rtol=1e-6)
    s = table(cfg, "seasonal_naive", load, train_end, day)
    for o in (10, 70, 200):
        for i, h in enumerate(hs):
            want = load[o + h - day] if o + h - day >= 0 else load[o]
            np.testing.assert_allclose(s[o, i], want, rtol=1e-6)
    e = table(cfg, "ewma", load, train_end, day)
    ew = Ewma(cfg["forecast_ewma_alpha"])
    ew.update(load[:121])
    np.testing.assert_allclose(e[120, 0], ew.value, rtol=1e-5)


@pytest.mark.parametrize("kind", LEARNED)
def test_learned_forecasters_recover_a_constant(kind):
    cfg, _, train_end, day = world()
    flat = np.full((300, 6), 5.0)
    f = table(cfg, kind, flat, train_end, day)
    np.testing.assert_allclose(f[train_end - 1:], 5.0, atol=0.1)


def test_day_steps_match_config():
    cfg = real_cfg()
    assert day_steps(cfg, "D1") == cfg["seasonal_period"]
    assert day_steps(cfg, "D2") == cfg["day_seconds"] // cfg["d2_step_seconds"]