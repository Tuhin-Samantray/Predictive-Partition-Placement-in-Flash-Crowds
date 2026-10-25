import numpy as np

import src.replicates as rp
from src.config import load_config
from src.replicates import Replicate, run_replicate


def world(cfg):
    rng = np.random.default_rng(1)
    n_part, n_nodes, steps = cfg["n_partitions"], cfg["n_nodes"], 60
    load = rng.gamma(2.0, 1.0, size=(steps, n_part))
    node0 = rng.permutation(np.arange(n_part) % n_nodes)
    mu = np.full(n_nodes, load.sum(axis=1).mean() / n_nodes * cfg["headroom"])
    return Replicate("e", "DX", "tune", "c", 0, 10, 20, 0, load, node0, mu)


def test_defaults_are_unchanged_and_overrides_reach_the_simulator_and_policy(monkeypatch):
    cfg = load_config("configs/default.yaml")
    rep = world(cfg)
    params = {"ewma_alpha": 0.3, "theta_hi": 0.9, "cooldown": 0}
    base = run_replicate(cfg, rep, "P1", params)
    assert base == run_replicate(cfg, rep, "P1", params, budget=cfg["migration_budget_B"], lead=None,
                                 margin_quantile=cfg["margin_quantile"])
    seen = {}
    real_sim, real_build = rp.simulate, rp.build_policy

    def spy_sim(*args, **kwargs):
        seen["budget"] = kwargs["budget"]
        seen["m_sim"] = kwargs["m"]
        return real_sim(*args, **kwargs)

    def spy_build(name, params, mu, ctx=None):
        seen["ctx"] = dict(ctx)
        return real_build(name, params, mu, ctx)

    monkeypatch.setattr(rp, "simulate", spy_sim)
    monkeypatch.setattr(rp, "build_policy", spy_build)
    run_replicate(cfg, rep, "P1", params, budget=1, lead=1, margin_quantile=0.8)
    assert seen["budget"] == 1
    assert seen["m_sim"] == cfg["migration_steps_m0"]
    assert seen["ctx"]["m"] == 1 and seen["ctx"]["margin_quantile"] == 0.8
    run_replicate(cfg, rep, "P1", params)
    assert seen["ctx"]["m"] == cfg["migration_steps_m0"]
    assert seen["ctx"]["margin_quantile"] == cfg["margin_quantile"]
    assert seen["budget"] == cfg["migration_budget_B"]
