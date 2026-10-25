"""RV4 logging (A43): logging changes no behavioural metric, the replay agrees with the live P4a
counters, and the trigger counts reconcile with the simulator."""
from pathlib import Path

import numpy as np

from src.config import load_config
from src.layer_runs import CrossData, run_replicate_layers
from src.layers import link_capacity, partition_sizes
from src.policies import build_policy
from src.replicates import Replicate, run_replicate
from src.rv4 import LoggedPolicy, rv4_fields
from src.simulator import node_capacity, simulate

ROOT = Path(__file__).parent.parent
STEPS = 120
PARAMS = {
    "P1": {"ewma_alpha": 0.3, "theta_hi": 0.8, "cooldown": 1},
    "P4": {"hysteresis": 0.05, "cooldown": 1},
    "P4a": {"hysteresis": 0.05, "cooldown": 1},
}
TRIG = {"trig_decisions", "trig_proposed", "trig_proposed_steps", "trig_no_slot_steps"}


def make_case():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    n_nodes = cfg["n_nodes"]
    n_part = 4 * n_nodes
    rng = np.random.default_rng(7)
    node0 = np.arange(n_part) % n_nodes
    load = rng.gamma(2.0, 1.0, (STEPS, n_part)) * (1 + 3 * (node0 == 0))
    noise = np.abs(1 + 0.1 * rng.standard_normal((STEPS, len(cfg["horizons"]), n_part)))
    fc = load[:, None, :] * noise
    mu = node_capacity(load, cfg["headroom"], n_nodes)
    rep = Replicate("ev", "D1", "tune", "x", 0, 0, STEPS - 1, 0, load, node0, mu)
    return cfg, rep, fc


def ctx_of(cfg, rep, m):
    return {"m": m, "d": cfg["monitor_delay_d"], "horizons": cfg["horizons"], "load": rep.load,
            "margin_quantile": cfg["margin_quantile"], "residual_window": cfg["residual_window"],
            "gamma": cfg["gamma"], "a_min": cfg["a_min"], "a_max": cfg["a_max"]}


def test_logging_does_not_change_behaviour():
    cfg, rep, fc = make_case()
    for name, params in PARAMS.items():
        off = run_replicate(cfg, rep, name, params, forecast=fc)
        on = run_replicate(cfg, rep, name, params, forecast=fc, log=True)
        assert not any(k.startswith(("rv4_", "trig_")) for k in off)
        assert all(on[k] == off[k] for k in off)
        assert TRIG <= set(on)
        assert any(k.startswith("rv4_") for k in on) == (name != "P1")


def test_adaptive_replay_matches_live_counters():
    cfg, rep, fc = make_case()
    m = cfg["migration_steps_m0"]
    d = cfg["monitor_delay_d"]
    ctx = ctx_of(cfg, rep, m)
    pol = build_policy("P4a", PARAMS["P4a"], rep.mu, ctx)
    logged = LoggedPolicy(pol)
    res = simulate(rep.load, rep.node0, rep.mu, m=m, kappa=cfg["migration_overhead_kappa"],
                   budget=cfg["migration_budget_B"], d=d, k=1, policy=logged,
                   cooldown=PARAMS["P4a"]["cooldown"], forecast=fc)
    n_live = STEPS - d
    fields = rv4_fields(rep.load[:n_live], fc[:n_live], ctx, cfg["margin_quantile"])
    assert fields["rv4_ad_hits"] == pol.hits
    assert fields["rv4_trials"] == pol.trials
    assert abs(fields["rv4_ad_alpha_last"] - pol.alpha_trace[fields["rv4_origins"] - 1]) < 1e-12
    assert logged.proposed == len(res.moves) + res.dropped_moves
    assert logged.decisions == STEPS


class Stub:
    def decide(self, t, observed, state):
        src = int(state["node_of"][0])
        if not state["movable"][0]:
            return []
        return [(0, src, (src + 1) % state["n_nodes"])]


def test_trigger_counts_reconcile():
    cfg, rep, fc = make_case()
    k = cfg["decision_interval_k"]
    logged = LoggedPolicy(Stub())
    res = simulate(rep.load, rep.node0, rep.mu, m=cfg["migration_steps_m0"], kappa=cfg["migration_overhead_kappa"],
                   budget=cfg["migration_budget_B"], d=cfg["monitor_delay_d"], k=k, policy=logged)
    assert logged.decisions == len(range(0, STEPS, k))
    assert logged.proposed == len(res.moves) + res.dropped_moves
    assert 0 < logged.proposed_steps <= logged.decisions
    assert logged.no_slot_steps <= logged.decisions


def test_layers_logging_does_not_change_behaviour():
    cfg, rep, fc = make_case()
    n_part = rep.load.shape[1]
    rng = np.random.default_rng(11)
    nbytes = rng.gamma(2.0, 1.0, (STEPS, n_part))
    link = link_capacity(nbytes, cfg["h_net"], cfg["n_nodes"])
    sizes = partition_sizes(n_part, cfg["migration_steps_m0"], cfg["rho"], link, cfg["sigma_size"], 1)
    cross = CrossData(nbytes, np.ones(n_part), link, sizes, cfg["h_net"])
    off = run_replicate_layers(cfg, rep, cross, "P4a", PARAMS["P4a"], forecast=fc)
    on = run_replicate_layers(cfg, rep, cross, "P4a", PARAMS["P4a"], forecast=fc, log=True)
    assert all(on[k] == off[k] for k in off)
    assert TRIG <= set(on)
    assert "rv4_ad_cov" in on and "rv4_fix_cov" in on
