import numpy as np
import pytest

from src.policies import build_policy
from src.simulator import simulate
from stub_policy import make_case

CASES = [
    ("P1", {"ewma_alpha": 0.5, "theta_hi": 0.8, "cooldown": 0}),
    ("P2", {"ewma_alpha": 0.5, "hysteresis": 0.1, "cooldown": 0}),
]


@pytest.mark.parametrize("name,params", CASES)
def test_reactive_no_lookahead(name, params):
    load, node0, mu = make_case(T=160, headroom=1.6)
    d, t_cut = 3, 80
    corrupt = load.copy()
    corrupt[t_cut:] = np.random.default_rng(99).gamma(2.0, 30.0, size=corrupt[t_cut:].shape)
    kw = dict(m=3, kappa=0.3, budget=4, d=d, k=1, cooldown=params["cooldown"])
    r1 = simulate(load, node0, mu, policy=build_policy(name, params, mu), **kw)
    r2 = simulate(corrupt, node0, mu, policy=build_policy(name, params, mu), **kw)
    # a decision at t reads load up to t - d, so moves started up to t_cut + d - 1 cannot differ
    early = lambda r: [mv for mv in r.moves if mv[0] <= t_cut + d - 1]
    assert len(early(r1)) > 0
    assert early(r1) == early(r2)