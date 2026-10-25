from src.config import load_config

WANT = {"h2_split": 0.5, "h3_ub": 0.2, "h4_rho": 0.5, "h4_ub": 0.7, "h5_ratio": 1.25, "h6_tau": 0.6,
        "h9_share": 0.05, "alpha_adj": 0.05, "h12_noninf": 0.02}


def test_threshold_keys_hold_the_registered_values():
    cfg = load_config("configs/default.yaml")
    for key, value in WANT.items():
        assert cfg[key] == value, key
    assert 0 < cfg["alpha_adj"] < 1 and cfg["h4_rho"] < cfg["h4_ub"]
