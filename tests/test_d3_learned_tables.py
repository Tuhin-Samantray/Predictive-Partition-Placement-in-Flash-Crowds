import numpy as np

from src.config import load_config
from src.d3 import load_d3
from src.replicates import load_forecast


def tables():
    cfg = load_config("configs/default.yaml")
    out = []
    for base in ("D1", "D2"):
        data = load_d3(cfg, base)
        for kind in cfg["d3_learned_forecasters"]:
            out.append((cfg, data, load_forecast(cfg, data.name, kind)))
    return out


def test_d3_learned_tables_shape_dtype_and_values():
    got = tables()
    assert len(got) == 2
    for cfg, data, table in got:
        assert table.shape == (data.n_steps, len(cfg["horizons"]), data.load.shape[1])
        assert table.dtype == np.float32
        assert np.isfinite(table).all() and (table >= 0).all()


def test_d3_learned_tables_carry_the_naive_forecast_before_the_first_refit():
    for cfg, data, table in tables():
        lo = data.train_end - 1
        assert np.array_equal(table[:lo, 0], np.asarray(data.load[:lo], dtype=np.float32))
