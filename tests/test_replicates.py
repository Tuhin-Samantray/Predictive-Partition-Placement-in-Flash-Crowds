import numpy as np
import pytest

from small_world import small
from src.replicates import eligible_events, make_replicate, select_events, tune_replicates


def test_windows_inside_tuning_span_and_contain_event():
    cfg, data = small()
    w = cfg["window_steps"]
    for ev in select_events(cfg, data, "tune", 12):
        rep = make_replicate(cfg, data, ev, 0, cfg["headroom"])
        assert data.train_end <= rep.start and rep.start + w <= data.tune_end
        assert rep.start <= ev["onset"] and ev["end"] <= rep.start + w - 1
        np.testing.assert_array_equal(rep.load, data.load[rep.start:rep.start + w])
        assert rep.load.shape == (w, cfg["n_partitions"])


def test_selection_excludes_ineligible_and_other_splits():
    cfg, data = small()
    elig = eligible_events(cfg, data, "tune")
    ids = set(elig["event_id"])
    assert len(elig) == 12 and "S-14" not in ids
    assert set(elig["split"]) == {"tune"}
    chosen = select_events(cfg, data, "tune", 4)
    assert len({ev["event_id"] for ev in chosen}) == 4
    assert all(ev["event_id"] in ids for ev in chosen)


def test_too_few_events_raises():
    cfg, data = small()
    cfg["tune_replicates"] = 100
    with pytest.raises(ValueError):
        tune_replicates(cfg, data)


def test_replicates_deterministic_and_seed_sensitive():
    cfg, data = small()
    a, b = tune_replicates(cfg, data), tune_replicates(cfg, data)
    assert [r.event_id for r in a] == [r.event_id for r in b]
    for x, y in zip(a, b):
        assert x.start == y.start
        assert x.node0.tobytes() == y.node0.tobytes() and x.mu.tobytes() == y.mu.tobytes()
    ev = select_events(cfg, data, "tune", 1)[0]
    r0 = make_replicate(cfg, data, ev, 0, cfg["headroom"])
    r1 = make_replicate(cfg, data, ev, 1, cfg["headroom"])
    assert r0.node0.tobytes() != r1.node0.tobytes() and r0.mu.tobytes() != r1.mu.tobytes()


def test_capacity_uses_training_prefix_only():
    cfg, data = small()
    ev = select_events(cfg, data, "tune", 1)[0]
    base = make_replicate(cfg, data, ev, 0, cfg["headroom"])
    data.load[data.train_end:] *= 50
    changed = make_replicate(cfg, data, ev, 0, cfg["headroom"])
    assert base.mu.tobytes() == changed.mu.tobytes()
    assert base.node0.tobytes() == changed.node0.tobytes()
    assert not np.array_equal(base.load, changed.load)


def test_initial_placement_balanced_and_jitter_bounds():
    cfg, data = small()
    ev = select_events(cfg, data, "tune", 1)[0]
    rep = make_replicate(cfg, data, ev, 0, cfg["headroom"])
    counts = np.bincount(rep.node0, minlength=cfg["n_nodes"])
    assert (counts == cfg["n_partitions"] // cfg["n_nodes"]).all()
    base = cfg["headroom"] * data.load[: data.train_end].sum(axis=1).mean() / cfg["n_nodes"]
    jit = cfg["capacity_jitter_frac"]
    ratio = rep.mu / base
    assert (ratio >= 1 - jit - 1e-12).all() and (ratio <= 1 + jit + 1e-12).all()