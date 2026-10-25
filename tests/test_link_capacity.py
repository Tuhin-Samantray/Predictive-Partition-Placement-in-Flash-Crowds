import numpy as np
import pytest

from src.layers import charge_links, link_capacity, migration_rates, partition_sizes


def test_link_capacity_formula():
    train = np.array([[4.0, 6.0], [5.0, 5.0], [10.0, 0.0], [2.0, 8.0]])
    assert link_capacity(train, 2.0, 2) == pytest.approx(10.0)
    assert link_capacity(train, 4.0, 4) == pytest.approx(10.0)


def test_partition_sizes_mean_and_determinism():
    s1 = partition_sizes(512, 5, 0.25, 100.0, 0.5, 11)
    s2 = partition_sizes(512, 5, 0.25, 100.0, 0.5, 11)
    s3 = partition_sizes(512, 5, 0.25, 100.0, 0.5, 12)
    assert (s1 == s2).all() and not (s1 == s3).all()
    assert (s1 > 0).all()
    assert s1.mean() == pytest.approx(5 * 0.25 * 100.0)


def test_partition_sizes_without_spread_are_constant():
    s = partition_sizes(16, 5, 0.25, 100.0, 0.0, 1)
    assert np.allclose(s, 5 * 0.25 * 100.0)


def test_single_move_without_foreground_runs_at_the_cap():
    rates = migration_rates([0], [1], np.zeros(3), 100.0, 0.25)
    assert rates.tolist() == [25.0]


def test_moves_from_one_source_share_its_link():
    rates = migration_rates([0, 0], [1, 2], np.zeros(3), 100.0, 0.8)
    assert rates.tolist() == [50.0, 50.0]


def test_foreground_traffic_leaves_less_for_migration():
    rates = migration_rates([0], [1], np.array([90.0, 0.0]), 100.0, 0.8)
    assert rates.tolist() == [10.0]


def test_saturated_link_stops_the_move():
    rates = migration_rates([0], [1], np.array([120.0, 0.0]), 100.0, 0.8)
    assert rates.tolist() == [0.0]


def test_migration_charge_never_exceeds_the_free_capacity():
    rng = np.random.default_rng(7)
    n_nodes, link = 6, 100.0
    for _ in range(50):
        fg = rng.uniform(0, 120, n_nodes)
        k = int(rng.integers(1, 10))
        src = rng.integers(0, n_nodes, k)
        dst = (src + rng.integers(1, n_nodes, k)) % n_nodes
        rates = migration_rates(src, dst, fg, link, 0.6)
        charge = charge_links(src, dst, rates, n_nodes)
        free = np.maximum(0, link - fg)
        assert (rates >= 0).all()
        assert (charge <= free + 1e-9).all()
        assert (np.minimum(fg, link) + charge <= link + 1e-9).all()


def test_no_moves_in_flight_gives_no_rates_and_no_charge():
    rates = migration_rates([], [], np.zeros(4), 100.0, 0.25)
    assert rates.size == 0
    assert charge_links([], [], rates, 4).tolist() == [0.0, 0.0, 0.0, 0.0]
