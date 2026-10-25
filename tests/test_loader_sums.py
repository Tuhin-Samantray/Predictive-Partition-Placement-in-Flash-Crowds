import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zstandard

from src import loaders
from src.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def small_cfg():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg.update(n_partitions=4, azure_days=2, azure_mins_per_day=6, azure_chunk_rows=3,
               d2_chunk_rows=5, d1_train_days=1, d1_tune_days=1)
    return cfg


def write_azure(folder, cfg):
    rng = np.random.default_rng(7)
    n = 11
    for day in range(1, cfg["azure_days"] + 1):
        ids = pd.DataFrame({
            "HashOwner": [f"o{i}" for i in range(n)],
            "HashApp": [f"a{i % 3}" for i in range(n)],
            "HashFunction": [f"f{day}{i}" for i in range(n)],
        })
        inv = ids.copy()
        inv["Trigger"] = "http"
        counts = rng.integers(0, 50, size=(n, cfg["azure_mins_per_day"]))
        for minute in range(cfg["azure_mins_per_day"]):
            inv[str(minute + 1)] = counts[:, minute]
        inv.to_csv(folder / cfg["azure_invocation_pattern"].format(day=day), index=False)
        dur = ids.copy()
        dur["Average"] = rng.uniform(1, 100, size=n)
        dur["Count"] = rng.integers(1, 20, size=n)
        dur.to_csv(folder / cfg["azure_duration_pattern"].format(day=day), index=False)


def write_d2(path, rows):
    text = "\n".join(",".join(str(v) for v in row) for row in rows) + "\n"
    path.write_bytes(zstandard.ZstdCompressor().compress(text.encode("utf-8")))


def make_rows():
    rng = np.random.default_rng(11)
    rows = []
    for i in range(23):
        rows.append([i * 4, f"key{rng.integers(0, 6)}", int(rng.integers(5, 20)),
                     int(rng.integers(0, 300)), f"c{i % 2}", "get" if i % 2 else "set", 0])
    return rows


def test_loader_sums(tmp_path):
    cfg = small_cfg()
    write_azure(tmp_path, cfg)
    first = loaders.load_d1(cfg, tmp_path)
    again = loaders.load_d1(cfg, tmp_path)
    expected = np.concatenate([
        pd.read_csv(tmp_path / cfg["azure_invocation_pattern"].format(day=d))
        .iloc[:, cfg["azure_skip_cols"]:].sum(axis=0).to_numpy(dtype=float)
        for d in range(1, cfg["azure_days"] + 1)])
    assert first["L"].shape == (cfg["azure_days"] * cfg["azure_mins_per_day"], cfg["n_partitions"])
    assert np.array_equal(first["L"].sum(axis=1), expected)
    assert np.array_equal(first["L"], again["L"])
    assert np.array_equal(first["Bytes"], again["Bytes"])
    assert (first["Bytes"] > 0).any()
    key = "o1|a1|f11"
    manual = int(hashlib.md5((cfg["hash_seed"] + cfg["key_sep"] + key).encode()).hexdigest(), 16) % cfg["n_partitions"]
    assert loaders.partition_ids([key], cfg["n_partitions"], cfg["hash_seed"], cfg["key_sep"])[0] == manual
    dur = pd.read_csv(tmp_path / cfg["azure_duration_pattern"].format(day=1))
    part = loaders.partition_ids(loaders.make_keys(dur, cfg["azure_id_cols"], cfg["key_sep"]),
                                 cfg["n_partitions"], cfg["hash_seed"], cfg["key_sep"])
    count = dur["Count"].to_numpy(dtype=float)
    weighted = dur["Average"].to_numpy(dtype=float) * count
    for p in range(cfg["n_partitions"]):
        mask = part == p
        want = weighted[mask].sum() / count[mask].sum() if mask.any() else cfg["demand_fallback"]
        assert np.isclose(first["demand"][p], want)


def test_loader_sums_d2(tmp_path):
    cfg = small_cfg()
    rows = make_rows()
    path = tmp_path / "d2.zst"
    write_d2(path, rows)
    result = loaders.load_d2(cfg, path, cfg["d2_step_seconds"])
    steps = np.array([row[0] for row in rows]) // cfg["d2_step_seconds"]
    assert result["L"].shape == (steps.max() + 1, cfg["n_partitions"])
    assert result["L"].sum() == len(rows)
    assert result["Bytes"].sum() == sum(row[2] + row[3] for row in rows)
    for step in range(steps.max() + 1):
        assert result["L"][step].sum() == (steps == step).sum()
    assert result["stats"]["ordered"] is True


def test_loader_rejects_unordered_d2(tmp_path):
    cfg = small_cfg()
    path = tmp_path / "d2.zst"
    write_d2(path, list(reversed(make_rows())))
    with pytest.raises(ValueError):
        loaders.load_d2(cfg, path, cfg["d2_step_seconds"])