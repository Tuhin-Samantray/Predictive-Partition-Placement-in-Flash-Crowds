"""Data layer: Azure Functions 2019 (D1) and Twitter cache trace (D2) to L[t, p]."""
import io
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import zstandard
from scipy import sparse
from scipy.stats import norm

from src.config import split_points
from src.seeds import md5_int


def make_keys(frame, cols, sep):
    key = frame[cols[0]].astype(str)
    for col in cols[1:]:
        key = key + sep + frame[col].astype(str)
    return key.tolist()


def partition_ids(keys, n_partitions, hash_seed, sep):
    keys = list(keys)
    return np.fromiter(
        (md5_int(hash_seed + sep + key) % n_partitions for key in keys),
        dtype=np.int64,
        count=len(keys),
    )


def unit_mean_lognormal(keys, master_seed, sigma, hash_bits, label, sep):
    """Seeded lognormal weight per key with mean 1, independent of row order."""
    keys = list(keys)
    scale = 2 ** hash_bits + 1
    u = np.fromiter(
        ((md5_int(sep.join((str(master_seed), label, key))) + 1) / scale for key in keys),
        dtype=np.float64,
        count=len(keys),
    )
    return np.exp(sigma * norm.ppf(u) - sigma ** 2 / 2)


def d1_demand(cfg, raw_dir):
    """Per-partition mean service time from the duration files of the training days."""
    n_part = cfg["n_partitions"]
    weighted = np.zeros(n_part)
    counts = np.zeros(n_part)
    cols = list(cfg["azure_id_cols"]) + [cfg["azure_dur_avg_col"], cfg["azure_dur_count_col"]]
    for day in range(1, cfg["d1_train_days"] + 1):
        path = Path(raw_dir) / cfg["azure_duration_pattern"].format(day=day)
        frame = pd.read_csv(path, usecols=cols)
        keys = make_keys(frame, cfg["azure_id_cols"], cfg["key_sep"])
        part = partition_ids(keys, n_part, cfg["hash_seed"], cfg["key_sep"])
        avg = frame[cfg["azure_dur_avg_col"]].to_numpy(dtype=np.float64)
        cnt = frame[cfg["azure_dur_count_col"]].to_numpy(dtype=np.float64)
        weighted += np.bincount(part, weights=avg * cnt, minlength=n_part)
        counts += np.bincount(part, weights=cnt, minlength=n_part)
    has = counts > 0
    demand = np.where(has, weighted / np.where(has, counts, 1), cfg["demand_fallback"])
    if not np.isfinite(demand).all():
        raise AssertionError("D1 demand contains non-finite values")
    return demand, int((~has).sum())


def verify_memory_files(cfg, raw_dir):
    """Application-memory files: schema check only; never used in a result."""
    expected = list(cfg["azure_memory_cols"])
    out = {}
    for day in range(1, cfg["azure_days"] + 1):
        path = Path(raw_dir) / cfg["azure_memory_pattern"].format(day=day)
        head = pd.read_csv(path, nrows=0)
        with open(path, "rb") as handle:
            lines = sum(1 for _ in handle)
        out[path.name] = {"columns_match": list(head.columns) == expected, "rows": lines - 1}
    return out


def load_d1(cfg, raw_dir):
    raw_dir = Path(raw_dir)
    n_part = cfg["n_partitions"]
    mins = cfg["azure_mins_per_day"]
    skip = cfg["azure_skip_cols"]
    steps = cfg["azure_days"] * mins
    load = np.zeros((steps, n_part))
    nbytes = np.zeros((steps, n_part))
    raw_sums = np.zeros(steps)
    rows_total = 0
    for day in range(1, cfg["azure_days"] + 1):
        path = raw_dir / cfg["azure_invocation_pattern"].format(day=day)
        base = (day - 1) * mins
        for chunk in pd.read_csv(path, chunksize=cfg["azure_chunk_rows"]):
            if chunk.shape[1] != skip + mins:
                raise ValueError(f"{path.name}: expected {skip + mins} columns, got {chunk.shape[1]}")
            if list(chunk.columns[:skip]) != list(cfg["azure_head_cols"]):
                raise ValueError(f"{path.name}: unexpected leading columns {list(chunk.columns[:skip])}")
            counts = chunk.iloc[:, skip:].to_numpy(dtype=np.float64)
            if not np.isfinite(counts).all():
                raise ValueError(f"{path.name}: non-finite counts")
            keys = make_keys(chunk, cfg["azure_id_cols"], cfg["key_sep"])
            part = partition_ids(keys, n_part, cfg["hash_seed"], cfg["key_sep"])
            weight = unit_mean_lognormal(
                keys, cfg["master_seed"], cfg["d1_bytes_sigma"], cfg["hash_bits"],
                cfg["d1_bytes_label"], cfg["key_sep"])
            rows = np.arange(len(keys))
            pick = sparse.csr_matrix((np.ones(len(keys)), (part, rows)), shape=(n_part, len(keys)))
            pick_w = sparse.csr_matrix((weight, (part, rows)), shape=(n_part, len(keys)))
            load[base:base + mins] += (pick @ counts).T
            nbytes[base:base + mins] += (pick_w @ counts).T
            raw_sums[base:base + mins] += counts.sum(axis=0)
            rows_total += len(keys)
        print(f"D1 day {day}: rows so far {rows_total}", flush=True)
    if not np.array_equal(load.sum(axis=1), raw_sums):
        raise AssertionError("D1: per-step sum of L differs from the raw column sums")
    demand, missing = d1_demand(cfg, raw_dir)
    stats = {
        "rows_total": rows_total,
        "n_steps": steps,
        "total_invocations": float(load.sum()),
        "partitions_without_load": int((load.sum(axis=0) == 0).sum()),
        "partitions_without_duration": missing,
        "bytes_total": float(nbytes.sum()),
    }
    return {"L": load, "Bytes": nbytes, "demand": demand, "stats": stats}


def _grow(array, rows, n_part):
    out = np.zeros((rows, n_part))
    out[:array.shape[0]] = array
    return out


def load_d2(cfg, path, step_len):
    n_part = cfg["n_partitions"]
    cols = list(cfg["d2_cols"])
    if len(cols) < cfg["d2_min_cols"]:
        raise ValueError("d2_cols is shorter than d2_min_cols")
    tcol, kcol = cfg["d2_time_col"], cfg["d2_key_col"]
    ksz, vsz, ocol = cfg["d2_ksize_col"], cfg["d2_vsize_col"], cfg["d2_op_col"]
    load = np.zeros((0, n_part))
    nbytes = load.copy()
    vbytes = load.copy()
    t0 = None
    prev_last = None
    ordered = True
    n_steps = 0
    rows_total = 0
    bytes_total = 0.0
    ops = Counter()
    zero_value = Counter()
    with open(path, "rb") as handle:
        reader = zstandard.ZstdDecompressor().stream_reader(handle, read_across_frames=True)
        text = io.TextIOWrapper(reader, encoding="utf-8")
        frames = pd.read_csv(
            text, header=None, names=cols, usecols=[tcol, kcol, ksz, vsz, ocol],
            dtype={kcol: str, ocol: str}, chunksize=cfg["d2_chunk_rows"])
        for number, chunk in enumerate(frames):
            ts = chunk[tcol].to_numpy(dtype=np.int64)
            if t0 is None:
                t0 = int(ts.min())
            if (np.diff(ts) < 0).any() or (prev_last is not None and ts[0] < prev_last):
                ordered = False
            prev_last = int(ts[-1])
            step = (ts - t0) // step_len
            if step.min() < 0:
                raise ValueError("D2 timestamps fall before the first chunk: file is not ordered")
            codes, uniques = pd.factorize(chunk[kcol])
            if (codes < 0).any():
                raise ValueError("D2 chunk has a missing key")
            part = partition_ids(list(uniques), n_part, cfg["hash_seed"], cfg["key_sep"])[codes]
            ks = chunk[ksz].to_numpy(dtype=np.float64)
            vs = chunk[vsz].to_numpy(dtype=np.float64)
            lo, hi = int(step.min()), int(step.max())
            span = hi - lo + 1
            if hi + 1 > load.shape[0]:
                rows = max(hi + 1, load.shape[0] * 2)
                load, nbytes, vbytes = (_grow(a, rows, n_part) for a in (load, nbytes, vbytes))
            flat = (step - lo) * n_part + part
            size = span * n_part
            load[lo:hi + 1] += np.bincount(flat, minlength=size).reshape(span, n_part)
            nbytes[lo:hi + 1] += np.bincount(flat, weights=ks + vs, minlength=size).reshape(span, n_part)
            vbytes[lo:hi + 1] += np.bincount(flat, weights=vs, minlength=size).reshape(span, n_part)
            n_steps = max(n_steps, hi + 1)
            rows_total += len(chunk)
            bytes_total += float((ks + vs).sum())
            ops.update(chunk[ocol].value_counts().to_dict())
            zero_value.update(chunk.loc[chunk[vsz] == 0, ocol].value_counts().to_dict())
            print(f"D2 chunk {number}: rows {rows_total}, steps {n_steps}", flush=True)
    load, nbytes, vbytes = load[:n_steps], nbytes[:n_steps], vbytes[:n_steps]
    if load.sum() != rows_total:
        raise AssertionError("D2: sum of L differs from the number of rows read")
    if nbytes.sum() != bytes_total:
        raise AssertionError("D2: sum of Bytes differs from the key plus value sizes read")
    train_end = split_points(cfg, "D2", n_steps)[0]
    count_train = load[:train_end].sum(axis=0)
    value_train = vbytes[:train_end].sum(axis=0)
    mean_value = np.where(
        count_train > 0,
        value_train / np.maximum(count_train, 1),
        value_train.sum() / max(count_train.sum(), 1))
    demand = cfg["c0"] + cfg["c1"] * mean_value
    stats = {
        "rows_total": rows_total,
        "n_steps": n_steps,
        "step_seconds": step_len,
        "first_timestamp": t0,
        "ordered": ordered,
        "bytes_total": bytes_total,
        "ops": dict(ops),
        "zero_value_size_by_op": dict(zero_value),
    }
    return {"L": load, "Bytes": nbytes, "demand": demand, "stats": stats}