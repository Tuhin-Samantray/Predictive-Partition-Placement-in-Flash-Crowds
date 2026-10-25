import numpy as np
import pandas as pd
import hashlib
import zstandard as zstd
import io
from pathlib import Path

def get_p(func_id, p, seed):
    h = hashlib.md5(str(seed).encode() + str(func_id).encode()).hexdigest()
    return int(h, 16) % p

def load_d1(cfg):
    p = cfg['n_partitions']
    seed = cfg['hash_seed']
    days = cfg['azure_days']
    mins = cfg['azure_mins_per_day']
    skip = cfg['azure_skip_cols']
    data_root = Path(cfg.get('data_root', 'data'))
    
    L = np.zeros((days * mins, p), dtype=np.float32)
    data_dir = data_root / "raw" / "azure2019"
    out_dir = data_root / "processed"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    for day in range(1, days + 1):
        fname = "invocations_per_function_md.anon.d" + str(day).zfill(2) + ".csv"
        files = list(data_dir.glob(fname))
        if not files:
            continue
        df = pd.read_csv(files[0])
        func_ids = df['HashFunction'].astype(str).values
        part_idx = np.array([get_p(fid, p, seed) for fid in func_ids])
        loads = df.iloc[:, skip:].fillna(0.0).values
        
        start_idx = (day - 1) * mins
        end_idx = day * mins
        for i in range(len(func_ids)):
            L[start_idx:end_idx, part_idx[i]] += loads[i]
            
    np.save(out_dir / "D1_L.npy", L)
    rng = np.random.default_rng(cfg['master_seed'])
    Bytes = L * rng.lognormal(mean=0.0, sigma=cfg['sigma_size'], size=L.shape)
    np.save(out_dir / "D1_Bytes.npy", Bytes)
    demand = np.ones(p, dtype=np.float32)
    np.save(out_dir / "D1_demand.npy", demand)

def load_d2(cfg):
    p = cfg['n_partitions']
    seed = cfg['hash_seed']
    step_sec = cfg['step_seconds']
    min_cols = cfg['d2_min_cols']
    data_root = Path(cfg.get('data_root', 'data'))
    out_dir = data_root / "processed"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    data_dir = data_root / "raw" / "twitter"
    files = list(data_dir.glob("*.zst"))
    if not files:
        return
    
    max_steps = cfg['window_steps'] * 2
    L = np.zeros((max_steps, p), dtype=np.float32)
    Bytes = np.zeros((max_steps, p), dtype=np.float32)
    
    with open(files[0], 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            for line in text_stream:
                parts = line.strip().split(',')
                if len(parts) < min_cols:
                    continue
                try:
                    ts = int(parts[0])
                    key = parts[1]
                    k_sz = int(parts[2])
                    v_sz = int(parts[3])
                    
                    step = ts // step_sec
                    if step >= max_steps:
                        pad = np.zeros((max_steps, p), dtype=np.float32)
                        L = np.vstack([L, pad])
                        Bytes = np.vstack([Bytes, pad])
                        max_steps *= 2
                    
                    part = get_p(key, p, seed)
                    L[step, part] += 1.0
                    Bytes[step, part] += (k_sz + v_sz)
                except ValueError:
                    continue
                    
    np.save(out_dir / "D2_L.npy", L)
    np.save(out_dir / "D2_Bytes.npy", Bytes)
    demand = cfg['c0'] + cfg['c1'] * np.mean(Bytes, axis=0)
    np.save(out_dir / "D2_demand.npy", demand)
