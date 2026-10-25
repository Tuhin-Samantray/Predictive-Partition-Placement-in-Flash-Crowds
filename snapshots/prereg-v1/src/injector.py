import numpy as np
from pathlib import Path

def inject_spikes(cfg, dataset):
    try:
        L = np.load(Path("data") / "processed" / (dataset + "_L.npy"))
    except FileNotFoundError:
        return
    rng = np.random.default_rng(cfg['master_seed'])
    
    T, P = L.shape
    D3_L = L.copy()
    
    num_spikes = cfg['min_events']
    min_w = cfg['detector']['W_b']
    
    for _ in range(num_spikes):
        p = rng.integers(0, P)
        start_t = rng.integers(min_w, T - cfg['injection_dur_max'])
        
        amp = rng.uniform(cfg['injection_amplitude_min'], cfg['injection_amplitude_max'])
        rise = rng.integers(cfg['injection_rise_min'], cfg['injection_rise_max'])
        dur = rng.integers(cfg['injection_dur_min'], cfg['injection_dur_max'])
        
        for step in range(rise):
            D3_L[start_t + step, p] += amp * (step / rise)
        
        for step in range(rise, dur):
            D3_L[start_t + step, p] += amp * (1.0 - (step - rise) / (dur - rise))
                
    np.save(Path("data") / "processed" / (dataset + "_D3_L.npy"), D3_L)
