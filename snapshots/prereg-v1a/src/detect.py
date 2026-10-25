import numpy as np
import pandas as pd
from pathlib import Path
import datetime

def detect_events(cfg, dataset):
    L = np.load(Path("data") / "processed" / (dataset + "_L.npy"))
    r = cfg['detector']['r']
    W_b = cfg['detector']['W_b']
    s = cfg['detector']['s']
    rise_lo = cfg['rise_lo']
    rise_hi = cfg['rise_hi']
    
    events = []
    T, P = L.shape
    
    df_L = pd.DataFrame(L)
    rolling_med = df_L.rolling(window=W_b, min_periods=1, closed='left').median().fillna(0.0).values
    is_high = L >= r * rolling_med
    
    for p in range(P):
        high_p = is_high[:, p]
        runs = np.convolve(high_p.astype(int), np.ones(s), mode='valid')
        starts = np.where(runs == s)[0]
        
        i = 0
        while i < len(starts):
            start_t = starts[i]
            end_t = start_t
            while end_t < T and high_p[end_t]:
                end_t += 1
            
            window_data = L[start_t:end_t, p]
            peak = np.max(window_data)
            duration = end_t - start_t
            
            lo_thresh = peak * rise_lo
            hi_thresh = peak * rise_hi
            t_lo = np.argmax(window_data >= lo_thresh)
            t_hi = np.argmax(window_data >= hi_thresh)
            rise_time = t_hi - t_lo if t_hi >= t_lo else 0
            
            events.append({
                'dataset': dataset,
                'event_id': dataset + "_" + str(p) + "_" + str(start_t),
                'partition': p,
                'onset': start_t,
                'peak': peak,
                'rise_time': rise_time,
                'duration': duration
            })
            
            while i < len(starts) and starts[i] < end_t:
                i += 1
                
    df_events = pd.DataFrame(events)
    out_file = Path("data") / "processed" / (dataset + "_events.csv")
    df_events.to_csv(out_file, index=False)
    
    if len(events) > 0:
        med_rise = df_events['rise_time'].median()
        print(dataset + " Found " + str(len(events)) + " events. Median rise: " + str(med_rise))
        amd_path = Path("AMENDMENTS.md")
        if amd_path.exists():
            dt = datetime.datetime.now().isoformat()
            with open(amd_path, "a") as f:
                f.write("\n| auto | " + dt + " | " + dataset + " onset cut point | median rise " + str(med_rise) + " | yes | N/A |")
