"""Flash-crowd detector (Phase 3.2).

An event is a run of at least s consecutive steps in which a partition's load is at least
r times the larger of its trailing median over W_b steps (window ends at t - 1) and a floor.
"""
import numpy as np
import pandas as pd

ROW_COLUMNS = ["partition", "onset", "peak_step", "end", "duration", "peak", "baseline", "rise_time"]


def detect_events(load, *, dataset, r, w_b, s, floor, rise_lo, rise_hi, train_end, tune_end):
    n_part = load.shape[1]
    med = pd.DataFrame(load).rolling(window=w_b, min_periods=w_b).median().shift(1).to_numpy()
    with np.errstate(invalid="ignore"):
        hot = load >= r * np.maximum(med, floor)
    rows = []
    for p in range(n_part):
        padded = np.concatenate(([False], hot[:, p], [False])).astype(np.int8)
        change = np.diff(padded)
        starts = np.flatnonzero(change > 0)
        stops = np.flatnonzero(change < 0)
        for a, b in zip(starts, stops):
            if b - a < s:
                continue
            peak_idx = int(a + np.argmax(load[a:b, p]))
            peak = float(load[peak_idx, p])
            base = float(max(med[a, p], floor))
            lo_level = base + rise_lo * (peak - base)
            hi_level = base + rise_hi * (peak - base)
            window = load[a:peak_idx + 1, p]
            t_lo = int(a) + int(np.argmax(window >= lo_level))
            t_hi = int(a) + int(np.argmax(window >= hi_level))
            rows.append({
                "partition": p, "onset": int(a), "peak_step": peak_idx, "end": int(b) - 1,
                "duration": int(b - a), "peak": peak, "baseline": base, "rise_time": t_hi - t_lo,
            })
    events = pd.DataFrame(rows, columns=ROW_COLUMNS)
    events = events.sort_values(["onset", "partition"]).reset_index(drop=True)
    events.insert(0, "event_id", [f"{dataset}-{i}" for i in range(len(events))])
    events.insert(1, "dataset", dataset)
    events["split"] = np.where(
        events["onset"] < train_end, "train", np.where(events["onset"] < tune_end, "tune", "eval"))
    return events


def onset_cut(events):
    return float(np.median(events["rise_time"].to_numpy(dtype=np.float64)))


def assign_onset_class(events, cut):
    out = events.copy()
    out["onset_class"] = np.where(out["rise_time"] >= cut, "ramped", "abrupt")
    return out