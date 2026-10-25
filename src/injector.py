"""D3 injector: semi-synthetic flash-crowd spikes planted on a baseline load matrix."""
import numpy as np
import pandas as pd

from src.seeds import derive_seed


def inject_spikes(load, *, dataset, master_seed, n_events, amp_min, amp_max, rise_min, rise_max,
                  dur_min, dur_max, hot_min, hot_max, w_b, floor, start, label):
    """Plant n_events spikes after `start`, spaced so baseline windows never overlap a spike.

    A spike adds amp x baseline x shape(k) to each hit partition, shape = linear ramp over
    `rise` steps then plateau for `dur` steps. Baseline = median of the W_b steps before onset.
    Key concentration = number of hit partitions (fewer = more concentrated).
    """
    rng = np.random.default_rng(derive_seed(master_seed, dataset, label))
    n_steps = load.shape[0]
    out = load.copy()
    gap = w_b + dur_max + rise_max
    first = max(start, w_b)
    last = n_steps - dur_max
    width = (last - first) // n_events
    if width < gap:
        raise ValueError(f"{n_events} events do not fit: slot width {width} < required gap {gap}")
    rows = []
    for i in range(n_events):
        onset = first + i * width + int(rng.integers(0, width - gap + 1))
        med = np.median(out[onset - w_b:onset], axis=0)
        eligible = np.flatnonzero((med >= floor) & (med > 0))
        if eligible.size == 0:
            raise ValueError(f"no eligible partition for spike {i}")
        amp = float(rng.uniform(amp_min, amp_max))
        rise = int(rng.integers(rise_min, rise_max + 1))
        dur = int(rng.integers(dur_min, dur_max + 1))
        hot = int(rng.integers(hot_min, hot_max + 1))
        targets = np.sort(rng.choice(eligible, size=min(hot, eligible.size), replace=False))
        shape = np.minimum(1, (np.arange(dur) + 1) / rise)
        for p in targets:
            out[onset:onset + dur, p] += amp * med[p] * shape
            rows.append({
                "event_id": f"{dataset}-{label}-{i}", "dataset": dataset, "partition": int(p),
                "onset": onset, "rise_steps": rise, "duration": dur, "amplitude": amp,
                "baseline": float(med[p]), "n_hot": int(targets.size), "split": "eval",
            })
    return out, pd.DataFrame(rows)