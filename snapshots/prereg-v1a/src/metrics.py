import numpy as np

def calc_metrics(sim, slo_threshold):
    total_arrivals = np.sum(sim.hist_cpu_in)
    if total_arrivals == 0: 
        total_arrivals = 1.0
        
    excess_wait = np.maximum(0.0, sim.hist_wait - slo_threshold)
    damage = np.sum(excess_wait * sim.hist_cpu_in) / total_arrivals
    
    violating_steps = (sim.hist_wait > slo_threshold)
    violation_rate = np.sum(violating_steps * sim.hist_cpu_in) / total_arrivals
    
    peak_backlog = np.max(sim.hist_wait) * np.mean(sim.cpu_cap)
    
    mean_util = np.mean(sim.hist_cpu_in, axis=1)
    max_util = np.max(sim.hist_cpu_in, axis=1)
    imbalance = np.mean(max_util / np.maximum(mean_util, 1e-9))
    
    return {
        'damage': float(damage),
        'violation_rate': float(violation_rate),
        'peak_backlog': float(peak_backlog),
        'imbalance': float(imbalance),
        'moves': len(sim.hist_moves)
    }
