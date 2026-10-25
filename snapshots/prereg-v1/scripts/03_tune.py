import argparse
import yaml
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import ParameterSampler
import sys
import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.simulator import Simulator
from src.metrics import calc_metrics
from src.policies import P1, P2

def run_replicates(cfg, policy_class, p_params, L_tune, demand, num_replicates=10):
    rng = np.random.default_rng(cfg['master_seed'])
    T_total, P = L_tune.shape
    N = cfg['n_nodes']
    W = cfg['window_steps']
    
    total_damage = 0.0
    for _ in range(num_replicates):
        start_t = rng.integers(0, max(1, T_total - W))
        L_window = L_tune[start_t:start_t+W, :]
        
        # Calculate capacity from training prefix (simulated here as mean * headroom)
        mu = cfg['headroom'] * np.mean(np.sum(L_tune * demand, axis=1)) / N
        
        sim = Simulator(cfg, P, N, L_window, demand, cpu_cap=mu)
        policy = policy_class(cfg, N, P, p_params)
        
        for t in range(W):
            d = cfg['monitor_delay_d']
            obs = L_window[:max(0, t - d + 1)]
            plan = policy.decide(t, obs, sim.node_of.copy(), sim.cpu_cap)
            sim.step(t, plan)
            
        metrics = calc_metrics(sim, slo_threshold=cfg['slo_threshold'])
        total_damage += metrics['damage']
        
    return total_damage / num_replicates

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f: cfg = yaml.safe_load(f)
    
    # Grid definition
    grid = {}
    if args.policy == 'P1':
        grid = {
            'alpha': cfg['ewma_alpha_grid'],
            'theta_hi': cfg['theta_hi_grid'],
            'cooldown': cfg['cooldown_grid']
        }
    elif args.policy == 'P2':
        grid = {
            'alpha': cfg['ewma_alpha_grid']
        }
    else:
        sys.exit(0)
        
    sampler = ParameterSampler(grid, n_iter=cfg['tuning_budget'], random_state=cfg['master_seed'])
    configs_to_try = list(sampler)
    
    # Load D1 Day 3 (Tuning Split)
    mins = cfg['azure_mins_per_day']
    data_root = Path(cfg.get('data_root', 'data'))
    L = np.load(data_root / "processed" / "D1_L.npy")
    demand = np.load(data_root / "processed" / "D1_demand.npy")
    
    start_idx, end_idx = 2 * mins, 3 * mins
    L_tune = L[start_idx:end_idx, :]
    
    best_damage = float('inf')
    best_config = None
    results = []
    
    Path(args.out).parent.mkdir(exist_ok=True, parents=True)
    policy_class = P1 if args.policy == 'P1' else P2
    
    for i, p_params in enumerate(configs_to_try):
        avg_dmg = run_replicates(cfg, policy_class, p_params, L_tune, demand)
        results.append({'iter': i, 'params': p_params, 'damage': avg_dmg})
        
        if avg_dmg < best_damage:
            best_damage = avg_dmg
            best_config = p_params
            
    # Plateau Check Logging
    if len(results) >= 5:
        recent_best = min(r['damage'] for r in results[-5:])
        if best_damage > 0 and (recent_best - best_damage) / best_damage > cfg['tune_plateau_tol']:
            print(f"[{args.policy}] Plateau tolerance triggered. Amendment needed.")
            
    with open(args.out, 'w') as f:
        for r in results: f.write(json.dumps(r) + '\n')
        
    print(f"[{args.policy}] Tuning complete. Best damage: {best_damage:.4f} with {best_config}")

if __name__ == '__main__':
    main()
