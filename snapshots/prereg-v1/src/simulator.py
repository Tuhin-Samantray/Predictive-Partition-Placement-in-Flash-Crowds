import numpy as np

class Simulator:
    def __init__(self, cfg, P, N, L_matrix, demand, cpu_cap, initial_placement=None):
        self.cfg = cfg
        self.P = P
        self.N = N
        self.L = L_matrix
        self.demand = demand
        self.cpu_cap = np.full(N, cpu_cap, dtype=np.float32)
        
        self.T = L_matrix.shape[0]
        self.kappa = cfg['migration_overhead_kappa']
        self.m0 = cfg.get('migration_steps_m0', 0)
        
        if initial_placement is not None:
            self.node_of = np.copy(initial_placement)
        else:
            self.node_of = np.arange(P) % N
            
        self.q_cpu = np.zeros(N, dtype=np.float32)
        self.moving_until = np.full(P, -1, dtype=np.int32)
        self.moving_from = np.full(P, -1, dtype=np.int32)
        self.moving_to = np.full(P, -1, dtype=np.int32)
        
        self.hist_wait = np.zeros((self.T, N), dtype=np.float32)
        self.hist_cpu_in = np.zeros((self.T, N), dtype=np.float32)
        self.hist_q_cpu = np.zeros((self.T, N), dtype=np.float32)
        self.hist_moves = []
        
    def step(self, t, policy_plan):
        # 1. Finish migrations scheduled for time t
        finished = (self.moving_until == t)
        if np.any(finished):
            self.node_of[finished] = self.moving_to[finished]
            self.moving_until[finished] = -1
            
        # 2. Start new migrations from the policy plan
        for (p, src, dst) in policy_plan:
            if self.moving_until[p] == -1:
                if self.m0 == 0:
                    self.node_of[p] = dst
                    self.hist_moves.append((t, p, src, dst))
                else:
                    self.moving_until[p] = t + self.m0
                    self.moving_from[p] = src
                    self.moving_to[p] = dst
                    self.hist_moves.append((t, p, src, dst))
                    
        # 3. Calculate loads (source pays overhead kappa for in-flight partitions)
        is_moving = (self.moving_until > t)
        current_load = self.L[t] * (1.0 + self.kappa * is_moving)
        weighted_load = current_load * self.demand
        
        cpu_in = np.bincount(self.node_of, weights=weighted_load, minlength=self.N).astype(np.float32)
        
        # 4. Fluid queue dynamics and wait proxy
        wait_cpu = (self.q_cpu + cpu_in / 2.0) / self.cpu_cap
        self.hist_wait[t] = wait_cpu
        self.hist_cpu_in[t] = cpu_in
        self.hist_q_cpu[t] = self.q_cpu
        
        self.q_cpu = np.maximum(0.0, self.q_cpu + cpu_in - self.cpu_cap)
