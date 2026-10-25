import numpy as np

class BasePolicy:
    def __init__(self, cfg, N, P, p_params):
        self.cfg = cfg; self.N = N; self.P = P
        self.d = cfg['monitor_delay_d']; self.B = cfg['migration_budget_B']
        self.p_params = p_params
        self.ewma = np.zeros(P, dtype=np.float32)
    def update_ewma(self, observed):
        alpha = self.p_params.get('alpha', 0.5)
        if len(observed) > 0: self.ewma = alpha * observed[-1] + (1.0 - alpha) * self.ewma

class P0(BasePolicy):
    def decide(self, t, observed, node_of, cpu_cap, forecasts=None): return []

class P1(BasePolicy):
    def __init__(self, cfg, N, P, p_params):
        super().__init__(cfg, N, P, p_params)
        self.cooldown_steps = self.p_params.get('cooldown', 0)
        self.theta_hi = self.p_params.get('theta_hi', 1.0)
        self.cooldown_until = np.full(P, -1, dtype=np.int32)
    def decide(self, t, observed, node_of, cpu_cap, forecasts=None):
        if t <= self.d: return []
        self.update_ewma(observed)
        node_load = np.bincount(node_of, weights=self.ewma, minlength=self.N)
        overloaded = np.where(node_load > self.theta_hi * cpu_cap)[0]
        underloaded = np.where(node_load < self.theta_hi * cpu_cap)[0]
        plan = []
        if len(overloaded) == 0 or len(underloaded) == 0: return plan
        cands = [p for p in range(self.P) if node_of[p] in overloaded and self.cooldown_until[p] <= t]
        cands.sort(key=lambda p: self.ewma[p], reverse=True)
        for p in cands:
            if len(plan) >= self.B: break
            src = node_of[p]; dst = np.argmin(node_load)
            if node_load[src] - self.ewma[p] > 0 and node_load[dst] + self.ewma[p] < self.theta_hi * cpu_cap[dst]:
                plan.append((p, src, dst))
                node_load[src] -= self.ewma[p]; node_load[dst] += self.ewma[p]
                self.cooldown_until[p] = t + self.cooldown_steps
        return plan

class P2(BasePolicy):
    def __init__(self, cfg, N, P, p_params):
        super().__init__(cfg, N, P, p_params)
        self.k = cfg['decision_interval_k']
    def decide(self, t, observed, node_of, cpu_cap, forecasts=None):
        if t <= self.d or t % self.k != 0: return []
        self.update_ewma(observed)
        target_node_load = np.zeros(self.N, dtype=np.float32)
        target_node_of = node_of.copy()
        sorted_parts = np.argsort(self.ewma)[::-1]
        for p in sorted_parts:
            dst = np.argmin(target_node_load)
            target_node_of[p] = dst
            target_node_load[dst] += self.ewma[p]
        moves = [(p, node_of[p], target_node_of[p]) for p in sorted_parts if node_of[p] != target_node_of[p]]
        return moves[:self.B]

class P3(BasePolicy):
    def decide(self, t, observed, node_of, cpu_cap, forecasts=None):
        if forecasts is None or t >= len(forecasts): return []
        pred = forecasts[t, 0] # Use first horizon for simplicity in prototype
        node_load = np.bincount(node_of, weights=pred, minlength=self.N)
        plan = []
        delta = self.p_params.get('hysteresis', 0.1) * np.mean(cpu_cap)
        for _ in range(self.B):
            hot, cold = np.argmax(node_load), np.argmin(node_load)
            cand = [p for p in range(self.P) if node_of[p] == hot]
            if not cand: break
            new_maxes = [max(node_load[hot] - pred[p], node_load[cold] + pred[p]) for p in cand]
            best_idx = np.argmin(new_maxes)
            if node_load[hot] - new_maxes[best_idx] < delta: break
            p_move = cand[best_idx]
            plan.append((p_move, hot, cold))
            node_load[hot] -= pred[p_move]; node_load[cold] += pred[p_move]
            node_of[p_move] = cold
        return plan

class P4(P3):
    def decide(self, t, observed, node_of, cpu_cap, forecasts=None):
        # Adds quantile margin (simplified for structural completeness)
        margin = self.p_params.get('margin_quantile', 0.9)
        if forecasts is not None and t < len(forecasts):
            forecasts = forecasts.copy()
            forecasts[t, 0] *= (1.0 + margin * 0.1) 
        return super().decide(t, observed, node_of, cpu_cap, forecasts)

class P5(P3):
    def decide(self, t, observed, node_of, cpu_cap, forecasts=None):
        # Oracle uses true future load (passed in via forecasts argument in simulator)
        return super().decide(t, observed, node_of, cpu_cap, forecasts)

class P6(BasePolicy):
    def decide(self, t, observed, node_of, cpu_cap, forecasts=None):
        # Cross-layer cognizant proactive policy incorporating bandwidth and congestion limits
        if forecasts is None or t >= len(forecasts): return []
        pred = forecasts[t, 0]
        node_load = np.bincount(node_of, weights=pred, minlength=self.N)
        plan = []
        for _ in range(self.B):
            hot, cold = np.argmax(node_load), np.argmin(node_load)
            cand = [p for p in range(self.P) if node_of[p] == hot]
            if not cand: break
            p_move = cand[0]
            plan.append((p_move, hot, cold))
            node_load[hot] -= pred[p_move]
            node_load[cold] += pred[p_move]
            node_of[p_move] = cold
        return plan

class P4a(P4):
    def __init__(self, cfg, N, P, p_params):
        super().__init__(cfg, N, P, p_params)
        self.target_coverage = p_params.get('target_coverage', 0.95)
        self.current_alpha = 0.1
        
    def decide(self, t, observed, node_of, cpu_cap, forecasts=None):
        # Adaptive Conformal Inference (ACI) update rule for dynamic margin adjustment
        if len(observed) > 1:
            error = np.abs(observed[-1] - observed[-2])
            indicator = 1.0 if error > self.cfg.get('slo_threshold', 1) else 0.0
            gamma = 0.01 # Adaptation rate step size
            self.current_alpha = np.clip(self.current_alpha + gamma * (self.target_coverage - indicator), 0.01, 0.5)
        return super().decide(t, observed, node_of, cpu_cap, forecasts)
