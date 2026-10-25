import numpy as np

class CrossLayerManager:
    def __init__(self, cfg, N, P):
        self.cfg = cfg
        self.N = N
        self.P = P
        self.enabled = cfg.get('layers', {}).get('enabled', False)
        self.h_net = cfg.get('h_net', 2.0)
        self.rho = cfg.get('rho', 0.25)
        self.phi = cfg.get('phi', 1.0)
        
    def get_cpu_demand(self, raw_load, dataset_type='D1'):
        if not self.enabled:
            return raw_load
        c0 = self.cfg.get('c0', 1.0)
        c1 = self.cfg.get('c1', 0.0)
        if dataset_type == 'D1':
            return raw_load * c0
        else:
            return raw_load * (c0 + c1 * 1.0) # Value size factored
            
    def get_link_capacity(self):
        return self.h_net
