import numpy as np

class BaseForecaster:
    def __init__(self, cfg, P, horizons):
        self.cfg = cfg
        self.P = P
        self.horizons = horizons

class NaiveForecaster(BaseForecaster):
    def fit(self, L_train): pass
    def predict(self, current_t, L_obs):
        last_val = L_obs[-1] if len(L_obs) > 0 else np.zeros(self.P)
        return np.tile(last_val, (len(self.horizons), 1))

class SeasonalForecaster(BaseForecaster):
    def __init__(self, cfg, P, horizons):
        super().__init__(cfg, P, horizons)
        self.season = cfg['seasonal_period']
    def fit(self, L_train): pass
    def predict(self, current_t, L_obs):
        preds = np.zeros((len(self.horizons), self.P))
        for i, h in enumerate(self.horizons):
            idx = current_t + h - self.season
            if idx >= 0 and idx < len(L_obs):
                preds[i] = L_obs[idx]
            else:
                preds[i] = L_obs[-1] if len(L_obs) > 0 else 0
        return preds
