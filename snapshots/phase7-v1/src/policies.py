"""Policies P0 to P5 (Section 4.2) and the shared greedy repack.

A policy sees only `observed` (load up to t - d, enforced by the simulator), the placement
state and, for P3 and P4, `state["forecast"]`: the forecast rows for origins up to t - d (A8,
enforced by the simulator). P5 is the oracle and is the only policy handed true future load.
Every parameter is an argument; none is defined in this module.
Use one policy instance per replicate: the EWMA state is not reset between runs.
"""
import numpy as np

# tunable parameter -> config key holding its grid
TUNABLE = {
    "P1": {"ewma_alpha": "ewma_alpha_grid", "theta_hi": "theta_hi_grid", "cooldown": "cooldown_grid"},
    "P2": {"ewma_alpha": "ewma_alpha_grid", "hysteresis": "hysteresis_grid", "cooldown": "cooldown_grid"},
}
# forecast-driven policies: grids smaller than tuning_budget, searched exhaustively (A12)
FORECAST_TUNABLE = {
    "P3": {"hysteresis": "hysteresis_grid", "cooldown": "cooldown_grid"},
    "P4": {"hysteresis": "hysteresis_grid", "cooldown": "cooldown_grid"},
    "P5": {"hysteresis": "hysteresis_grid", "cooldown": "cooldown_grid", "oracle_window": "oracle_window_grid"},
}


class Ewma:
    """Per-partition exponentially weighted moving average, updated incrementally."""

    def __init__(self, alpha):
        self.alpha = alpha
        self.value = None
        self.n_seen = 0

    def update(self, observed):
        if observed.shape[0] < self.n_seen:
            raise ValueError("policy instance reused across runs")
        for row in observed[self.n_seen:]:
            if self.value is None:
                self.value = np.array(row, dtype=np.float64)
            else:
                self.value = self.alpha * row + (1 - self.alpha) * self.value
        self.n_seen = observed.shape[0]
        return self.value


def greedy_plan(pred, node_load, node_of, movable, free, delta):
    """Greedy hot-to-cold repack (Section 4.5 sketch). `pred` is the per-partition load the
    policy believes in (EWMA for P2, forecast for P3, truth for P5)."""
    nl = np.array(node_load, dtype=np.float64)
    movable = np.array(movable, dtype=bool)
    plan = []
    for _ in range(free):
        hot, cold = int(nl.argmax()), int(nl.argmin())
        if hot == cold:
            break
        cand = np.flatnonzero((node_of == hot) & movable)
        if cand.size == 0:
            break
        others = np.delete(nl, [hot, cold])
        rest = others.max() if others.size else 0
        new_max = np.maximum.reduce(
            [nl[hot] - pred[cand], nl[cold] + pred[cand], np.full(cand.size, rest)]
        )
        j = int(new_max.argmin())
        if nl.max() - new_max[j] <= delta:
            break
        p = int(cand[j])
        plan.append((p, hot, cold))
        nl[hot] -= pred[p]
        nl[cold] += pred[p]
        movable[p] = False
    return plan


def horizon_weights(horizons, target):
    """Linear interpolation along the stored horizons to the completion horizon `target`
    (A11). Returns (i0, i1, w): value = (1 - w) x F[i0] + w x F[i1]. Clamped at both ends."""
    hs = np.asarray(horizons, dtype=np.float64)
    if target <= hs[0]:
        return 0, 0, 0
    if target >= hs[-1]:
        return hs.size - 1, hs.size - 1, 0
    i1 = int(np.searchsorted(hs, target, side="left"))
    i0 = i1 - 1
    return i0, i1, float((target - hs[i0]) / (hs[i1] - hs[i0]))


def blend(f, i0, i1, w):
    """Interpolate over the horizon axis (second to last) of a (..., H, P) forecast array."""
    f = np.asarray(f, dtype=np.float64)
    return (1 - w) * np.take(f, i0, axis=-2) + w * np.take(f, i1, axis=-2)


def positive_quantile(resid, q):
    """Per-column linear-interpolation quantile q of the strictly positive entries of
    resid (W, P); columns without positive entries give 0. Vectorized equivalent of
    nanquantile over np.where(resid > 0, resid, nan), using numpy's lerp."""
    pos = resid > 0
    count = pos.sum(axis=0)
    srt = np.sort(np.where(pos, resid, np.inf), axis=0)
    idx = q * (count - 1)
    lo = np.clip(np.floor(idx).astype(np.int64), 0, None)
    hi = np.minimum(lo + 1, np.maximum(count - 1, 0))
    t = np.where(count > 0, idx - lo, 0)
    a = np.where(count > 0, np.take_along_axis(srt, lo[None, :], axis=0)[0], 0)
    b = np.where(count > 0, np.take_along_axis(srt, hi[None, :], axis=0)[0], 0)
    diff = b - a
    return np.where(t >= 1 - t, b - diff * (1 - t), a + diff * t)


class P0Static:
    """Static hash placement: never moves."""

    def decide(self, t, observed, state):
        return []


class P1Threshold:
    """E-Store style: a node whose EWMA load exceeds theta_hi x capacity sheds its hottest
    movable partition that fits under the limit of the least-utilized node."""

    def __init__(self, mu, ewma_alpha, theta_hi):
        self.mu = np.asarray(mu, dtype=np.float64)
        self.limit = theta_hi * self.mu
        self.ewma = Ewma(ewma_alpha)

    def decide(self, t, observed, state):
        e = self.ewma.update(observed)
        if e is None:
            return []
        node_of, n_nodes = state["node_of"], state["n_nodes"]
        nl = np.bincount(node_of, weights=e, minlength=n_nodes)
        movable = np.array(state["movable"], dtype=bool)
        free = state["free_slots"]
        plan = []
        for hot in np.argsort(-(nl - self.limit), kind="stable"):
            hot = int(hot)
            while free > 0 and nl[hot] > self.limit[hot]:
                cold = int(np.argmin(nl / self.mu))
                if cold == hot:
                    break
                fits = (node_of == hot) & movable & (e + nl[cold] <= self.limit[cold])
                cand = np.flatnonzero(fits)
                if cand.size == 0:
                    break
                p = int(cand[np.argmax(e[cand])])
                plan.append((p, hot, cold))
                nl[hot] -= e[p]
                nl[cold] += e[p]
                movable[p] = False
                free -= 1
        return plan


class P2Greedy:
    """Reactive greedy repack on EWMA load; hysteresis is a fraction of mean node load;
    the number of moves is capped by the free migration budget."""

    def __init__(self, ewma_alpha, hysteresis):
        self.ewma = Ewma(ewma_alpha)
        self.hysteresis = hysteresis

    def decide(self, t, observed, state):
        e = self.ewma.update(observed)
        if e is None:
            return []
        node_of = state["node_of"]
        nl = np.bincount(node_of, weights=e, minlength=state["n_nodes"])
        return greedy_plan(e, nl, node_of, state["movable"], state["free_slots"],
                           self.hysteresis * nl.mean())


class P3Forecast:
    """Forecast-driven greedy repack. At step t it reads F[t - d], the newest forecast it may
    see, takes the load forecast for the completion step t + m, which is horizon m + d from
    that origin (interpolated between stored horizons, A11), and repacks as P2 does."""

    def __init__(self, hysteresis, m, d, horizons):
        self.hysteresis = hysteresis
        self.target = m + d
        self.i0, self.i1, self.w = horizon_weights(horizons, self.target)

    def pred_load(self, observed, fc):
        return np.maximum(blend(fc[-1], self.i0, self.i1, self.w), 0)

    def decide(self, t, observed, state):
        fc = state.get("forecast")
        if fc is None:
            raise ValueError("forecast policies need a forecast table")
        if fc.shape[0] == 0:
            return []
        pred = self.pred_load(observed, fc)
        node_of = state["node_of"]
        nl = np.bincount(node_of, weights=pred, minlength=state["n_nodes"])
        return greedy_plan(pred, nl, node_of, state["movable"], state["free_slots"],
                           self.hysteresis * nl.mean())


class P4Hedged(P3Forecast):
    """P3 with an upper-quantile margin (A12). Per partition, the margin is the margin_quantile
    quantile of the positive residuals (truth minus the same interpolated forecast) over the
    last residual_window origins whose target step has already been observed; zero when a
    partition has none. Residuals come from `observed` only."""

    def __init__(self, hysteresis, m, d, horizons, margin_quantile, residual_window):
        super().__init__(hysteresis, m, d, horizons)
        self.q = margin_quantile
        self.window = residual_window

    def margin(self, observed, fc):
        if observed.shape[0] != fc.shape[0]:
            raise ValueError("observed and forecast rows must cover the same origins")
        hi = fc.shape[0] - 1 - self.target
        if hi < 0:
            return 0
        lo = max(0, hi - self.window + 1)
        pred = blend(fc[lo:hi + 1], self.i0, self.i1, self.w)
        resid = observed[lo + self.target: hi + self.target + 1] - pred
        return positive_quantile(resid, self.q)

    def pred_load(self, observed, fc):
        return super().pred_load(observed, fc) + self.margin(observed, fc)


class P5Oracle:
    """Same mechanism as P3 with the true load, averaged over `window` consecutive steps starting
    at the completion step t + m (clamped and truncated at the end of the replicate window, A18).
    Window 1 is the single completion step as first specified. The only policy that reads the
    future, by design."""

    def __init__(self, hysteresis, m, load, window):
        self.hysteresis = hysteresis
        self.m = m
        self.window = window
        self.load = np.asarray(load, dtype=np.float64)

    def target(self, t):
        start = min(t + self.m, self.load.shape[0] - 1)
        return self.load[start:start + self.window].mean(axis=0)

    def decide(self, t, observed, state):
        pred = self.target(t)
        node_of = state["node_of"]
        nl = np.bincount(node_of, weights=pred, minlength=state["n_nodes"])
        return greedy_plan(pred, nl, node_of, state["movable"], state["free_slots"],
                           self.hysteresis * nl.mean())


def build_policy(name, params, mu, ctx=None):
    if name == "P0":
        return P0Static()
    if name == "P1":
        return P1Threshold(mu, params["ewma_alpha"], params["theta_hi"])
    if name == "P2":
        return P2Greedy(params["ewma_alpha"], params["hysteresis"])
    if name in ("P3", "P4", "P5"):
        if ctx is None:
            raise ValueError(name + " needs ctx (m, d, horizons, load, margin_quantile, residual_window)")
        if name == "P3":
            return P3Forecast(params["hysteresis"], ctx["m"], ctx["d"], ctx["horizons"])
        if name == "P4":
            return P4Hedged(params["hysteresis"], ctx["m"], ctx["d"], ctx["horizons"],
                            ctx["margin_quantile"], ctx["residual_window"])
        return P5Oracle(params["hysteresis"], ctx["m"], ctx["load"], params["oracle_window"])
    raise ValueError("unknown policy: " + str(name))
