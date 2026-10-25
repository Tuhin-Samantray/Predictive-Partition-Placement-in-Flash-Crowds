"""RV4 logging (A43): trigger counts and forecast-quality fields for P4 and P4a rows.

Off by default. The trigger counts come from a pass-through wrapper around policy.decide that
returns the plan unchanged. The interval fields are replayed after the run from the replicate truth
and forecast window, origin by origin, with the policy classes themselves, so they do not depend on
the decision interval k and cannot change a behavioural metric. The interval is one-sided (an upper
bound); its width is the margin above the clipped point forecast. Every parameter is an argument.
"""
import numpy as np

from src.policies import P4Hedged, P4aAdaptive, blend, horizon_weights


class LoggedPolicy:
    """Counts decision steps and proposals of the wrapped policy; the plan is returned unchanged."""

    def __init__(self, policy):
        self.policy = policy
        self.decisions = 0
        self.proposed = 0
        self.proposed_steps = 0
        self.no_slot_steps = 0

    def decide(self, t, observed, state):
        plan = self.policy.decide(t, observed, state)
        self.decisions += 1
        self.proposed += len(plan)
        self.proposed_steps += int(len(plan) > 0)
        self.no_slot_steps += int(state["free_slots"] == 0)
        return plan

    def fields(self):
        return {
            "trig_decisions": self.decisions,
            "trig_proposed": self.proposed,
            "trig_proposed_steps": self.proposed_steps,
            "trig_no_slot_steps": self.no_slot_steps,
        }


def rv4_fields(load, fc, ctx, margin_quantile):
    """Point error and fixed-quantile and adaptive upper-bound coverage, width and alpha, over the
    origins o whose target step o + m + d lies inside the window. Coverage is hits over trials."""
    load = np.asarray(load, dtype=np.float64)
    fc = np.asarray(fc)
    n_steps, n_part = load.shape
    if fc.ndim != 3 or fc.shape[0] != n_steps or fc.shape[2] != n_part:
        raise ValueError("forecast must have shape (T, horizons, P) matching the load window")
    target = ctx["m"] + ctx["d"]
    n_orig = n_steps - target
    if n_orig < 1:
        raise ValueError("window too short for the completion horizon")
    horizons = ctx["horizons"]
    window = ctx["residual_window"]
    i0, i1, w = horizon_weights(horizons, target)
    base = np.maximum(blend(fc[:n_orig], i0, i1, w), 0)
    truth = load[target:]
    err = truth - base
    fixed = P4Hedged(0, target, 0, horizons, margin_quantile, window)
    adaptive = P4aAdaptive(0, target, 0, horizons, margin_quantile, window, ctx["gamma"], ctx["a_min"],
                           ctx["a_max"])
    u_fix = np.zeros((n_orig, n_part))
    u_ad = np.zeros((n_orig, n_part))
    for o in range(n_orig):
        u_fix[o] = base[o] + fixed.margin(load[:o + 1], fc[:o + 1])
        u_ad[o] = adaptive.advance(load[:o + 1], fc[:o + 1])
    cells = n_orig * n_part
    hits_fix = int((truth <= u_fix).sum())
    hits_ad = int((truth <= u_ad).sum())
    return {
        "rv4_origins": n_orig,
        "rv4_trials": cells,
        "rv4_point_mae": float(np.abs(err).mean()),
        "rv4_point_bias": float(err.mean()),
        "rv4_fix_hits": hits_fix,
        "rv4_fix_cov": hits_fix / cells,
        "rv4_fix_width": float((u_fix - base).mean()),
        "rv4_ad_hits": hits_ad,
        "rv4_ad_cov": hits_ad / cells,
        "rv4_ad_width": float((u_ad - base).mean()),
        "rv4_ad_alpha_mean": float(np.mean(adaptive.alpha_trace)),
        "rv4_ad_alpha_last": float(adaptive.alpha_trace[-1]),
    }
