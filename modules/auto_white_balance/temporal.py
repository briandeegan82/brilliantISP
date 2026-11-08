"""Opt-in AWB plausibility clamp and temporal filter.

Keys (``auto_white_balance``), all optional; absent keys keep legacy behaviour:
  r_gain_range / b_gain_range: [lo, hi] plausibility box for estimated gains
  time_constant_s: time constant (s) of a first-order IIR on log gains
  max_rate_per_s: max |d ln gain / dt| (1/s) after smoothing
  frame_rate: frame rate (Hz) used to convert the above to per-frame values
  low_light_prior: [r, b] gains the estimate is blended toward in low light
  low_light_gain_range: [g0, g1] applied digital gain where the blend weight
    ramps (log-linearly) from 0 to 1
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


def _range(parm: Mapping[str, Any], key: str) -> tuple[float, float] | None:
    value = parm.get(key)
    if value is None:
        return None
    lo, hi = (float(v) for v in value)
    if not (0 < lo <= hi and math.isfinite(hi)):
        raise ValueError(f"auto_white_balance.{key} must be 0 < lo <= hi")
    return lo, hi


def is_active(parm: Mapping[str, Any]) -> bool:
    return any(k in parm for k in ("r_gain_range", "b_gain_range", "time_constant_s", "max_rate_per_s",
                                   "low_light_prior"))


def low_light_weight(input_gain: float, parm: Mapping[str, Any]) -> float:
    if parm.get("low_light_prior") is None:
        return 0.0
    g0, g1 = _range(parm, "low_light_gain_range") or (math.inf, math.inf)
    if not math.isfinite(g0) or g1 <= g0:
        raise ValueError("auto_white_balance.low_light_gain_range must be finite with lo < hi")
    w = (math.log2(max(input_gain, 1e-12)) - math.log2(g0)) / (math.log2(g1) - math.log2(g0))
    return float(np.clip(w, 0.0, 1.0))


def filter_gains(
    estimate: tuple[float, float],
    previous: tuple[float, float] | None,
    parm: Mapping[str, Any],
    input_gain: float = 1.0,
) -> tuple[float, float]:
    r, b = float(estimate[0]), float(estimate[1])
    w = low_light_weight(input_gain, parm)
    if w > 0:
        pr, pb = (float(v) for v in parm["low_light_prior"])
        r = math.exp((1 - w) * math.log(r) + w * math.log(pr))
        b = math.exp((1 - w) * math.log(b) + w * math.log(pb))
    rr, br = _range(parm, "r_gain_range"), _range(parm, "b_gain_range")
    if rr:
        r = float(np.clip(r, *rr))
    if br:
        b = float(np.clip(b, *br))
    if previous is None or not all(math.isfinite(v) and v > 0 for v in previous):
        return r, b
    fps = float(parm.get("frame_rate", 30.0))
    tau = float(parm.get("time_constant_s", 0.0))
    if not (fps > 0 and tau >= 0):
        raise ValueError("auto_white_balance.frame_rate must be > 0 and time_constant_s >= 0")
    alpha = 1.0 if tau == 0 else 1.0 - math.exp(-1.0 / (fps * tau))
    rate = parm.get("max_rate_per_s")
    step_cap = None if rate is None else float(rate) / fps
    out = []
    for new, old in ((r, previous[0]), (b, previous[1])):
        d = alpha * (math.log(new) - math.log(old))
        if step_cap is not None:
            d = float(np.clip(d, -float(step_cap), float(step_cap)))
        out.append(old * math.exp(d))
    return out[0], out[1]
