"""Scene-adaptive global white point (``tone_mapping.adaptive_white``).

One global statistic per frame (a high percentile of the max RGB channel of the
scene-linear image) sets the desired white W >= 1 in normalized units; W is
low-pass filtered in the log domain (time constant in seconds) and applied to the
NEXT frame as the curve's white: out = f(g*n) / f(g*W). It is the same global
monotone curve for every pixel of a frame.

Keys: percentile (default 99.7), max_white (default 8), time_constant_s (default 1),
frame_rate (default 10), meter_stride (default 8).
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


def _params(cfg: Mapping[str, Any]) -> tuple[float, float, float, float, int]:
    pct = float(cfg.get("percentile", 99.7))
    wmax = float(cfg.get("max_white", 8.0))
    tau = float(cfg.get("time_constant_s", 1.0))
    fps = float(cfg.get("frame_rate", 10.0))
    stride = int(cfg.get("meter_stride", 8))
    if not (0 < pct <= 100 and wmax >= 1 and tau >= 0 and fps > 0 and stride >= 1):
        raise ValueError("tone_mapping.adaptive_white: need 0<percentile<=100, max_white>=1, "
                         "time_constant_s>=0, frame_rate>0, meter_stride>=1")
    return pct, wmax, tau, fps, stride


def desired_white(rgb_linear: np.ndarray, cfg: Mapping[str, Any]) -> float:
    pct, wmax, _, _, stride = _params(cfg)
    sub = np.asarray(rgb_linear, dtype=np.float64)[::stride, ::stride]
    peak = np.max(sub, axis=-1)
    if peak.size == 0:
        return 1.0
    return float(np.clip(np.percentile(peak, pct), 1.0, wmax))


def next_white(previous: float | None, desired: float, cfg: Mapping[str, Any]) -> float:
    _, wmax, tau, fps, _ = _params(cfg)
    desired = float(np.clip(desired, 1.0, wmax))
    if previous is None or not math.isfinite(previous) or previous < 1.0:
        return desired
    alpha = 1.0 if tau == 0 else 1.0 - math.exp(-1.0 / (fps * tau))
    return float(previous * math.exp(alpha * (math.log(desired) - math.log(previous))))
