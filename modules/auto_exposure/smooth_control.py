"""Opt-in smooth AE controller (``auto_exposure.smooth_control: true``).

Continuous digital gain chosen so the *tone-mapped* mean luminance hits
``output_target``; the per-frame demand is filtered in the log2 domain with
a deadband, a first-order IIR (``time_constant_s``) and a slew limit
(``max_rate_stops_per_s``). Temporal constants are in seconds; the per-frame
coefficients are derived from ``frame_rate`` (Hz) so behaviour is
rate-independent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np


@dataclass(frozen=True)
class SmoothAEParams:
    output_target: float = 0.18
    time_constant_s: float = 0.35
    deadband_stops: float = 0.05
    max_rate_stops_per_s: float = 5.0
    frame_rate: float = 30.0
    gain_min: float = 0.03125
    gain_max: float = 16384.0
    meter_stride: int = 8
    highlight_clip: float = 1.0

    @classmethod
    def from_config(cls, parm_ae: Mapping[str, Any]) -> "SmoothAEParams":
        d = cls()
        out = cls(
            output_target=float(parm_ae.get("output_target", d.output_target)),
            time_constant_s=float(parm_ae.get("time_constant_s", d.time_constant_s)),
            deadband_stops=float(parm_ae.get("deadband_stops", d.deadband_stops)),
            max_rate_stops_per_s=float(parm_ae.get("max_rate_stops_per_s", d.max_rate_stops_per_s)),
            frame_rate=float(parm_ae.get("frame_rate", d.frame_rate)),
            gain_min=float(parm_ae.get("gain_min", d.gain_min)),
            gain_max=float(parm_ae.get("gain_max", d.gain_max)),
            meter_stride=int(parm_ae.get("meter_stride", d.meter_stride)),
            highlight_clip=float(parm_ae.get("highlight_clip", d.highlight_clip)),
        )
        if not (0 < out.output_target < 1 and out.time_constant_s >= 0 and out.deadband_stops >= 0
                and out.max_rate_stops_per_s > 0 and out.frame_rate > 0
                and 0 < out.gain_min < out.gain_max and out.meter_stride >= 1):
            raise ValueError(f"invalid smooth AE parameters: {out}")
        return out

    @property
    def alpha(self) -> float:
        """Per-frame IIR weight of the new demand."""
        if self.time_constant_s == 0:
            return 1.0
        return 1.0 - math.exp(-1.0 / (self.frame_rate * self.time_constant_s))

    @property
    def max_step_stops(self) -> float:
        return self.max_rate_stops_per_s / self.frame_rate


def scene_luminance(rgb_linear: np.ndarray, applied_gain: float, stride: int) -> np.ndarray:
    """Subsampled unity-gain scene-linear BT.709 luminance (nonnegative)."""
    sub = np.asarray(rgb_linear, dtype=np.float64)[::stride, ::stride]
    y = sub[..., 0] * 0.2126 + sub[..., 1] * 0.7152 + sub[..., 2] * 0.0722
    return np.maximum(y, 0.0) / float(applied_gain)


def solve_gain(
    y_unity: np.ndarray,
    tone_map: Callable[[np.ndarray], np.ndarray],
    p: SmoothAEParams,
    iterations: int = 18,
) -> float:
    """Gain g (log-bisection) with mean(min(TM(g*y), clip)) == output_target."""
    if y_unity.size == 0 or not np.any(y_unity > 0):
        return p.gain_max

    def out_mean(g: float) -> float:
        mapped = np.asarray(tone_map(y_unity * g), dtype=np.float64)
        return float(np.mean(np.clip(mapped, 0.0, p.highlight_clip)))

    lo, hi = math.log2(p.gain_min), math.log2(p.gain_max)
    if out_mean(2.0 ** hi) <= p.output_target:
        return p.gain_max
    if out_mean(2.0 ** lo) >= p.output_target:
        return p.gain_min
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if out_mean(2.0 ** mid) < p.output_target:
            lo = mid
        else:
            hi = mid
    return 2.0 ** (0.5 * (lo + hi))


def next_gain(previous: float | None, desired: float, p: SmoothAEParams) -> float:
    """Deadband + log-domain first-order IIR + slew limit."""
    desired = float(np.clip(desired, p.gain_min, p.gain_max))
    if previous is None or not math.isfinite(previous) or previous <= 0:
        return desired
    err = math.log2(desired) - math.log2(previous)
    if abs(err) <= p.deadband_stops:
        return float(previous)
    err -= math.copysign(p.deadband_stops, err)
    step = float(np.clip(p.alpha * err, -p.max_step_stops, p.max_step_stops))
    return float(np.clip(previous * 2.0 ** step, p.gain_min, p.gain_max))
