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
    low_light_target_stops: float = 0.0
    low_light_gain_range: tuple[float, float] = (1e4, 1e5)
    clip_level: float = 0.98
    max_clip_fraction: float = 1.0
    clip_max_backoff_stops: float = 1.0
    low_light_deadband_stops: float | None = None
    target_schedule: tuple[tuple[float, float], ...] | None = None

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
            low_light_target_stops=float(parm_ae.get("low_light_target_stops", d.low_light_target_stops)),
            low_light_gain_range=tuple(float(v) for v in parm_ae.get("low_light_gain_range", d.low_light_gain_range)),
            clip_level=float(parm_ae.get("clip_level", d.clip_level)),
            max_clip_fraction=float(parm_ae.get("max_clip_fraction", d.max_clip_fraction)),
            clip_max_backoff_stops=float(parm_ae.get("clip_max_backoff_stops", d.clip_max_backoff_stops)),
            low_light_deadband_stops=(None if parm_ae.get("low_light_deadband_stops") is None
                                      else float(parm_ae["low_light_deadband_stops"])),
            target_schedule=(None if parm_ae.get("target_schedule") is None
                             else tuple((float(g), float(st)) for g, st in parm_ae["target_schedule"])),
        )
        ts = out.target_schedule
        if ts is not None and (len(ts) < 2 or any(g <= 0 or not math.isfinite(st) for g, st in ts)
                               or any(b[0] <= a[0] for a, b in zip(ts, ts[1:]))):
            raise ValueError("auto_exposure.target_schedule must be >= 2 [gain, stops] points, gains increasing")
        g0, g1 = out.low_light_gain_range
        if not (0 < out.output_target < 1 and out.time_constant_s >= 0 and out.deadband_stops >= 0
                and out.max_rate_stops_per_s > 0 and out.frame_rate > 0
                and 0 < out.gain_min < out.gain_max and out.meter_stride >= 1
                and out.low_light_target_stops >= 0 and 0 < g0 < g1
                and 0 < out.clip_level <= 1 and 0 < out.max_clip_fraction <= 1
                and out.clip_max_backoff_stops >= 0
                and (out.low_light_deadband_stops is None or out.low_light_deadband_stops >= 0)):
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

    def target_at(self, gain: float) -> float:
        """Output target. With ``target_schedule`` ([gain, stops] points, log-gain linear
        interpolation, flat beyond the ends) it is output_target * 2**stops(gain); otherwise it is
        lowered by up to ``low_light_target_stops`` as total gain rises
        through ``low_light_gain_range`` (log-linear), so noise-limited scenes are not
        forced to full mid-grey."""
        if self.target_schedule is not None:
            lg = [math.log2(g) for g, _ in self.target_schedule]
            st = [v for _, v in self.target_schedule]
            return self.output_target * 2.0 ** float(np.interp(math.log2(gain), lg, st))
        if self.low_light_target_stops == 0:
            return self.output_target
        return self.output_target * 2.0 ** (-self.low_light_target_stops * self.low_light_weight(gain))

    def low_light_weight(self, gain: float) -> float:
        """0 below ``low_light_gain_range``, 1 above it, log-linear in between."""
        g0, g1 = (math.log2(v) for v in self.low_light_gain_range)
        return min(max((math.log2(gain) - g0) / (g1 - g0), 0.0), 1.0)

    def deadband_at(self, gain: float) -> float:
        """Deadband, widened toward ``low_light_deadband_stops`` across the low-light gain range
        (noise-limited metering at high gain must not drive gain hunting)."""
        if self.low_light_deadband_stops is None:
            return self.deadband_stops
        w = self.low_light_weight(gain)
        return self.deadband_stops + (self.low_light_deadband_stops - self.deadband_stops) * w


def scene_luminance(rgb_linear: np.ndarray, applied_gain: float, stride: int) -> np.ndarray:
    """Subsampled unity-gain scene-linear BT.709 luminance (nonnegative)."""
    sub = np.asarray(rgb_linear, dtype=np.float64)[::stride, ::stride]
    y = sub[..., 0] * 0.2126 + sub[..., 1] * 0.7152 + sub[..., 2] * 0.0722
    return np.maximum(y, 0.0) / float(applied_gain)


def scene_peak(rgb_linear: np.ndarray, applied_gain: float, stride: int) -> np.ndarray:
    """Subsampled unity-gain per-pixel max channel (nonnegative)."""
    sub = np.asarray(rgb_linear, dtype=np.float64)[::stride, ::stride]
    return np.maximum(sub.max(axis=-1), 0.0) / float(applied_gain)


def solve_gain(
    y_unity: np.ndarray,
    tone_map: Callable[[np.ndarray], np.ndarray],
    p: SmoothAEParams,
    iterations: int = 18,
    peak_unity: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> float:
    """Gain g (log-bisection) with mean(min(TM(g*y), clip)) == target_at(g), capped so the
    fraction of tone-mapped pixels at or above ``clip_level`` stays within ``max_clip_fraction``
    (highlight protection backs off at most ``clip_max_backoff_stops``)."""
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != y_unity.shape or not np.any(weights > 0):
            raise ValueError("metering weights must match the meter shape and have positive mass")
        y_unity = np.where(weights > 0, y_unity, 0.0)
    if y_unity.size == 0 or not np.any(y_unity > 0):
        return p.gain_max

    def wmean(v: np.ndarray) -> float:
        return float(np.average(v, weights=weights)) if weights is not None else float(np.mean(v))

    def excess(g: float) -> float:
        mapped = np.asarray(tone_map(y_unity * g), dtype=np.float64)
        return wmean(np.clip(mapped, 0.0, p.highlight_clip)) - p.target_at(g)

    def clipped(g: float) -> float:
        mapped = np.asarray(tone_map(y_unity * g), dtype=np.float64)
        if peak_unity is not None:
            ratio = np.divide(mapped, y_unity * g, out=np.zeros_like(mapped), where=y_unity > 0)
            mapped = np.maximum(mapped, peak_unity * g * ratio)
        return wmean((mapped >= p.clip_level).astype(np.float64)) - p.max_clip_fraction

    def bisect(f: Callable[[float], float]) -> float:
        lo, hi = math.log2(p.gain_min), math.log2(p.gain_max)
        if f(2.0 ** hi) <= 0:
            return p.gain_max
        if f(2.0 ** lo) >= 0:
            return p.gain_min
        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            if f(2.0 ** mid) < 0:
                lo = mid
            else:
                hi = mid
        return 2.0 ** (0.5 * (lo + hi))

    gain = bisect(excess)
    if p.max_clip_fraction < 1.0 and clipped(gain) > 0:
        gain = max(bisect(clipped), gain * 2.0 ** -p.clip_max_backoff_stops)
    return gain


def next_gain(previous: float | None, desired: float, p: SmoothAEParams) -> float:
    """Deadband + log-domain first-order IIR + slew limit."""
    desired = float(np.clip(desired, p.gain_min, p.gain_max))
    if previous is None or not math.isfinite(previous) or previous <= 0:
        return desired
    err = math.log2(desired) - math.log2(previous)
    deadband = p.deadband_at(previous)
    if abs(err) <= deadband:
        return float(previous)
    err -= math.copysign(deadband, err)
    step = float(np.clip(p.alpha * err, -p.max_step_stops, p.max_step_stops))
    return float(np.clip(previous * 2.0 ** step, p.gain_min, p.gain_max))
