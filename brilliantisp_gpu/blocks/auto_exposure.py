"""GPU AE meter + direct gain suggestion (native scene-linear path).

Mirrors AutoExposure._native_exposure_feedback and
suggest_direct_gain_index: f64 Rec.709 luminance clamped below at 0,
true-division SUM/count mean (torch.mean's reciprocal averaging differs),
skewness centered on the configured center with max-|delta| scaling and
Fisher factor, then thresholded feedback. Meter always runs — the
enable/disable decision belongs to the engine. Only scalar averages,
moments, and statuses cross to host.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from util.color import _coefficients

from ..types import GpuError, require_torch


@dataclass(frozen=True)
class AeMeasurement:
    feedbacks: tuple
    meter_averages: tuple
    sample_counts: tuple
    skewness: tuple


def _validate_params(params: dict) -> tuple:
    center = params.get("center_illuminance", 0.5)
    target = params.get("target_luminance")
    upper = params.get("histogram_skewness", 0.9)
    try:
        center = float(center)
    except (TypeError, ValueError) as exc:
        raise ValueError("native AE center must be finite nonnegative") from exc
    if target is not None:
        try:
            t = float(target)
        except (TypeError, ValueError) as exc:
            raise ValueError("native AE target must be finite nonnegative") from exc
        if not math.isfinite(t) or t < 0:
            raise ValueError("native AE target must be finite nonnegative")
    try:
        upper = float(upper)
    except (TypeError, ValueError) as exc:
        raise ValueError("histogram_skewness must be finite nonnegative") from exc
    if not math.isfinite(center) or center < 0:
        raise ValueError("native AE center must be finite nonnegative")
    if not math.isfinite(upper) or upper < 0:
        raise ValueError("histogram_skewness must be finite nonnegative")
    return center, (float(target) if target is not None else None), upper


def _meter_frame(rgb, center, upper, torch):
    """(avg, count, fb, skew) for one HWC3 frame on device."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    n = int(r.numel())
    if n == 0:
        return None, 0, None, None
    kr, kg, kb = _coefficients("bt709")
    y = ((r * kr + g * kg) + b * kb).clamp(min=0)
    count = torch.as_tensor(float(n), dtype=torch.float64, device=rgb.device)
    mean = float(y.sum() / count)
    if mean <= 0:
        return mean, n, None, None
    delta = y - center
    scale = float(delta.abs().max())
    if scale == 0:
        skew = 0.0
    else:
        u = delta / torch.as_tensor(scale, dtype=torch.float64, device=rgb.device)
        uu = u * u
        m2 = float(uu.sum() / count)
        m3 = float((uu * u).sum() / count)
        g1 = math.sqrt(n * (n - 1)) / (n - 2) if n > 2 else 1.0
        skew = (m3 / abs(m2) ** 1.5) * g1 if m2 != 0 else 0.0
    fb = -1 if skew < -upper else (1 if skew > upper else 0)
    return mean, n, fb, skew


def auto_exposure(image, params: dict) -> AeMeasurement:
    """BHWC3 scene-linear RGB -> per-frame meter feedback."""
    torch = require_torch()
    arr = torch.as_tensor(image)
    if arr.ndim != 4 or arr.shape[-1] != 3 or arr.shape[0] < 1:
        raise GpuError(f"auto_exposure expects nonempty BHWC3 input, got {tuple(arr.shape)}")
    if arr.dtype == torch.bool or arr.is_complex():
        raise GpuError(f"auto_exposure expects real input, got {arr.dtype}")
    if not bool(torch.isfinite(arr).all()):
        raise GpuError("native AE input must be finite")
    center, target, upper = _validate_params(params)
    rgb = arr.to(torch.float64)
    feedbacks, averages, counts, skews = [], [], [], []
    for i in range(arr.shape[0]):
        avg, cnt, fb, skew = _meter_frame(rgb[i], center, upper, torch)
        averages.append(avg)
        counts.append(cnt)
        feedbacks.append(fb)
        skews.append(skew)
    return AeMeasurement(tuple(feedbacks), tuple(averages),
                         tuple(counts), tuple(skews))


def suggest_direct_gain_index(current_gain_index: int, gain_array: list,
                              meter_average: float | None, params: dict) -> int:
    """Nearest declared-gain index toward target (absolute ratio)."""
    if meter_average is None or not gain_array or meter_average <= 0:
        return int(current_gain_index)
    raw = params.get("target_luminance")
    target = 0.5 if raw is None else float(raw)
    arr = np.asarray(gain_array, dtype=np.float64)
    ratio = target / float(meter_average)
    gi = int(np.clip(int(current_gain_index), 0, len(gain_array) - 1))
    desired = float(np.clip(float(gain_array[gi]) * ratio,
                            float(arr.min()), float(arr.max())))
    return int(np.argmin(np.abs(arr - desired)))
