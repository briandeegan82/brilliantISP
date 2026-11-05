"""Frame statistics, normalization, and safe RGB luma/recombination.

Normalization is declared-range or per-frame only — bit depth is never
inferred from observed data. ``FrameStats`` preserves exact integer or
float extrema. Integer normalization widens to signed 64-bit before
subtracting so below-minimum codes cannot wrap positive.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

from modules.tone_mapping.registry import ToneMappingError

FLOAT_LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)
LUMA_WEIGHTS = (2126, 7152, 722)
LUMA_DENOMINATOR = 10000


def _real_extreme(name: str, value: Any) -> int | float:
    if isinstance(value, bool):
        raise ValueError(f"FrameStats.{name} must be a finite number, got {value!r}")
    if isinstance(value, (int, np.integer)):
        try:
            representable = math.isfinite(float(value))
        except OverflowError:
            representable = False
        if not representable:
            raise ValueError(f"FrameStats.{name} is not representable in float64")
        return int(value)
    if isinstance(value, (float, np.floating)):
        out = float(value)
        if not math.isfinite(out):
            raise ValueError(f"FrameStats.{name} must be finite, got {value!r}")
        return out
    raise ValueError(f"FrameStats.{name} must be a finite number, got {value!r}")


@dataclass(frozen=True)
class FrameStats:
    """Whole-frame extrema; integers for integer frames, exact floats otherwise."""

    minimum: int | float
    maximum: int | float

    def __post_init__(self) -> None:
        lo = _real_extreme("minimum", self.minimum)
        hi = _real_extreme("maximum", self.maximum)
        if hi < lo:
            raise ValueError(
                f"FrameStats must be ordered: minimum {lo} > maximum {hi}"
            )
        object.__setattr__(self, "minimum", lo)
        object.__setattr__(self, "maximum", hi)


def frame_stats(luminance: np.ndarray) -> FrameStats:
    """Whole-frame extrema; never infers bit depth or truncates floats."""
    arr = np.asarray(luminance)
    if arr.size == 0:
        raise ValueError("cannot derive stats from an empty frame")
    if arr.dtype.kind not in "uif":
        raise ValueError(f"unsupported dtype {arr.dtype} for frame stats")
    lo = arr.min()
    hi = arr.max()
    if arr.dtype.kind == "f":
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError("frame extrema must be finite")
        return FrameStats(float(lo), float(hi))
    return FrameStats(int(lo), int(hi))


def _integral_bound(name: str, value: int | float) -> int:
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(
                f"integer normalization requires integral bounds; "
                f"{name}={value} is fractional"
            )
        value = int(value)
    out = int(value)
    if out < 0 or out > 2**32 - 1:
        raise ValueError(f"{name}={value} is outside 0..2**32-1")
    return out


def normalize_declared(luminance: np.ndarray, input_max: float) -> np.ndarray:
    """Float64 normalized by the declared code maximum; headroom preserved."""
    with np.errstate(over="ignore", invalid="ignore"):
        out = np.maximum(np.asarray(luminance, dtype=np.float64), 0.0) / float(input_max)
    if not np.all(np.isfinite(out)):
        raise ValueError("declared-range normalization is not representable in float64")
    return out


def normalize_per_frame(
    luminance: np.ndarray, stats: FrameStats
) -> np.ndarray | None:
    """Per-frame [0,1] normalization; None when the frame is constant."""
    span = float(stats.maximum) - float(stats.minimum)
    if not math.isfinite(span):
        raise ValueError("per-frame normalization span is not representable in float64")
    if span <= 0:
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        bounded = np.clip(np.asarray(luminance, dtype=np.float64), float(stats.minimum), float(stats.maximum))
        out = (bounded - float(stats.minimum)) / span
    if not np.all(np.isfinite(out)):
        raise ValueError("per-frame normalization is not representable in float64")
    return np.clip(out, 0.0, 1.0)


def normalize_luminance(
    luminance: np.ndarray,
    *,
    normalization: str,
    input_max: float,
    stats: FrameStats | None,
) -> np.ndarray:
    """Normalize once per the declared policy.

    Per-frame uses supplied stats (reused unchanged for ROI calls) or
    whole-frame stats when omitted; a constant frame falls back to
    declared_range rather than producing all-black output.
    """
    if normalization == "per_frame":
        use = stats if stats is not None else frame_stats(np.asarray(luminance))
        per = normalize_per_frame(np.asarray(luminance), use)
        if per is not None:
            return per
    return normalize_declared(np.asarray(luminance), input_max)


def prepare_curve_input(
    luminance: np.ndarray,
    *,
    input_max: float,
    normalization: str,
    gain: float,
    lut_domain: float,
    stats: FrameStats | None = None,
) -> np.ndarray:
    """Shared normalized -> gain -> domain-saturated curve input.

    The multiply is masked at ``threshold = lut_domain / gain`` so positive
    values at or above the threshold saturate to the domain without
    overflowing to inf; threshold underflow to zero saturates every
    positive input. Nonrepresentable normalization errors propagate.
    """
    normalized = normalize_luminance(
        luminance, normalization=normalization, input_max=input_max, stats=stats
    )
    if not math.isfinite(gain) or gain <= 0 or not math.isfinite(lut_domain) or lut_domain <= 0:
        raise ValueError(
            f"gain and lut_domain must be finite positive, got {gain!r}/{lut_domain!r}"
        )
    threshold = lut_domain / gain
    if threshold == 0.0:
        return np.where(normalized > 0.0, lut_domain, 0.0)
    out = np.full(normalized.shape, lut_domain, dtype=np.float64)
    hot = normalized >= threshold
    np.multiply(normalized, gain, out=out, where=~hot)
    return np.clip(out, 0.0, lut_domain)


def require_code_ranges(input_max: Any, output_max: Any) -> tuple[float, int]:
    """Validate declared scalar code ranges for all public backends."""
    if isinstance(input_max, bool) or not isinstance(input_max, (int, float, np.integer, np.floating)):
        raise ToneMappingError("input_max must be a positive finite number <= 2**32-1")
    try:
        maximum = float(input_max)
    except OverflowError as exc:
        raise ToneMappingError("input_max is not representable in float64") from exc
    if not math.isfinite(maximum) or not 0 < maximum <= 2**32 - 1:
        raise ToneMappingError("input_max must be in (0, 2**32-1]")
    if isinstance(output_max, bool) or not isinstance(output_max, (int, np.integer)) or not 1 <= output_max <= 65535:
        raise ToneMappingError("output_max must be an integer in 1..65535")
    return maximum, int(output_max)


def require_input_reference(input_reference: Any, input_max: float) -> float:
    """Resolve the optional white-reference denominator.

    ``input_max`` remains the physical unsigned-code bound; the reference
    defaults to it and must be a positive finite numeric (non-bool) no
    larger than the physical maximum. Errors name ``input_reference``.
    """
    if input_reference is None:
        return float(input_max)
    if isinstance(input_reference, bool) or not isinstance(
        input_reference, (int, float, np.integer, np.floating)
    ):
        raise ToneMappingError(
            f"input_reference must be a positive finite number <= input_max, "
            f"got {input_reference!r}"
        )
    try:
        ref = float(input_reference)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ToneMappingError(
            f"input_reference is not representable in float64, got {input_reference!r}"
        ) from exc
    if not math.isfinite(ref) or not 0 < ref <= input_max:
        raise ToneMappingError(
            f"input_reference must be in (0, {input_max}], got {input_reference!r}"
        )
    return ref


def require_integral_reference(
    input_reference: Any, reference: float, input_max: float
) -> int:
    """Integral denominator for unsigned integer backends.

    An explicit fractional reference is a named ``input_reference`` error;
    when omitted, the legacy ``input_max`` message is preserved.
    """
    if input_reference is None:
        if not float(input_max).is_integer():
            raise ToneMappingError(
                "integer backends require an integral input_max (no silent truncation)"
            )
        return int(input_max)
    if not float(reference).is_integer():
        raise ToneMappingError(
            f"integer backends require an integral input_reference (no "
            f"silent truncation), got {input_reference!r}"
        )
    return int(reference)


def code_max_from_bits(sensor_info: Any, key: str, default: int, limit: int) -> int:
    """Return the declared code maximum without truncating fractional bits."""
    value = sensor_info.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ToneMappingError(f"sensor_info.{key} must be an integer in 1..{limit}")
    if not 1 <= value <= limit or int(value) != value:
        raise ToneMappingError(f"sensor_info.{key} must be an integer in 1..{limit}")
    return (1 << int(value)) - 1


def rgb_luminance_numerator(rgb: np.ndarray) -> np.ndarray:
    """Exact uint64 luminance numerator for unsigned RGB code values."""
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[-1] != 3 or arr.dtype.kind != "u" or arr.dtype.itemsize not in (1, 2, 4):
        raise ToneMappingError("integer luminance requires uint8/16/32 RGB with shape (H,W,3)")
    u = arr.astype(np.uint64)
    return u[..., 0] * LUMA_WEIGHTS[0] + u[..., 1] * LUMA_WEIGHTS[1] + u[..., 2] * LUMA_WEIGHTS[2]


def validate_tone_source(luminance: np.ndarray, *, for_integer: bool) -> np.ndarray:
    """Shared source validation for every tone-mapping entry point."""
    src = np.asarray(luminance)
    if src.size == 0:
        raise ValueError("tone-mapping input must not be empty")
    if src.dtype.kind not in "uif":
        raise ValueError(
            f"unsupported input dtype {src.dtype}; expected real numeric "
            "unsigned integer, signed integer, or float"
        )
    if src.dtype.kind == "f" and not np.all(np.isfinite(src)):
        raise ValueError("tone-mapping input must contain only finite values")
    if src.dtype.kind == "u" and src.dtype.itemsize not in (1, 2, 4):
        raise ValueError(f"unsigned integer input must be uint8/16/32, got {src.dtype}")
    if for_integer and src.dtype.kind != "u":
        raise ValueError(
            f"integer backends require unsigned integer input, got {src.dtype}"
        )
    return src


def check_code_range(src: np.ndarray, input_max: float, error_type: type[Exception]) -> None:
    """Unsigned source codes must not exceed the declared code maximum."""
    if src.dtype.kind == "u" and int(np.max(src)) > input_max:
        raise error_type(
            "unsigned source codes exceed the declared input_max; "
            "declared code range is authoritative"
        )


def normalize_integer(
    luminance: np.ndarray,
    *,
    normalization: str,
    input_max: int,
    stats: FrameStats | None,
) -> tuple[np.ndarray, int]:
    """Integer normalization: returns (uint64 numerator, int denominator).

    Values are widened to signed 64-bit before any subtraction so codes
    below the per-frame minimum clip to zero rather than wrapping. Stats
    bounds must be integral (integer-valued floats accepted); denominators
    never exceed 2**32-1. Constant frames fall back to the declared
    numerator/denominator.
    """
    x = np.asarray(luminance)
    if x.dtype.kind != "u" or x.dtype.itemsize not in (1, 2, 4):
        raise ValueError(
            f"integer normalization requires unsigned uint8/16/32 input, "
            f"got dtype {x.dtype}"
        )
    if normalization == "per_frame":
        use = stats if stats is not None else frame_stats(x)
        lo = _integral_bound("stats.minimum", use.minimum)
        hi = _integral_bound("stats.maximum", use.maximum)
        span = hi - lo
        if span > 0:
            shifted = x.astype(np.int64) - np.int64(lo)
            clipped = np.clip(shifted, 0, np.int64(span)).astype(np.uint64)
            return clipped, span
    return x.astype(np.uint64), int(input_max)


def rgb_luminance(rgb: np.ndarray) -> np.ndarray:
    """Float64 code-space luminance.

    Unsigned sources use the uint64 weighted sum
    ``(2126*R + 7152*G + 722*B) / 10000`` before division (uint8/16/32 only,
    so products cannot overflow); float sources use the normalized weights
    to avoid scaling up finite magnitudes.
    """
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected an (H,W,3) RGB array, got shape {arr.shape}")
    if arr.dtype.kind == "f":
        if not np.all(np.isfinite(arr)):
            raise ValueError("RGB input must contain only finite values")
        f = arr.astype(np.float64)
        return (
            f[..., 0] * FLOAT_LUMA_WEIGHTS[0]
            + f[..., 1] * FLOAT_LUMA_WEIGHTS[1]
            + f[..., 2] * FLOAT_LUMA_WEIGHTS[2]
        )
    if arr.dtype.kind == "i":
        f = arr.astype(np.float64)
        if not np.all(np.isfinite(f)):
            raise ValueError("RGB input must contain only finite values")
        return (
            f[..., 0] * FLOAT_LUMA_WEIGHTS[0]
            + f[..., 1] * FLOAT_LUMA_WEIGHTS[1]
            + f[..., 2] * FLOAT_LUMA_WEIGHTS[2]
        )
    if arr.dtype.kind != "u" or arr.dtype.itemsize not in (1, 2, 4):
        raise ValueError(
            f"unsupported RGB dtype {arr.dtype}; expected uint8/16/32, "
            "signed integer, or float"
        )
    weighted = rgb_luminance_numerator(arr)
    return weighted.astype(np.float64) / LUMA_DENOMINATOR


def recombine(
    mapped_luminance: np.ndarray,
    rgb: np.ndarray,
    luminance: np.ndarray,
    *,
    output_max: int,
) -> np.ndarray:
    """Recombine mapped code luminance with the original RGB ratio.

    The denominator is floored at ``max(min subnormal, output_max*1e-12)``;
    zero/below-floor luminance pixels map to black. Mapped luminance is
    clamped to the output range before scaling, products may saturate to
    +inf intentionally (clipped afterwards), and the clip happens BEFORE
    the unsigned cast.
    """
    _, output_max = require_code_ranges(1, output_max)
    y = np.asarray(luminance, dtype=np.float64)
    mapped = np.asarray(mapped_luminance, dtype=np.float64)
    base = np.asarray(rgb, dtype=np.float64)
    if base.ndim != 3 or base.shape[2] != 3 or base.shape[:2] != y.shape or y.shape != mapped.shape:
        raise ValueError(
            f"shape mismatch: luminance {y.shape}, mapped {mapped.shape}, "
            f"rgb {base.shape}"
        )
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(base)) and np.all(np.isfinite(mapped))):
        raise ValueError("recombine inputs must be finite")
    floor = max(sys.float_info.min, float(output_max) * 1e-12)
    mapped = np.clip(mapped, 0.0, float(output_max))
    usable = y >= floor
    denom = np.where(usable, y, 1.0)
    scale = mapped / denom
    scale = np.clip(scale, 0.0, np.finfo(np.float64).max)
    with np.errstate(over="ignore"):
        out = base * scale[..., None]
    out[~usable] = 0.0
    clipped = np.clip(out, 0.0, float(output_max))
    if not np.all(np.isfinite(clipped)):
        raise ValueError("recombine produced nonfinite output")
    return clipped.astype(np.uint16)
