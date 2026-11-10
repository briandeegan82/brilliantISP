"""Native normalized-domain tone-mapping entry (M4).

Accepts an already-normalized float Bayer plane or RGB frame in 0..1
linear units and returns a float32 mapped image of the same shape. The
fixed backend bridges through an explicit Q8.24 code domain (reference
2**24, physical max 2**32-1) with the declared-normalization substitute —
no image-derived scale and no diagnostics/files.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from modules.tone_mapping.backends import map_luminance
from modules.tone_mapping.normalization import FrameStats, normalize_luminance
from modules.tone_mapping.registry import ToneMappingSpec, ToneMappingError
from util.color import luminance as _rgb_luma
from util.native_array import as_float32

_Q24 = float(2**24)
_U32_MAX = 2**32 - 1
_RGB_Y_FLOOR = 1e-12


def _native_float_source(image: np.ndarray) -> np.ndarray:
    src = np.asarray(image)
    if src.ndim not in (2, 3) or src.size == 0:
        raise ToneMappingError(
            f"native input must be a nonempty 2-D Bayer or HWC3 RGB frame, got {src.shape}"
        )
    if src.ndim == 3 and src.shape[-1] != 3:
        raise ToneMappingError(f"native RGB input needs last axis 3, got {src.shape}")
    if src.dtype.kind != "f":
        raise ToneMappingError(f"native input must be floating point, got {src.dtype}")
    if not np.all(np.isfinite(src)):
        raise ToneMappingError("native input must be finite")
    return src


def _map_plane(
    y: np.ndarray, spec: ToneMappingSpec, frame_stats: FrameStats | None
) -> np.ndarray:
    """Map a nonnegative normalized luminance plane to normalized output."""
    if spec.backend == "fixed":
        gain = 2.0**spec.exposure_ev * spec.domain_scale
        cap = spec.lut_domain / gain
        u_cap = min(1.0, cap) if spec.normalize_output else cap
        normalized = normalize_luminance(
            y, normalization=spec.normalization, input_max=1.0, stats=frame_stats
        )
        saturated = np.minimum(normalized, u_cap)
        coord_max = _U32_MAX / _Q24
        if float(np.max(saturated)) > coord_max:
            raise ToneMappingError(
                "normalized coordinate exceeds the representable fixed "
                "Q8.24 domain (uncapped value > (2^32-1)/2^24)"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            coord = saturated * _Q24
        if not np.all(np.isfinite(coord)) or float(np.max(coord)) > _U32_MAX:
            raise ToneMappingError(
                "normalized coordinate exceeds the representable fixed "
                "Q8.24 domain (uncapped value > (2^32-1)/2^24)"
            )
        codes = np.floor(coord + 0.5).astype(np.uint32)
        sub = dataclasses.replace(spec, normalization="declared_range")
        return np.asarray(
            map_luminance(
                codes,
                input_max=_U32_MAX,
                input_reference=_Q24,
                output_max=65535,
                config=sub,
            ),
            dtype=np.float64,
        ) / 65535.0
    mapped = map_luminance(
        np.asarray(y, dtype=np.float64),
        input_max=1,
        output_max=65535,
        config=spec,
        frame_stats=frame_stats,
    )
    return np.asarray(mapped, dtype=np.float64) / 65535.0


def map_native(
    image: np.ndarray,
    *,
    config: ToneMappingSpec,
    frame_stats: FrameStats | None = None,
) -> np.ndarray:
    """Map a normalized float Bayer/RGB image through the declared spec.

    Disabled specs return the same object before any work. RGB luminance
    is max(0, 709-weights dot RGB); Bayer is max(0, input). RGB output is
    recombined on the original positive-Y ratio (zero/near-black maps to
    black), preserving component negatives/headroom — no gamut clipping.
    """
    if not config.enabled:
        return image
    src = _native_float_source(image)
    if src.ndim == 2:
        y = np.maximum(src.astype(np.float64), 0.0)
        mapped = _map_plane(y, config, frame_stats)
        try:
            return as_float32(mapped, name="native_bayer_tone_map")
        except ValueError as exc:
            raise ToneMappingError(str(exc)) from exc
    rgb = src.astype(np.float64)
    y = np.maximum(_rgb_luma(rgb, dtype=np.float64), 0.0)
    mapped_y = _map_plane(y, config, frame_stats)
    safe = np.where(y > _RGB_Y_FLOOR, y, 1.0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        out = rgb * (mapped_y / safe)[..., None]
    out[y <= _RGB_Y_FLOOR] = 0.0
    try:
        return as_float32(out, name="native_rgb_tone_map")
    except ValueError as exc:
        raise ToneMappingError(str(exc)) from exc
