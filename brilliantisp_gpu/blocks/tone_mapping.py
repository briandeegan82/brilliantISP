"""GPU ports of the fixed-point Q32 tone backend and native Q8.24 bridge.

Mirrors ``modules.tone_mapping.backends._fixed_reinhard`` and the fixed
branch of ``modules.tone_mapping.native._map_plane`` bit-exactly using the
U64 helpers; no double division approximates the integer recipes.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from modules.tone_mapping.normalization import (
    require_code_ranges,
    require_input_reference,
    require_integral_reference,
    _integral_bound,
)
from modules.tone_mapping.registry import ToneMappingSpec, ToneMappingError

from modules.tone_mapping.normalization import FrameStats
from modules.tone_mapping.backends import (
    _LUT_CACHE, _table_domain, _theoretical_reference)
from modules.tone_mapping import operators as _ops
from .durand import _durand_plane

from ..types import GpuError, require_torch
from ..ops import uint64
from ..ops.uint64 import U64
from ..ops.array import checked_float32, codes_to_int64
from ..ops.color import luminance

M = 2**32 - 1
_Q24 = float(2**24)
_U32_MAX = 2**32 - 1


def _q32_coordinate(n: U64, d: int, r: float, ref):
    """Saturating Q32 coordinate for ratio r per the integer recipe."""
    torch = require_torch()
    if d <= 0:
        raise ToneMappingError(f"normalization denominator must be positive, got {d}")
    if not math.isfinite(r) or r < 0:
        raise ToneMappingError(f"coordinate ratio must be finite nonnegative, got {r!r}")
    zero = uint64.zeros_like(n)
    if r == 0:
        return zero
    all_m = uint64.from_scalar(M, n)
    n_pos = uint64.gt(n, zero)
    if r >= 2**64 / M:
        return uint64.where(n_pos, all_m, zero)
    c = round(r * M)
    if c <= 0:
        return zero
    if c > 2**64 - 1:
        return uint64.where(n_pos, all_m, zero)
    threshold = (d * M) // c
    clipped = uint64.minimum(n, uint64.from_scalar(threshold, n))
    q, _ = uint64.divmod_u33(uint64.multiply(clipped, uint64.from_scalar(c, n)), d)
    return uint64.where(uint64.gt(n, uint64.from_scalar(threshold, n)), all_m, q)


def _normalize_integer(codes, *, normalization: str, ref_int: int, stats):
    """Integer (numerator U64, denominator int) per normalize_integer."""
    torch = require_torch()
    if normalization == "per_frame" and stats is not None:
        lo = _integral_bound("stats.minimum", stats.minimum)
        hi = _integral_bound("stats.maximum", stats.maximum)
        span = hi - lo
        if span > 0:
            shifted = codes - lo
            clipped = torch.clamp(shifted, 0, span)
            return uint64.from_nonnegative_int64(clipped), span
    return uint64.from_nonnegative_int64(codes), int(ref_int)


class _Stats:
    __slots__ = ("minimum", "maximum")

    def __init__(self, minimum, maximum):
        self.minimum = minimum
        self.maximum = maximum


def _fixed_reinhard_plane(codes, spec: ToneMappingSpec, int_max: int, ref: float,
                          ref_int: int, output_max: int, stats):
    """Single-frame fixed backend; ``codes`` is a 2-D int64 tensor."""
    torch = require_torch()
    params = spec.parameters_map()
    gain = 2.0**spec.exposure_ev * spec.domain_scale
    use = stats
    if spec.normalization == "per_frame" and use is None:
        use = _Stats(int(codes.min()), int(codes.max()))
    nonconstant = False
    if spec.normalization == "per_frame":
        lo = _integral_bound("stats.minimum", use.minimum)
        hi = _integral_bound("stats.maximum", use.maximum)
        nonconstant = hi - lo > 0
    n, d = _normalize_integer(
        codes, normalization=spec.normalization, ref_int=ref_int, stats=use
    )
    H = 1.0 if nonconstant else int_max / ref
    if spec.normalize_output:
        u_ref = min(1.0, spec.lut_domain / gain)
    else:
        u_ref = min(H, spec.lut_domain / gain)
    effective_gain = gain * u_ref
    x = _q32_coordinate(n, d, 1.0 / u_ref, codes)
    w = (params["knee"] / params["strength"]) / effective_gain
    if not math.isfinite(w) or w <= 0:
        raise ToneMappingError(
            "knee/(strength*effective_gain) must be representable positive"
        )
    zero = uint64.zeros_like(x)
    one = uint64.from_scalar(1, x)
    m_u = uint64.from_scalar(M, x)
    if w <= 1.0:
        W = round(w * M)
        denom = uint64.add(x, uint64.from_scalar(W, x))
        denom = uint64.where(uint64.eq(denom, zero), one, denom)
        if spec.normalize_output:
            half = (M + W + 1) // 2
            y, _ = uint64.divmod_u33(uint64.mul_scalar(x, half), uint64.to_int64(denom))
            yden = (M + 1) // 2
        else:
            y, _ = uint64.divmod_u33(uint64.mul_scalar(x, M), uint64.to_int64(denom))
            yden = M
    else:
        Hq = round(M / w)
        z, _ = uint64.divmod_u33(uint64.mul_scalar(x, Hq), M)
        if spec.normalize_output:
            half = (M + Hq + 1) // 2
            y, _ = uint64.divmod_u33(
                uint64.mul_scalar(x, half), uint64.to_int64(uint64.add(m_u, z))
            )
            yden = (M + 1) // 2
        else:
            y, _ = uint64.divmod_u33(
                uint64.mul_scalar(z, M), uint64.to_int64(uint64.add(m_u, z))
            )
            yden = M
    num = uint64.add(uint64.mul_scalar(y, output_max), uint64.from_scalar(yden // 2, y))
    out, _ = uint64.divmod_u33(num, yden)
    return uint64.to_int64(uint64.minimum(out, uint64.from_scalar(output_max, out)))


def fixed_reinhard(codes, *, config: ToneMappingSpec, input_max,
                   output_max: int = 65535, input_reference=None):
    """Fixed Q32 Reinhard backend on int64 logical-unsigned codes.

    ``codes`` is an HW or BHW int64 tensor of unsigned code values in
    0..2**32-1 (per-frame independent statistics for BHW). Returns the
    int64 logical uint16 output codes, matching
    ``backends._fixed_reinhard`` exactly.
    """
    torch = require_torch()
    src = codes_to_int64(codes, name="fixed_reinhard.codes")
    if src.ndim not in (2, 3):
        raise ToneMappingError(
            f"fixed_reinhard expects HW or BHW codes, got {tuple(src.shape)}"
        )
    maximum, output_max = require_code_ranges(input_max, output_max)
    if not float(maximum).is_integer():
        raise ToneMappingError(
            "integer backends require an integral input_max (no silent truncation)"
        )
    int_max = int(maximum)
    if int(src.max()) > int_max:
        raise ToneMappingError(
            "unsigned source codes exceed the declared input_max; "
            "declared code range is authoritative"
        )
    reference = require_input_reference(input_reference, maximum)
    ref_int = require_integral_reference(input_reference, reference, maximum)
    if src.ndim == 2:
        return _fixed_reinhard_plane(
            src, config, int_max, reference, ref_int, output_max, None
        )
    out = torch.empty(src.shape, dtype=torch.int64, device=src.device)
    for b in range(src.shape[0]):
        out[b] = _fixed_reinhard_plane(
            src[b], config, int_max, reference, ref_int, output_max, None
        )
    return out


def _normalize_plane_f64(y, normalization: str, stats):
    """Per-plane float64 normalize_luminance with input_max=1.0."""
    torch = require_torch()
    if normalization == "per_frame":
        use = stats
        if use is None:
            use = FrameStats(float(y.min()), float(y.max()))
        lo = float(use.minimum)
        hi = float(use.maximum)
        span = hi - lo
        if not math.isfinite(span):
            raise ValueError("per-frame normalization span is not representable in float64")
        if span > 0:
            bounded = torch.clamp(y, lo, hi)
            out = (bounded - lo) / torch.as_tensor(
                span, dtype=torch.float64, device=y.device)
            if not bool(torch.isfinite(out).all()):
                raise ValueError("per-frame normalization is not representable in float64")
            return torch.clamp(out, 0.0, 1.0)
    out = torch.clamp(y, min=0.0) / 1.0
    if not bool(torch.isfinite(out).all()):
        raise ValueError("declared-range normalization is not representable in float64")
    return out


def _prepare_curve_input_f64(y, spec: ToneMappingSpec, gain: float, stats, torch):
    normalized = _normalize_plane_f64(y, spec.normalization, stats)
    if not math.isfinite(gain) or gain <= 0 or not math.isfinite(spec.lut_domain) or spec.lut_domain <= 0:
        raise ValueError(
            f"gain and lut_domain must be finite positive, got {gain!r}/{spec.lut_domain!r}")
    threshold = spec.lut_domain / gain
    if threshold == 0.0:
        return torch.where(normalized > 0.0,
                           torch.full_like(normalized, spec.lut_domain),
                           torch.zeros_like(normalized))
    hot = normalized >= threshold
    scaled = normalized * gain
    out = torch.where(hot, torch.full_like(normalized, spec.lut_domain), scaled)
    return torch.clamp(out, 0.0, spec.lut_domain)


def _curve_f64(spec: ToneMappingSpec, x, torch):
    params = spec.parameters_map()
    x = torch.clamp(x, min=0.0)
    if spec.operator == "aces_fitted":
        num = x * (2.51 * x + 0.03)
        den = x * (2.43 * x + 0.59) + 0.14
        return torch.clamp(num / den, 0.0, 1.0)
    if spec.operator == "reinhard":
        white = params["knee"] / params["strength"]
        if not math.isfinite(white) or white <= 0:
            raise ValueError("knee/strength must have a representable positive ratio")
        return torch.clamp(x / (x + white), 0.0, 1.0)
    if spec.operator == "hable":
        bias = float(params["exposure_bias"])
        reference = float(_ops.hable_partial(np.asarray([params["white_point"]]))[0])
        if not math.isfinite(reference) or reference <= 0:
            raise ValueError("white_point has no representable positive Hable reference")
        a = 0.15 * (1.0 - 0.02 / 0.30)
        b = 0.50 * (0.10 - 0.02 / 0.30)
        xb = x * bias
        ref_d = torch.as_tensor(reference, dtype=torch.float64, device=x.device)
        mapped = (xb * (a * xb + b) / (xb * (0.15 * xb + 0.50) + 0.20 * 0.30)) / ref_d
        return torch.clamp(mapped, 0.0, 1.0)
    raise GpuError(f"{spec.operator} has no scalar GPU curve")


def _map_fixed_plane_f64(y, spec: ToneMappingSpec, stats):
    """Fixed-backend bridge; returns f64 mapped codes/65535 (no f32 cast)."""
    torch = require_torch()
    gain = 2.0**spec.exposure_ev * spec.domain_scale
    cap = spec.lut_domain / gain
    u_cap = min(1.0, cap) if spec.normalize_output else cap
    normalized = _normalize_plane_f64(y, spec.normalization, stats)
    saturated = torch.minimum(
        normalized,
        torch.as_tensor(u_cap, dtype=torch.float64, device=y.device))
    coord_max = _U32_MAX / _Q24
    if float(saturated.max()) > coord_max:
        raise ToneMappingError(
            "normalized coordinate exceeds the representable fixed "
            "Q8.24 domain (uncapped value > (2^32-1)/2^24)"
        )
    coord = saturated * _Q24
    if not bool(torch.isfinite(coord).all()) or float(coord.max()) > _U32_MAX:
        raise ToneMappingError(
            "normalized coordinate exceeds the representable fixed "
            "Q8.24 domain (uncapped value > (2^32-1)/2^24)"
        )
    codes = torch.floor(coord + 0.5).to(torch.int64)
    sub = dataclasses.replace(spec, normalization="declared_range")
    mapped = fixed_reinhard(
        codes,
        config=sub,
        input_max=_U32_MAX,
        input_reference=_Q24,
        output_max=65535,
    )
    return mapped.to(torch.float64) / torch.as_tensor(
        65535.0, dtype=torch.float64, device=y.device)


def _map_fixed_plane(y, spec: ToneMappingSpec, stats):
    return checked_float32(_map_fixed_plane_f64(y, spec, stats),
                           name="native_fixed_tone_map")


def _map_plane_f64(y, spec: ToneMappingSpec, stats, torch, table=None):
    """Nonnegative normalized f64 Y -> normalized f64 mapped codes/65535."""
    if spec.backend == "fixed":
        return _map_fixed_plane_f64(y, spec, stats)
    gain = 2.0**spec.exposure_ev * spec.domain_scale
    x = _prepare_curve_input_f64(y, spec, gain, stats, torch)
    if spec.operator == "durand":
        out = _durand_plane(x, **spec.parameters_map())
        return (out * 65535.0) / torch.as_tensor(
            65535.0, dtype=torch.float64, device=y.device)
    if spec.backend == "lut":
        if table is None:
            table = torch.as_tensor(np.array(_LUT_CACHE.get(spec, 65535)),
                                    dtype=torch.float64, device=y.device)
        domain = _table_domain(spec, gain)
        dev_d = torch.as_tensor(domain, dtype=torch.float64, device=y.device)
        position = torch.minimum(x, dev_d) / dev_d * (spec.lut_size - 1)
        i = torch.floor(position).to(torch.int64)
        f = position - i.to(torch.float64)
        i = torch.clamp(i, 0, spec.lut_size - 1)
        j = torch.clamp(i + 1, max=spec.lut_size - 1)
        return (table[i] * (1.0 - f) + table[j] * f) / torch.as_tensor(
            65535.0, dtype=torch.float64, device=y.device)
    ref = _theoretical_reference(spec, gain)
    ref_d = torch.as_tensor(ref, dtype=torch.float64, device=y.device)
    mapped = _curve_f64(spec, x, torch) / ref_d
    codes = torch.clamp(mapped, 0.0, 1.0) * 65535.0
    return codes / torch.as_tensor(65535.0, dtype=torch.float64, device=y.device)


def map_native(image, *, config: ToneMappingSpec, color_model: str,
               frame_stats=None):
    """Global native tone map for Bayer HW/BHW or RGB HWC3/BHWC3 floats."""
    if not config.enabled:
        return image
    torch = require_torch()
    arr = torch.as_tensor(image)
    model = str(color_model).strip().lower()
    if model == "bayer":
        if arr.ndim not in (2, 3):
            raise ToneMappingError(
                f"native bayer input must be HW or BHW, got {tuple(arr.shape)}")
        batched = arr.ndim == 3
    elif model == "rgb":
        if arr.ndim not in (3, 4) or arr.shape[-1] != 3:
            raise ToneMappingError(
                f"native RGB input must be HWC3 or BHWC3, got {tuple(arr.shape)}")
        batched = arr.ndim == 4
    else:
        raise ValueError(f"unknown color_model {color_model!r}")
    if not arr.numel():
        raise ToneMappingError("native input must be nonempty")
    if arr.dtype == torch.bool or arr.is_complex() or not arr.is_floating_point():
        raise ToneMappingError(f"native input must be floating point, got {arr.dtype}")
    if not bool(torch.isfinite(arr).all()):
        raise ToneMappingError("native input must be finite")
    if config.operator == "durand" and not arr.is_cuda:
        raise GpuError("operator 'durand' requires a CUDA/HIP device tensor")
    b = arr.shape[0] if batched else 1
    if isinstance(frame_stats, FrameStats):
        stats_seq = [frame_stats] * b
    elif frame_stats is None:
        stats_seq = [None] * b
    else:
        stats_seq = list(frame_stats)
        if len(stats_seq) != b:
            raise ToneMappingError(
                f"frame_stats sequence length {len(stats_seq)} != batch {b}")
        for s in stats_seq:
            if s is not None and not isinstance(s, FrameStats):
                raise ToneMappingError(
                    f"frame_stats elements must be FrameStats or None, got {type(s)!r}")
    table = None
    if config.backend == "lut":
        table = torch.as_tensor(np.array(_LUT_CACHE.get(config, 65535)),
                                dtype=torch.float64, device=arr.device)
    frames = arr if batched else arr[None]
    if model == "bayer":
        out = torch.empty(frames.shape, dtype=torch.float32, device=arr.device)
        for i in range(b):
            y = torch.maximum(frames[i].to(torch.float64),
                              torch.as_tensor(0.0, dtype=torch.float64, device=arr.device))
            out[i] = checked_float32(
                _map_plane_f64(y, config, stats_seq[i], torch, table=table),
                name="native_bayer_tone_map")
        return out[0] if not batched else out
    rgb = frames.to(torch.float64)
    out = torch.empty(frames.shape, dtype=torch.float32, device=arr.device)
    for i in range(b):
        y = torch.maximum(luminance(frames[i], dtype=np.float64),
                          torch.as_tensor(0.0, dtype=torch.float64, device=arr.device))
        mapped_y = _map_plane_f64(y, config, stats_seq[i], torch, table=table)
        safe = torch.where(y > 1e-12, y, torch.ones_like(y))
        res = rgb[i] * (mapped_y / safe)[..., None]
        res = torch.where((y <= 1e-12)[..., None], torch.zeros_like(res), res)
        out[i] = checked_float32(res, name="native_rgb_tone_map")
    return out[0] if not batched else out


def map_fixed_native(y, *, config: ToneMappingSpec):
    """Native normalized-domain float32 map for the fixed backend.

    ``y`` is a finite floating HW or BHW plane in normalized units; each
    frame's statistics are computed independently. Mirrors the
    ``backend == 'fixed'`` branch of ``native._map_plane``.
    """
    torch = require_torch()
    arr = torch.as_tensor(y)
    if arr.ndim not in (2, 3) or arr.numel() == 0:
        raise ToneMappingError(
            f"native input must be a nonempty HW or BHW plane, got {tuple(arr.shape)}"
        )
    if not arr.is_floating_point():
        raise ToneMappingError(f"native input must be floating point, got {arr.dtype}")
    src = arr.to(torch.float64)
    if not bool(torch.isfinite(src).all()):
        raise ToneMappingError("native input must be finite")
    if arr.ndim == 2:
        plane = torch.maximum(src, torch.as_tensor(0.0, dtype=torch.float64, device=src.device))
        stats = None
        if config.normalization == "per_frame":
            stats = FrameStats(float(plane.min()), float(plane.max()))
        return _map_fixed_plane(plane, config, stats)
    out = torch.empty(src.shape, dtype=torch.float32, device=src.device)
    for b in range(src.shape[0]):
        plane = torch.maximum(src[b], torch.as_tensor(0.0, dtype=torch.float64, device=src.device))
        stats = None
        if config.normalization == "per_frame":
            stats = FrameStats(float(plane.min()), float(plane.max()))
        out[b] = _map_fixed_plane(plane, config, stats)
    return out
