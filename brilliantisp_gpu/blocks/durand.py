"""GPU Durand local tone-mapping plane primitive.

Mirrors modules.tone_mapping.local_durand._durand_plane: log10 domain,
INTER_AREA downsample, separable gaussian base layer, edge-weight
reconstruction, contrast-compressed exponent.
"""

from __future__ import annotations

import math

import numpy as np

from modules.tone_mapping.registry import ToneMappingError

from ..types import GpuError, require_torch
from ..ops.filters import correlate
from ..ops.reference_math import reference_log10_float32
from ..ops.resize_float32 import area_float32
from .scaling import _linear_native_f32


def _gaussian_float32(x, sigma):
    """Separable reflect-gaussian: f64 accumulate, f32 after each axis."""
    torch = require_torch()
    if sigma <= 1e-15:
        return x
    radius = int(4.0 * sigma + 0.5)
    p = np.arange(-radius, radius + 1)
    weights = np.exp(-0.5 / (sigma * sigma) * p ** 2)
    weights = weights / weights.sum()
    k = torch.as_tensor(weights, dtype=torch.float64, device=x.device)
    tmp = correlate(x, k[:, None]).to(torch.float32)
    return correlate(tmp, k[None, :]).to(torch.float32)


def _durand_plane(curve_input, *, sigma_space, sigma_color, contrast_factor,
                  downsample_factor):
    """HW normalized f64 curve input -> HW f64 normalized output."""
    torch = require_torch()
    arr = curve_input if isinstance(curve_input, torch.Tensor)         else torch.as_tensor(curve_input)
    if not arr.is_cuda:
        raise GpuError(
            "_durand_plane requires a CUDA/HIP device tensor (no CPU path)")
    if arr.ndim != 2 or not arr.numel():
        raise ToneMappingError(
            f"durand input must be a nonempty HW frame, got {tuple(arr.shape)}")
    if arr.dtype not in (torch.float32, torch.float64):
        raise ToneMappingError(
            f"durand input must be float32 or float64, got {arr.dtype}")
    try:
        x = arr.to(torch.float32)
    except (RuntimeError, OverflowError) as exc:
        raise ToneMappingError(
            f"durand curve input must be representable as float32: {exc}"
        ) from exc
    if not bool(torch.isfinite(x).all()):
        raise ToneMappingError("durand curve input must be finite")
    dev = x.device
    eps = torch.as_tensor(1e-6, dtype=torch.float32, device=dev)
    log_y = reference_log10_float32(x + eps)
    h, w = x.shape[-2], x.shape[-1]
    factor = int(downsample_factor)
    if factor > 1:
        oh = max(1, math.ceil(h / factor))
        ow = max(1, math.ceil(w / factor))
        small = area_float32(log_y, oh, ow)
        gauss_sigma = sigma_space / factor
    else:
        small = log_y
        gauss_sigma = sigma_space
    smooth = _gaussian_float32(small, gauss_sigma)
    diff = small - smooth
    c = torch.as_tensor(sigma_color, dtype=torch.float32, device=dev)
    ratio = diff / c
    arg = -0.5 * (ratio ** 2)
    weight = torch.exp(arg.to(torch.float64)).to(torch.float32)
    base_small = smooth + weight * diff
    if factor > 1:
        base = _linear_native_f32(base_small, h, w)
    else:
        base = base_small
    con = torch.as_tensor(contrast_factor, dtype=torch.float32, device=dev)
    exponent = base / con + (log_y - base)
    power = torch.pow(10.0, exponent.to(torch.float64)).to(torch.float32)
    lo, hi = float(power.min()), float(power.max())
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ToneMappingError("durand output is not finite")
    if hi > lo:
        return ((power - lo) / torch.as_tensor(
            hi - lo, dtype=torch.float32, device=dev)).to(torch.float64)
    return arr.to(torch.float64).clamp(0, 1)
