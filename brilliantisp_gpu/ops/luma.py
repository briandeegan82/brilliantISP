"""GPU shared declared-scale luma helpers.

Mirrors util.luma: declared ``scale`` normalize/restore around float64
plane math and chroma-preserving luminance replacement. The clip dtype
follows ``np.result_type(input dtype, 0.0, scale)`` computed on metadata
only; no image arrays on the CPU. Torch is lazy.
"""

from __future__ import annotations

import math

import numpy as np

from ..types import GpuError, require_torch
from ..ops.array import checked_float32
from ..ops.color import luminance as _luminance

_NP_TO_TORCH = {
    "float16": "float16", "float32": "float32", "float64": "float64",
    "uint8": "uint8", "uint16": "uint16", "uint32": "uint32",
    "int8": "int8", "int16": "int16", "int32": "int32", "int64": "int64",
}
_TORCH_TO_NP = {v: k for k, v in _NP_TO_TORCH.items()}


def _require_scale(scale):
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise ValueError(f"scale must be a positive finite number, got {scale!r}")
    try:
        out = float(scale)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"scale must be a positive finite number, got {scale!r}") from exc
    if not math.isfinite(out) or out <= 0:
        raise ValueError(f"scale must be a positive finite number, got {scale!r}")
    return out


def _plane(name, values, torch):
    arr = torch.as_tensor(values)
    if arr.ndim not in (2, 3) or not arr.numel():
        raise GpuError(f"{name} requires a nonempty HW or BHW plane, got {tuple(arr.shape)}")
    if arr.dtype == torch.bool or arr.is_complex() or \
            (not arr.is_floating_point() and not str(arr.dtype).startswith("torch.int")
             and not str(arr.dtype).startswith("torch.uint")):
        raise GpuError(f"{name} requires real numeric input, got {arr.dtype}")
    if not bool(torch.isfinite(arr.to(torch.float64)).all()):
        raise ValueError(f"{name} must be finite")
    return arr


def _clip_dtype(arr, s, torch):
    np_dt = np.dtype(_TORCH_TO_NP.get(str(arr.dtype).replace("torch.", ""), "float64"))
    result = np.result_type(np_dt, 0.0, s)
    return getattr(torch, _NP_TO_TORCH[str(result)])


def normalize_plane(values, *, scale):
    """Real HW/BHW -> float64 clipped to [0, scale] / scale."""
    torch = require_torch()
    s = _require_scale(scale)
    arr = _plane("luma plane", values, torch)
    clip_dt = _clip_dtype(arr, s, torch)
    clipped = arr.to(clip_dt).clamp(0.0, s)
    out = clipped.to(torch.float64) / torch.as_tensor(
        s, dtype=torch.float64, device=arr.device)
    return out


def restore_plane(normalized, *, scale):
    """Normalized float output -> checked float32 in declared units."""
    torch = require_torch()
    s = _require_scale(scale)
    arr = _plane("luma restore", normalized, torch)
    out = arr.to(torch.float64) * s
    if not bool(torch.isfinite(out).all()):
        raise ValueError("luma restore produced nonfinite output")
    return checked_float32(out, name="luma_restore")


def replace_luminance(rgb, new_y, *, matrix: str = "bt709"):
    """rgb + (new_y - luminance(rgb)); no gamut clip, f32 out."""
    torch = require_torch()
    arr = torch.as_tensor(rgb)
    if arr.ndim not in (3, 4) or arr.shape[-1] != 3 or not arr.numel():
        raise GpuError(f"rgb must be HWC3 or BHWC3, got {tuple(arr.shape)}")
    if not arr.is_floating_point():
        raise GpuError(f"rgb must be a floating-point array, got {arr.dtype}")
    f = arr.to(torch.float64)
    if not bool(torch.isfinite(f).all()):
        raise ValueError("rgb must be finite")
    src = torch.as_tensor(new_y, device=arr.device)
    if src.dtype == torch.bool or src.is_complex() or (
            not src.is_floating_point()
            and not str(src.dtype).startswith(("torch.int", "torch.uint"))):
        raise ValueError("new_y must be real numeric")
    y = src.to(torch.float64)
    if y.shape != f.shape[:-1] or not bool(torch.isfinite(y).all()):
        raise ValueError("new_y must be finite with the same spatial shape")
    out = f + (y - _luminance(f, matrix=matrix, dtype=np.float64))[..., None]
    return checked_float32(out, name="replace_luminance")
